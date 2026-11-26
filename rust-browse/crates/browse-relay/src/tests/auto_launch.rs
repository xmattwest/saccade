//! Auto-launch against the real relay / fake extension protocol, without opening Chrome.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::auto_launch::{AutoLaunch, Config, Launcher};

#[derive(Default)]
struct FakeLauncher {
    opened: AtomicUsize,
    fallback: AtomicUsize,
    launched: Notify,
    fell_back: Notify,
}

#[async_trait::async_trait]
impl Launcher for FakeLauncher {
    fn available(&self) -> bool {
        true
    }
    async fn open(&self, app: &str, profile: &str) -> anyhow::Result<()> {
        assert_eq!(app, "Google Chrome");
        assert_eq!(profile, "Default");
        self.opened.fetch_add(1, Ordering::SeqCst);
        self.launched.notify_one();
        Ok(())
    }
    async fn fallback(&self, _app: &str, _profile: &str) -> anyhow::Result<()> {
        self.fallback.fetch_add(1, Ordering::SeqCst);
        self.fell_back.notify_one();
        Ok(())
    }
}

fn configure(h: &Harness, timeout: Duration, enabled: bool) -> Arc<FakeLauncher> {
    std::fs::create_dir_all(h.paths.data.join("Default")).unwrap();
    let launcher = Arc::new(FakeLauncher::default());
    let config = Config { enabled, app: "Google Chrome".into(), profile: Some("Default".into()), chrome_root: h.paths.data.clone(), timeout };
    *h.handle.relay.auto_launch.write().unwrap() = AutoLaunch::new(config, launcher.clone());
    launcher
}

async fn reconnect(h: &Harness) -> mpsc::UnboundedSender<Value> {
    let socket = pair_with_token(h.handle.port(), &h.paths, "auto-launch-profile").await;
    let (_, events) = fake_extension(socket, 500);
    events.send(json!({ "method": "ready" })).unwrap();
    eventually("the launched extension to connect", || h.handle.relay.st().connected()).await;
    events
}

#[tokio::test]
async fn parallel_tool_calls_share_one_launch_and_continue() {
    let h = harness().await;
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let launcher = configure(&h, Duration::from_secs(3), true);
    let mut calls = Vec::new();
    for _ in 0..12 {
        let port = h.handle.port();
        let token = h.handle.token().to_string();
        calls.push(tokio::spawn(async move { http(port, "POST", "/tools/call", Some(&token), Some(json!({ "name": "tabs", "arguments": { "action": "new" } }))).await }));
    }
    launcher.launched.notified().await;
    let _events = reconnect(&h).await;
    for call in calls {
        let (status, output) = call.await.unwrap();
        assert_eq!(status, 200);
        assert_ne!(output["isError"], true, "{output}");
    }
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1);
    assert_eq!(launcher.fallback.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancelling_first_waiter_does_not_cancel_or_duplicate_the_launch() {
    let h = harness().await;
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let launcher = configure(&h, Duration::from_secs(3), true);
    let relay = h.handle.relay.clone();
    let first = tokio::spawn(async move { relay.ensure_browser_connected().await });
    launcher.launched.notified().await;
    first.abort();
    let relay = h.handle.relay.clone();
    let second = tokio::spawn(async move { relay.ensure_browser_connected().await });
    let _events = reconnect(&h).await;
    second.await.unwrap().unwrap();
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn timeout_has_profile_advice_and_failed_attempts_cool_down() {
    let h = harness().await;
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let launcher = configure(&h, Duration::from_millis(100), true);
    let error = h.handle.relay.ensure_browser_connected().await.unwrap_err().to_string();
    assert!(error.contains("Default") && error.contains("100 ms") && error.contains("enabled and paired"), "{error}");
    let started = Instant::now();
    let second = h.handle.relay.ensure_browser_connected().await.unwrap_err().to_string();
    assert!(second.contains("cooling down"), "{second}");
    assert!(started.elapsed() < Duration::from_millis(50));
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1);
    assert_eq!(launcher.fallback.load(Ordering::SeqCst), 1);
    let _events = reconnect(&h).await;
    h.handle.relay.ensure_browser_connected().await.unwrap();
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1, "a connection wins even during cooldown");
}

#[tokio::test]
async fn fallback_can_reconnect_when_open_ignores_profile_arguments() {
    let h = harness().await;
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let launcher = configure(&h, Duration::from_secs(3), true);
    let relay = h.handle.relay.clone();
    let call = tokio::spawn(async move { relay.ensure_browser_connected().await });
    launcher.fell_back.notified().await;
    let _events = reconnect(&h).await;
    call.await.unwrap().unwrap();
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1);
    assert_eq!(launcher.fallback.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn disabled_unpaired_and_already_connected_relays_do_not_launch() {
    let h = harness().await;
    let launcher = configure(&h, Duration::from_secs(1), true);
    h.handle.relay.ensure_browser_connected().await.unwrap();
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 0, "unpaired");
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let disabled = configure(&h, Duration::from_secs(1), false);
    h.handle.relay.ensure_browser_connected().await.unwrap();
    assert_eq!(disabled.opened.load(Ordering::SeqCst), 0, "disabled");
    let enabled = configure(&h, Duration::from_secs(1), true);
    let _events = reconnect(&h).await;
    h.handle.relay.ensure_browser_connected().await.unwrap();
    assert_eq!(enabled.opened.load(Ordering::SeqCst), 0, "already connected");
}

struct ProbeTool(&'static str);
#[async_trait::async_trait]
impl browse_core::Tool for ProbeTool {
    fn spec(&self) -> browse_core::ToolSpec {
        browse_core::ToolSpec { name: self.0.into(), description: String::new(), input_schema: json!({ "type": "object" }) }
    }
    async fn call(&self, _ctx: ToolContext, _args: Value) -> anyhow::Result<browse_core::ToolOutput> {
        panic!("an offline call must wait for reconnection before executing this tool");
    }
}

#[tokio::test]
async fn all_requested_tool_entrypoints_wait_but_status_does_not_launch() {
    let h = harness().await;
    pairing::issue_browser_token(&h.paths, "auto-launch-profile", None, None).unwrap();
    let launcher = configure(&h, Duration::from_millis(100), true);
    let (_, status) = http(h.handle.port(), "POST", "/tools/call", Some(h.handle.token()), Some(json!({ "name": "status", "arguments": {} }))).await;
    assert_ne!(status["isError"], true);
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 0);
    for name in ["execute", "saccade_act", "saccade_extract", "saccade_sweep"] {
        h.handle.relay.tools.write().unwrap().push(Arc::new(ProbeTool(name)));
    }
    for name in ["execute", "tabs", "saccade_act", "saccade_extract", "saccade_sweep"] {
        let (_, output) = http(h.handle.port(), "POST", "/tools/call", Some(h.handle.token()), Some(json!({ "name": name, "arguments": { "tab": "restored-tab" } }))).await;
        assert_eq!(output["isError"], true, "{name}: {output}");
        assert!(
            output["content"][0]["text"].as_str().unwrap().contains("auto-launch") || output["content"][0]["text"].as_str().unwrap().contains("Auto-launch"),
            "{name}: {output}"
        );
    }
    assert_eq!(launcher.opened.load(Ordering::SeqCst), 1, "failure cooldown is shared across tools");
}
