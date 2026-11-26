//! On-demand, single-flight Chrome profile startup. Status/discovery never launch a browser.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::{Mutex, watch};

use crate::relay::{Relay, State};

const COOLDOWN: Duration = Duration::from_secs(20);
const OPEN_GRACE: Duration = Duration::from_secs(2);
type Outcome = Result<(), String>;

#[derive(Clone)]
pub(crate) struct Config {
    pub enabled: bool,
    pub app: String,
    pub profile: Option<String>,
    pub chrome_root: PathBuf,
    pub timeout: Duration,
}

impl Config {
    fn from_env() -> Self {
        Self::from_lookup(crate::paths::home_dir(), |name| std::env::var(name).ok())
    }

    fn from_lookup(home: PathBuf, lookup: impl Fn(&str) -> Option<String>) -> Self {
        let app = lookup("BROWSE_CHROME_APP").filter(|v| !v.trim().is_empty()).unwrap_or_else(|| "Google Chrome".into());
        let app_name = Path::new(&app).file_stem().and_then(|s| s.to_str()).unwrap_or(&app);
        let directory = match app_name {
            "Google Chrome Beta" => "Chrome Beta",
            "Google Chrome Dev" => "Chrome Dev",
            "Google Chrome Canary" => "Chrome Canary",
            _ => "Chrome",
        };
        let enabled = !lookup("BROWSE_AUTO_LAUNCH").is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false"));
        let timeout = lookup("BROWSE_AUTO_LAUNCH_TIMEOUT_MS").and_then(|v| v.trim().parse::<u64>().ok()).filter(|v| *v > 0).unwrap_or(8000).min(120_000);
        Self {
            enabled,
            app,
            profile: lookup("BROWSE_CHROME_PROFILE").map(|v| v.trim().to_string()).filter(|v| !v.is_empty()),
            chrome_root: home.join("Library/Application Support/Google").join(directory),
            timeout: Duration::from_millis(timeout),
        }
    }
}

#[async_trait]
pub(crate) trait Launcher: Send + Sync {
    fn available(&self) -> bool;
    async fn open(&self, app: &str, profile: &str) -> anyhow::Result<()>;
    async fn fallback(&self, app: &str, profile: &str) -> anyhow::Result<()>;
}

struct MacLauncher;

#[async_trait]
impl Launcher for MacLauncher {
    fn available(&self) -> bool {
        // Unit tests must never open the user's browser. They inject a fake launcher instead.
        cfg!(target_os = "macos") && !cfg!(test)
    }

    async fn open(&self, app: &str, profile: &str) -> anyhow::Result<()> {
        let mut command = tokio::process::Command::new("/usr/bin/open");
        command.args(["-g", "-a", app, "--args", &format!("--profile-directory={profile}")]);
        background(&mut command);
        let status = command.kill_on_drop(true).status().await.context("could not run macOS open")?;
        if !status.success() {
            bail!("macOS open exited with {status}");
        }
        Ok(())
    }

    async fn fallback(&self, app: &str, profile: &str) -> anyhow::Result<()> {
        // LaunchServices may discard --args for an already-running app. Calling its executable
        // reaches Chrome's process singleton and initializes the requested profile instead.
        let bundle = app_bundle(app).context("could not locate the Chrome app bundle for the direct-launch fallback")?;
        let executable = tokio::process::Command::new("/usr/libexec/PlistBuddy")
            .args(["-c", "Print :CFBundleExecutable"])
            .arg(bundle.join("Contents/Info.plist"))
            .output()
            .await
            .context("could not read Chrome's CFBundleExecutable")?;
        if !executable.status.success() {
            bail!("could not read Chrome's CFBundleExecutable");
        }
        let name = String::from_utf8(executable.stdout)?.trim().to_string();
        validate_profile(&name)?;
        let mut command = tokio::process::Command::new(bundle.join("Contents/MacOS").join(name));
        // Initialize the profile/extension without creating or activating a browser window.
        command.args([format!("--profile-directory={profile}"), "--no-first-run".into(), "--no-startup-window".into()]);
        background(&mut command);
        let mut child = command.spawn().context("could not launch Chrome's executable")?;
        // Reap the short-lived singleton client (or Chrome on a cold start) without tying its
        // lifetime to this MCP request. Dropping/cancelling a tool never kills Chrome.
        tokio::spawn(async move {
            let _ = child.wait().await;
        });
        Ok(())
    }
}

fn background(command: &mut tokio::process::Command) {
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    #[cfg(unix)]
    command.process_group(0);
}

fn app_bundle(app: &str) -> Option<PathBuf> {
    let path = Path::new(app);
    if path.is_absolute() {
        return path.is_dir().then(|| path.to_path_buf());
    }
    let filename = if app.ends_with(".app") { app.to_string() } else { format!("{app}.app") };
    [PathBuf::from("/Applications"), crate::paths::home_dir().join("Applications")].into_iter().map(|root| root.join(&filename)).find(|p| p.is_dir())
}

#[derive(Default)]
struct FlightState {
    running: Option<watch::Receiver<Option<Outcome>>>,
    failed: Option<(Instant, String)>,
}

pub(crate) struct AutoLaunch {
    config: Config,
    launcher: Arc<dyn Launcher>,
    flight: Mutex<FlightState>,
}

impl AutoLaunch {
    pub fn from_env() -> Arc<Self> {
        Self::new(Config::from_env(), Arc::new(MacLauncher))
    }

    pub fn new(config: Config, launcher: Arc<dyn Launcher>) -> Arc<Self> {
        Arc::new(Self { config, launcher, flight: Mutex::new(FlightState::default()) })
    }

    pub async fn ensure(self: &Arc<Self>, relay: &Arc<Relay>) -> anyhow::Result<()> {
        if relay.st().connected() || !self.config.enabled || !self.launcher.available() {
            return Ok(());
        }
        // Startup is not installation/pairing, and must not open Chrome for an unpaired relay.
        if crate::pairing::read_browser_pairings(&relay.paths).is_empty() {
            return Ok(());
        }
        let mut flight = self.flight.lock().await;
        if relay.st().connected() {
            return Ok(());
        }
        let mut receiver = if let Some(running) = &flight.running {
            running.clone()
        } else {
            if let Some((at, message)) = &flight.failed
                && at.elapsed() < COOLDOWN
            {
                bail!("{message} Auto-launch is cooling down for {} more seconds; no additional launch was made.", COOLDOWN.saturating_sub(at.elapsed()).as_secs() + 1);
            }
            let (sender, receiver) = watch::channel(None);
            flight.running = Some(receiver.clone());
            let launch = self.clone();
            let relay = relay.clone();
            // The shared attempt outlives the first waiter, including a cancelled MCP request.
            tokio::spawn(async move {
                let deadline = tokio::time::Instant::now() + launch.config.timeout;
                let outcome = match launch.profile(&relay) {
                    Ok(profile) => match tokio::time::timeout_at(deadline, launch.attempt(&relay, &profile)).await {
                        Ok(result) => result.map_err(|e| format!("{e:#}")),
                        Err(_) => Err(launch.timeout_message(&profile)),
                    },
                    Err(error) => Err(format!("{error:#}")),
                };
                let mut flight = launch.flight.lock().await;
                flight.failed = outcome.as_ref().err().map(|message| (Instant::now(), message.clone()));
                let _ = sender.send(Some(outcome));
                flight.running = None;
            });
            receiver
        };
        drop(flight);
        loop {
            let outcome = receiver.borrow_and_update().clone();
            if let Some(outcome) = outcome {
                return outcome.map_err(|message| anyhow!(message));
            }
            receiver.changed().await.context("Chrome auto-launch task ended before reporting its result")?;
        }
    }

    fn profile(&self, relay: &Relay) -> anyhow::Result<String> {
        if let Some(profile) = &self.config.profile {
            validate_profile(profile)?;
            if !self.config.chrome_root.join(profile).is_dir() {
                bail!("Chrome profile {profile:?} does not exist under {}. Check BROWSE_CHROME_PROFILE.", self.config.chrome_root.display());
            }
            return Ok(profile.clone());
        }
        let keys = crate::pairing::read_browser_pairings(&relay.paths).into_keys().collect::<Vec<_>>();
        detect_profile(&self.config.chrome_root, &keys)
    }

    fn timeout_message(&self, profile: &str) -> String {
        format!(
            "Chrome auto-launch was attempted for {profile:?} ({}) but the paired browse extension did not reconnect within {} ms. Verify the browse extension is enabled and paired in that profile; set BROWSE_CHROME_PROFILE if needed.",
            self.config.app,
            self.config.timeout.as_millis()
        )
    }

    async fn attempt(&self, relay: &Arc<Relay>, profile: &str) -> anyhow::Result<()> {
        if relay.st().connected() {
            return Ok(());
        }
        crate::rlog!("browser.auto_launch", app = self.config.app.clone(), profile = profile);
        let opened = self.launcher.open(&self.config.app, profile).await;
        if opened.is_ok() && relay.wait_until(OPEN_GRACE.min(self.config.timeout / 4), State::connected).await {
            crate::rlog!("browser.auto_launch.connected", profile = profile);
            return Ok(());
        }
        if relay.st().connected() {
            return Ok(());
        }
        crate::rlog!("browser.auto_launch.fallback", app = self.config.app.clone(), profile = profile);
        if let Err(fallback) = self.launcher.fallback(&self.config.app, profile).await {
            let open_error = opened.err().map(|e| format!("; macOS open: {e:#}")).unwrap_or_default();
            bail!(
                "Chrome auto-launch was attempted for {profile:?}; direct-launch fallback failed: {fallback:#}{open_error}. Verify the Chrome app and the browse extension in that profile."
            );
        }
        // The outer timeout includes profile detection, launching, fallback and reconnection.
        if relay.wait_until(self.config.timeout, State::connected).await {
            crate::rlog!("browser.auto_launch.connected", profile = profile);
            Ok(())
        } else {
            bail!("{}", self.timeout_message(profile))
        }
    }
}

impl Relay {
    pub(crate) async fn ensure_browser_connected(self: &Arc<Self>) -> anyhow::Result<()> {
        let auto_launch = self.auto_launch.read().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        auto_launch.ensure(self).await
    }
}

fn validate_profile(profile: &str) -> anyhow::Result<()> {
    if profile.is_empty() || matches!(profile, "." | "..") || profile.chars().any(|c| matches!(c, '/' | '\\') || c.is_control()) {
        bail!("BROWSE_CHROME_PROFILE must be a directory name such as Default or Profile 1, not a path");
    }
    Ok(())
}

fn browse_extensions(profile: &Path) -> Vec<String> {
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let store = crate::protocol::store_id();
    let unpacked = crate::protocol::stable_unpacked_origin().trim_start_matches("chrome-extension://").to_string();
    for filename in ["Secure Preferences", "Preferences"] {
        let Some(prefs) = crate::paths::read_json::<Value>(&profile.join(filename)) else { continue };
        let Some(settings) = prefs.pointer("/extensions/settings").and_then(Value::as_object) else { continue };
        for (id, entry) in settings {
            // Secure Preferences wins when an older plain Preferences record also exists.
            if !seen.insert(id.clone()) {
                continue;
            }
            let name = entry.pointer("/manifest/name").and_then(Value::as_str).unwrap_or_default().to_ascii_lowercase();
            let browse = id == &store || id == &unpacked || matches!(name.as_str(), "browse" | "opencode browse");
            let disabled = entry.get("state").and_then(Value::as_u64).is_some_and(|s| s != 1)
                || entry.get("disable_reasons").is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()) || v.as_u64().is_some_and(|n| n != 0));
            if browse && !disabled && !ids.contains(id) {
                ids.push(id.clone());
            }
        }
    }
    ids
}

// Chrome does not publish its directory name in the extension handshake. Its small local
// LevelDB records can identify a pairing UUID, when present uncompressed. Treat this only as
// positive evidence; ambiguous or compressed records require an explicit profile override.
fn contains_pairing(profile: &Path, ids: &[String], keys: &[String]) -> bool {
    for id in ids {
        let Ok(files) = std::fs::read_dir(profile.join("Local Extension Settings").join(id)) else { continue };
        let mut budget = 24 * 1024 * 1024;
        for file in files.flatten() {
            if !matches!(file.path().extension().and_then(|s| s.to_str()), Some("ldb" | "log")) {
                continue;
            }
            let Ok(metadata) = file.metadata() else { continue };
            let size = metadata.len() as usize;
            if size > 8 * 1024 * 1024 || size > budget {
                continue;
            }
            budget -= size;
            let Ok(bytes) = std::fs::read(file.path()) else { continue };
            if keys.iter().any(|key| !key.is_empty() && bytes.windows(key.len()).any(|window| window == key.as_bytes())) {
                return true;
            }
        }
    }
    false
}

fn detect_profile(root: &Path, keys: &[String]) -> anyhow::Result<String> {
    let local = crate::paths::read_json::<Value>(&root.join("Local State")).unwrap_or(Value::Null);
    let mut names = local.pointer("/profile/info_cache").and_then(Value::as_object).map(|map| map.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
    if let Ok(dirs) = std::fs::read_dir(root) {
        for dir in dirs.flatten() {
            let name = dir.file_name().to_string_lossy().into_owned();
            if (name == "Default" || name.starts_with("Profile ")) && dir.path().is_dir() && !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names.sort();
    let candidates = names
        .into_iter()
        .filter(|name| validate_profile(name).is_ok())
        .filter_map(|name| {
            let ids = browse_extensions(&root.join(&name));
            (!ids.is_empty()).then_some((name, ids))
        })
        .collect::<Vec<_>>();
    let matched = candidates.iter().filter(|(name, ids)| contains_pairing(&root.join(name), ids, keys)).map(|(name, _)| name.clone()).collect::<Vec<_>>();
    if matched.len() == 1 {
        return Ok(matched[0].clone());
    }
    if candidates.len() == 1 {
        return Ok(candidates[0].0.clone());
    }
    let candidates = candidates.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(", ");
    bail!(
        "Could not uniquely detect the paired Chrome profile under {} (enabled browse profiles: {}). Set BROWSE_CHROME_PROFILE to its directory name, and verify the browse extension is enabled there.",
        root.display(),
        if candidates.is_empty() { "none" } else { &candidates }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("browse-autolaunch-{}", crate::util::random_hex(6)));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn profile(&self, name: &str, disabled: bool) {
            let path = self.0.join(name);
            std::fs::create_dir_all(&path).unwrap();
            crate::paths::write_private_json(
                &path.join("Secure Preferences"),
                &json!({ "extensions": { "settings": {
                crate::protocol::PUBLIC_STORE_ID: { "manifest": { "name": "OpenCode Browse" }, "disable_reasons": if disabled { vec![1] } else { vec![] } }
            } } }),
            )
            .unwrap();
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn environment_defaults_and_overrides() {
        let defaults = Config::from_lookup(PathBuf::from("/home/test"), |_| None);
        assert!(defaults.enabled);
        assert_eq!(defaults.app, "Google Chrome");
        assert_eq!(defaults.profile, None);
        assert_eq!(defaults.timeout, Duration::from_secs(8));
        for disabled in ["0", "false", " FALSE "] {
            let config = Config::from_lookup(PathBuf::from("/home/test"), |name| match name {
                "BROWSE_AUTO_LAUNCH" => Some(disabled.into()),
                "BROWSE_CHROME_APP" => Some("Google Chrome Beta".into()),
                "BROWSE_CHROME_PROFILE" => Some("Profile 1".into()),
                "BROWSE_AUTO_LAUNCH_TIMEOUT_MS" => Some("1234".into()),
                _ => None,
            });
            assert!(!config.enabled);
            assert!(config.chrome_root.ends_with("Google/Chrome Beta"));
            assert_eq!(config.profile.as_deref(), Some("Profile 1"));
            assert_eq!(config.timeout, Duration::from_millis(1234));
        }
        for invalid in ["0", "-1", "bad"] {
            let config = Config::from_lookup(PathBuf::from("/home/test"), |name| (name == "BROWSE_AUTO_LAUNCH_TIMEOUT_MS").then(|| invalid.into()));
            assert_eq!(config.timeout, Duration::from_secs(8));
        }
    }

    #[test]
    fn profile_detection_prefers_pairing_evidence_and_never_guesses_between_profiles() {
        let root = Temp::new();
        root.profile("Default", false);
        root.profile("Profile 1", true);
        assert_eq!(detect_profile(&root.0, &[]).unwrap(), "Default");
        root.profile("Profile 1", false);
        let error = detect_profile(&root.0, &[]).unwrap_err().to_string();
        assert!(error.contains("BROWSE_CHROME_PROFILE") && error.contains("Default") && error.contains("Profile 1"));
        let storage = root.0.join("Profile 1/Local Extension Settings").join(crate::protocol::PUBLIC_STORE_ID);
        std::fs::create_dir_all(&storage).unwrap();
        std::fs::write(storage.join("000001.log"), b"local record: paired-profile-uuid").unwrap();
        assert_eq!(detect_profile(&root.0, &["paired-profile-uuid".into()]).unwrap(), "Profile 1");
    }

    #[test]
    fn local_state_and_unpacked_extension_are_supported() {
        let root = Temp::new();
        let profile = root.0.join("Work");
        std::fs::create_dir_all(&profile).unwrap();
        crate::paths::write_private_json(&root.0.join("Local State"), &json!({ "profile": { "info_cache": { "Work": { "name": "My Work Profile" } } } })).unwrap();
        crate::paths::write_private_json(
            &profile.join("Preferences"),
            &json!({ "extensions": { "settings": {
            "unpacked-id": { "manifest": { "name": "browse" }, "state": 1 }
        } } }),
        )
        .unwrap();
        assert_eq!(detect_profile(&root.0, &[]).unwrap(), "Work");
    }

    #[test]
    fn profile_paths_and_control_characters_are_rejected() {
        for valid in ["Default", "Profile 1", "Work Profile"] {
            assert!(validate_profile(valid).is_ok());
        }
        for invalid in ["", ".", "..", "../Default", "/Default", "Profile\\1", "Default\n", "Default\0"] {
            assert!(validate_profile(invalid).is_err());
        }
    }
}
