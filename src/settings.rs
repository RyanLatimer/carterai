use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub http_proxy: String,
    pub https_proxy: String,
    pub no_proxy: String,
    pub endpoint: String,
    pub offline: bool,
    pub backend: String,
    pub import_folder: String,
    #[serde(skip)]
    pub token: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            endpoint: DEFAULT_ENDPOINT.into(),
            offline: false,
            backend: "auto".into(),
            import_folder: String::new(),
            token: String::new(),
        }
    }
}
pub fn data_dir() -> Result<PathBuf> {
    Ok(directories::ProjectDirs::from("", "", "CarterAI")
        .context("Cannot find application data folder")?
        .data_dir()
        .to_owned())
}
impl Settings {
    pub fn load() -> Self {
        #[allow(unused_mut)]
        let mut s: Self = data_dir()
            .ok()
            .and_then(|d| fs::read(d.join("settings.json")).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        #[cfg(target_os = "macos")]
        {
            s.token = security_framework::passwords::get_generic_password(
                "app.carter.localchat.hf-token",
                "Carter",
            )
            .ok()
            .and_then(|v| String::from_utf8(v).ok())
            .unwrap_or_default();
        }
        s
    }
    pub fn save(&self) -> Result<()> {
        self.validate()?;
        let dir = data_dir()?;
        fs::create_dir_all(&dir)?;
        let path = dir.join("settings.json.tmp");
        let mut options = fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        use std::io::Write;
        let mut out = options.open(&path)?;
        out.write_all(&serde_json::to_vec_pretty(self)?)?;
        out.sync_all()?;
        drop(out);
        fs::rename(path, dir.join("settings.json"))?;
        #[cfg(target_os = "macos")]
        {
            if self.token.trim().is_empty() {
                let _ = security_framework::passwords::delete_generic_password(
                    "app.carter.localchat.hf-token",
                    "Carter",
                );
            } else {
                security_framework::passwords::set_generic_password(
                    "app.carter.localchat.hf-token",
                    "Carter",
                    self.token.trim().as_bytes(),
                )
                .context("Could not store Hugging Face token in macOS Keychain")?;
            }
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        validate_endpoint(self.endpoint.trim())?;
        for p in [&self.http_proxy, &self.https_proxy] {
            if !p.trim().is_empty() {
                crate::network::parse_proxy(p.trim())?;
            }
        }
        Ok(())
    }
    pub fn endpoint(&self) -> Result<String> {
        let value = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| self.endpoint.clone());
        let url = validate_endpoint(value.trim())?;
        Ok(url.as_str().trim_end_matches('/').into())
    }
    pub fn token(&self) -> Option<String> {
        std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| (!self.token.trim().is_empty()).then(|| self.token.trim().into()))
    }
    pub fn offline(&self) -> bool {
        self.offline
            || std::env::var("HF_HUB_OFFLINE")
                .is_ok_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
    }
}
fn validate_endpoint(value: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(value).context("Enter a valid Hugging Face endpoint URL")?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "Endpoint must use HTTP or HTTPS"
    );
    anyhow::ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "Endpoint must not contain credentials, query parameters, or a fragment"
    );
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn secrets_are_not_serialized_and_endpoint_is_validated() {
        let s = Settings {
            token: "hf_test_secret".into(),
            ..Settings::default()
        };
        assert!(
            !serde_json::to_string(&s)
                .unwrap()
                .contains("hf_test_secret")
        );
        assert!(validate_endpoint("https://host.example/mirror").is_ok());
        assert!(validate_endpoint("https://user:password@host.example").is_err());
        assert!(validate_endpoint("file:///tmp/models").is_err());
    }
}
