use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub struct Model {
    pub name: &'static str,
    pub id: &'static str,
    pub detail: &'static str,
    pub rev: &'static str,
    sha: &'static str,
    bytes: u64,
}
pub const MODELS: [Model; 3] = [
    Model {
        name: "Dumb",
        id: "gpt2-large",
        detail: "GPT-2 Large · 774M · 3.25 GB",
        rev: "32b71b12589c2f8d625668d2335a01cac3249519",
        sha: "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce",
        bytes: 3247159078,
    },
    Model {
        name: "Dumber",
        id: "gpt2-medium",
        detail: "GPT-2 Medium · 355M · 1.52 GB",
        rev: "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
        sha: "fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd",
        bytes: 1519984962,
    },
    Model {
        name: "Dumbest",
        id: "gpt2",
        detail: "GPT-2 Small · 124M · 548 MB",
        rev: "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        sha: "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
        bytes: 548105171,
    },
];
pub type Cancel = Arc<AtomicBool>;
pub fn cancelled(c: &Cancel) -> bool {
    c.load(Ordering::Relaxed)
}
pub fn cache_dir() -> Result<PathBuf> {
    Ok(directories::ProjectDirs::from("", "", "CarterAI")
        .context("Cannot find application data folder")?
        .data_dir()
        .join("models"))
}
impl Model {
    fn local_dir(&self, settings: &crate::settings::Settings) -> Result<Option<PathBuf>> {
        if !settings.import_folder.trim().is_empty() {
            let root = PathBuf::from(settings.import_folder.trim());
            let path = if root.join("config.json").is_file() {
                root
            } else {
                root.join(self.id)
            };
            anyhow::ensure!(
                path.join("config.json").is_file(),
                "Imported folder needs config.json, tokenizer.json, and model.safetensors for {}",
                self.id
            );
            return Ok(Some(path));
        }
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|p| PathBuf::from(p).join("hub")))
            .or_else(|| {
                directories::BaseDirs::new().map(|d| d.home_dir().join(".cache/huggingface/hub"))
            });
        if let Some(hub) = hub {
            let p = hub
                .join(format!("models--openai-community--{}", self.id))
                .join("snapshots")
                .join(self.rev);
            if ["config.json", "tokenizer.json", "model.safetensors"]
                .iter()
                .all(|f| p.join(f).is_file())
            {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }
    fn verify(&self, path: &Path, cancel: &Cancel, mut status: impl FnMut(String)) -> Result<()> {
        let config = load_config(path)?;
        let expected_layers = match self.id {
            "gpt2-large" => 36,
            "gpt2-medium" => 24,
            _ => 12,
        };
        anyhow::ensure!(
            config.n_layer == expected_layers && config.vocab_size == 50257,
            "Model folder does not match {}",
            self.id
        );
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Invalid tokenizer.json: {e}"))?;
        status(format!("Verifying {}…", self.name));
        let file = path.join("model.safetensors");
        anyhow::ensure!(
            file.metadata()?.len() == self.bytes,
            "Wrong model weight size for {}",
            self.id
        );
        let mut file = fs::File::open(file)?;
        let mut hash = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            if cancelled(cancel) {
                bail!("Cancelled");
            }
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(&buf[..n]);
        }
        anyhow::ensure!(
            format!("{:x}", hash.finalize()) == self.sha,
            "Model checksum mismatch for {}. Remove the corrupted model.safetensors and retry.",
            self.id
        );
        Ok(())
    }
    pub fn download(
        &self,
        settings: &crate::settings::Settings,
        cancel: &Cancel,
        mut status: impl FnMut(String),
    ) -> Result<PathBuf> {
        if let Some(dir) = self.local_dir(settings)? {
            self.verify(&dir, cancel, &mut status)?;
            return Ok(dir);
        }
        let dir = cache_dir()?.join(self.id);
        fs::create_dir_all(&dir)?;
        let lock = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("download.lock"))?;
        fs2::FileExt::try_lock_exclusive(&lock).context(
            "Another Carter process is using this model; wait for its download to finish",
        )?;
        let mut http = crate::network::Http::new(settings)?;
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            if cancelled(cancel) {
                bail!("Cancelled");
            }
            let path = dir.join(file);
            if path.exists()
                && (file != "model.safetensors" || path.metadata()?.len() == self.bytes)
            {
                continue;
            }
            anyhow::ensure!(
                !settings.offline(),
                "{} is not cached. Offline mode prevents downloads; import a complete model folder in Network settings.",
                self.id
            );
            let part = if file == "model.safetensors" {
                dir.join("model.safetensors.incomplete")
            } else {
                path.with_extension("part")
            };
            // Reuse a partial download created by an older Carter build.
            if file == "model.safetensors" && !part.exists() && path.with_extension("part").exists()
            {
                fs::rename(path.with_extension("part"), &part)?;
            }
            let url = format!(
                "{}/openai-community/{}/resolve/{}/{file}",
                settings.endpoint()?,
                self.id,
                self.rev
            );
            let etag_path = part.with_extension("etag");
            let mut last = None;
            for attempt in 0..4 {
                if cancelled(cancel) {
                    bail!("Cancelled");
                }
                status(format!(
                    "Downloading {} · {file}{}",
                    self.name,
                    if attempt > 0 { " (retrying)" } else { "" }
                ));
                let result = download_file(
                    &mut http,
                    &url,
                    &part,
                    &etag_path,
                    if file == "model.safetensors" {
                        Some((self.bytes, self.sha))
                    } else {
                        None
                    },
                    cancel,
                    |count, total| {
                        if file == "model.safetensors" {
                            status(format!(
                                "Downloading {} · {:.0} / {:.0} MB",
                                self.name,
                                count as f64 / 1e6,
                                total as f64 / 1e6
                            ));
                        }
                    },
                );
                match result {
                    Ok(()) => {
                        fs::rename(&part, &path)?;
                        let _ = fs::remove_file(&etag_path);
                        last = None;
                        break;
                    }
                    Err(e) => {
                        crate::network::record_error(&e);
                        last = Some(e);
                        if cancelled(cancel) {
                            bail!("Cancelled");
                        }
                        if attempt < 3 {
                            for _ in 0..(10 * (1 << attempt)) {
                                if cancelled(cancel) {
                                    bail!("Cancelled");
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                            }
                        }
                    }
                }
            }
            if let Some(error) = last {
                return Err(error).with_context(|| format!("Could not download {file}"));
            }
        }
        self.verify(&dir, cancel, &mut status)?;
        Ok(dir)
    }
}

fn download_file(
    http: &mut crate::network::Http,
    url: &str,
    path: &Path,
    etag_path: &Path,
    expected: Option<(u64, &str)>,
    cancel: &Cancel,
    mut progress: impl FnMut(u64, u64),
) -> Result<()> {
    let mut offset = path.metadata().map(|m| m.len()).unwrap_or(0);
    if let Some((size, sha)) = expected
        && offset == size
    {
        if hash_file(path, cancel)? == sha {
            return Ok(());
        }
        fs::remove_file(path)?;
        offset = 0;
    }
    let saved_etag = fs::read_to_string(etag_path).ok();
    let mut response = http.get(
        url,
        (offset > 0).then_some((offset, None)),
        saved_etag.as_deref(),
    )?;
    let partial = response.status().as_u16() == 206;
    let mut total = response
        .content_length()
        .map(|n| n + if partial { offset } else { 0 })
        .unwrap_or(expected.map(|p| p.0).unwrap_or(0));
    if partial {
        let range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok())
            .context("Server omitted Content-Range on resumed download")?;
        let (start, size) = parse_content_range(range)?;
        anyhow::ensure!(start == offset, "Server returned wrong resume offset");
        total = size;
        if let Some(saved) = saved_etag.as_deref()
            && let Some(current) = response
                .headers()
                .get(reqwest::header::ETAG)
                .and_then(|v| v.to_str().ok())
        {
            if saved != current {
                fs::remove_file(path)?;
                let _ = fs::remove_file(etag_path);
                bail!("Server ETag changed; restarting download");
            }
        }
    } else {
        offset = 0;
    } // A 200 response to Range/If-Range means restart, never append.
    if let Some((size, _)) = expected {
        anyhow::ensure!(
            total == 0 || total == size,
            "Model server returned an unexpected weight size"
        );
        total = size;
    }
    if let Some(etag) = response
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
    {
        fs::write(etag_path, etag)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .append(offset > 0)
        .open(path)?;
    let mut count = offset;
    let mut last = count;
    let mut buf = vec![0; 256 * 1024];
    loop {
        if cancelled(cancel) {
            bail!("Cancelled; partial download retained");
        }
        let n = response
            .read(&mut buf)
            .context("Connection interrupted while reading model; partial download retained")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        count += n as u64;
        if total > 0 && count > total {
            file.set_len(0)?;
            bail!("Server sent too much model data");
        }
        if count - last >= 4 * 1024 * 1024 || count == total {
            progress(count, total);
            last = count;
        }
    }
    file.sync_all()?;
    drop(file);
    anyhow::ensure!(
        total == 0 || count == total,
        "Incomplete download ({count}/{total}); retry will resume"
    );
    if let Some((size, sha)) = expected {
        if count != size || hash_file(path, cancel)? != sha {
            fs::remove_file(path)?;
            bail!("Model checksum mismatch; corrupted partial removed for retry");
        }
    } else {
        let _: serde_json::Value = serde_json::from_reader(fs::File::open(path)?)
            .context("Endpoint returned invalid model JSON")?;
    }
    Ok(())
}
fn hash_file(path: &Path, cancel: &Cancel) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        if cancelled(cancel) {
            bail!("Cancelled");
        }
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn parse_content_range(value: &str) -> Result<(u64, u64)> {
    let v = value
        .strip_prefix("bytes ")
        .context("Invalid Content-Range unit")?;
    let (range, total) = v.split_once('/').context("Invalid Content-Range")?;
    let (start, end) = range
        .split_once('-')
        .context("Invalid Content-Range bounds")?;
    let (start, end, total) = (
        start.parse::<u64>()?,
        end.parse::<u64>()?,
        total.parse::<u64>()?,
    );
    anyhow::ensure!(start <= end && end < total, "Invalid Content-Range bounds");
    Ok((start, total))
}
pub fn load_config(path: &Path) -> Result<crate::gpt2::Config> {
    Ok(serde_json::from_reader(fs::File::open(
        path.join("config.json"),
    )?)?)
}
pub fn check_downloads() -> Result<()> {
    let settings = crate::settings::Settings::load();
    let mut http = crate::network::Http::new(&settings)?;
    for model in &MODELS {
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            let url = format!(
                "{}/openai-community/{}/resolve/{}/{file}",
                settings.endpoint()?,
                model.id,
                model.rev
            );
            let mut response = http.get(
                &url,
                if file == "model.safetensors" {
                    Some((0, Some(7)))
                } else {
                    None
                },
                None,
            )?;
            if file == "model.safetensors" {
                let mut bytes = [0; 8];
                response.read_exact(&mut bytes)?;
                anyhow::ensure!(
                    (1..10_000_000).contains(&u64::from_le_bytes(bytes)),
                    "Invalid safetensors header"
                );
            } else {
                let _: serde_json::Value = serde_json::from_reader(response)?;
            }
            println!("Download check OK: {} {file}", model.id);
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "download_tests.rs"]
mod tests;
