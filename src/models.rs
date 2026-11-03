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
    pub fn download(&self, cancel: &Cancel, mut status: impl FnMut(String)) -> Result<PathBuf> {
        let dir = cache_dir()?.join(self.id);
        fs::create_dir_all(&dir)?;
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(7200))
            .build()?;
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            if cancelled(cancel) {
                bail!("Cancelled");
            }
            let path = dir.join(file);
            // Only atomically renamed, fully verified weights are considered cached.
            if path.exists()
                && (file != "model.safetensors" || path.metadata()?.len() == self.bytes)
            {
                continue;
            }
            let part = path.with_extension("part");
            let result = (|| -> Result<()> {
                let url = format!(
                    "https://huggingface.co/openai-community/{}/resolve/{}/{}",
                    self.id, self.rev, file
                );
                let mut response = client.get(url).send()?.error_for_status()?;
                let total = response
                    .content_length()
                    .unwrap_or(if file == "model.safetensors" {
                        self.bytes
                    } else {
                        0
                    });
                let mut out = fs::File::create(&part)?;
                let mut sha = Sha256::new();
                let (mut count, mut reported) = (0u64, 0u64);
                let mut buf = [0u8; 256 * 1024];
                loop {
                    if cancelled(cancel) {
                        bail!("Cancelled");
                    }
                    let n = response.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n])?;
                    sha.update(&buf[..n]);
                    count += n as u64;
                    if count - reported >= 4 * 1024 * 1024 || count == total {
                        status(format!(
                            "Downloading {} · {:.0} / {:.0} MB",
                            self.name,
                            count as f64 / 1e6,
                            total as f64 / 1e6
                        ));
                        reported = count;
                    }
                }
                if file == "model.safetensors"
                    && (count != self.bytes || format!("{:x}", sha.finalize()) != self.sha)
                {
                    bail!("Download checksum mismatch. Please retry.");
                }
                out.sync_all()?;
                drop(out);
                fs::rename(&part, &path)?;
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&part);
            }
            result.with_context(|| format!("Could not download {file}"))?;
        }
        Ok(dir)
    }
}

pub fn load_config(path: &Path) -> Result<crate::gpt2::Config> {
    Ok(serde_json::from_reader(fs::File::open(
        path.join("config.json"),
    )?)?)
}
