use crate::{
    gpt2::Gpt2,
    models::{self, Cancel, MODELS},
};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use rand::Rng;
use std::{
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::Instant,
};
use tokenizers::Tokenizer;

pub enum Command {
    Load(usize, Cancel, crate::settings::Settings),
    Generate(String, Cancel),
}
pub enum Event {
    Status(String),
    Ready(usize, String),
    Text(String),
    Done(String),
    Error(String),
}
pub struct Worker {
    pub tx: Sender<Command>,
    pub rx: Receiver<Event>,
}
struct Engine {
    model: Backend,
    tokenizer: Tokenizer,
    backend: String,
    cached_tokens: Vec<u32>,
}
/// Keep the selectable implementations behind the same token-level contract.
/// `auto` remains Candle until the real-model Mac benchmark establishes a winner.
enum Backend {
    Candle(Gpt2),
    #[cfg(target_os = "macos")]
    Mlx(crate::mlx_gpt2::MlxGpt2),
}
impl Backend {
    fn config(&self) -> &crate::gpt2::Config {
        match self {
            Self::Candle(m) => &m.config,
            #[cfg(target_os = "macos")]
            Self::Mlx(m) => &m.config,
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Candle(m) => m.reset(),
            #[cfg(target_os = "macos")]
            Self::Mlx(m) => m.reset(),
        }
    }
    fn forward(&mut self, ids: &[u32], past: usize) -> Result<Vec<f32>> {
        match self {
            Self::Candle(m) => Ok(m.forward(ids, past)?.to_vec1::<f32>()?),
            #[cfg(target_os = "macos")]
            Self::Mlx(m) => m.forward(ids, past),
        }
    }
    fn warmup(&mut self, cancel: &Cancel, emit: &impl Fn(Event)) -> Result<f64> {
        let start = Instant::now();
        let token = (self.config().vocab_size - 1) as u32;
        // Exercise short, aligned, and unaligned prefill plus single-token decoding. Distinct Metal
        // pipeline variants compile on demand, so finish them before the composer is enabled.
        for (i, count) in [8usize, 64, 65].into_iter().enumerate() {
            anyhow::ensure!(!models::cancelled(cancel), "Cancelled");
            emit(Event::Status(format!("Preparing GPU kernels… {}/3", i + 1)));
            self.reset();
            let count = count.min(self.config().n_positions.saturating_sub(1));
            let _ = self.forward(&vec![token; count], 0)?;
            let _ = self.forward(&[token], count)?;
        }
        self.reset();
        Ok(start.elapsed().as_secs_f64())
    }
}
impl Engine {
    fn load(
        index: usize,
        settings: &crate::settings::Settings,
        cancel: &Cancel,
        emit: &impl Fn(Event),
    ) -> Result<Self> {
        let dir = MODELS[index].download(settings, cancel, |s| emit(Event::Status(s)))?;
        anyhow::ensure!(!models::cancelled(cancel), "Cancelled");
        emit(Event::Status(format!("Loading {}…", MODELS[index].name)));
        let requested =
            std::env::var("CARTER_BACKEND").unwrap_or_else(|_| settings.backend.clone());
        let requested = requested.trim();
        let (mut model, backend) = if requested == "mlx" {
            #[cfg(target_os = "macos")]
            {
                (
                    Backend::Mlx(crate::mlx_gpt2::MlxGpt2::load(&dir)?),
                    "MLX · FP16".to_string(),
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("MLX requires an Apple Silicon Mac");
            }
        } else {
            anyhow::ensure!(
                matches!(requested, "" | "auto" | "candle"),
                "Unknown inference backend: {requested}"
            );
            #[cfg(target_os = "macos")]
            let (device, dtype, label) = (
                Device::new_metal(0).context("Metal is required. Use an Apple Silicon Mac.")?,
                DType::F16,
                "Candle Metal · FP16",
            );
            #[cfg(not(target_os = "macos"))]
            let (device, dtype, label) = (Device::Cpu, DType::F32, "CPU · development");
            // Owned, checksum-verified cache files are never modified in place by this app.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("model.safetensors")],
                    dtype,
                    &device,
                )?
            };
            (
                Backend::Candle(Gpt2::load(vb, models::load_config(&dir)?)?),
                label.to_string(),
            )
        };
        let warmup = model.warmup(cancel, emit)?;
        crate::network::log(&format!(
            "Inference ready: {backend}; kernel warmup {warmup:.3}s"
        ));
        anyhow::ensure!(!models::cancelled(cancel), "Cancelled");
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            model,
            tokenizer,
            backend,
            cached_tokens: Vec::new(),
        })
    }
    fn generate(&mut self, prompt: &str, cancel: &Cancel, emit: &impl Fn(Event)) -> Result<()> {
        let started = Instant::now();
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let all = encoded.get_ids();
        // Reserve 192 tokens for the reply; keep the newest conversation when context fills.
        let limit = self.model.config().n_positions - 192;
        let ids = &all[all.len().saturating_sub(limit)..];
        let mut past =
            if ids.starts_with(&self.cached_tokens) && ids.len() > self.cached_tokens.len() {
                self.cached_tokens.len()
            } else {
                self.model.reset();
                self.cached_tokens.clear();
                0
            };
        let reused = past;
        let mut input = ids[past..].to_vec();
        let mut generated = Vec::new();
        let mut rng = rand::rng();
        let mut first_token = None;
        let mut decoded_after_first = 0usize;
        let mut decoded = String::new();
        let mut published = String::new();
        for _ in 0..192 {
            if models::cancelled(cancel) {
                break;
            }
            let logits = self.model.forward(&input, past)?;
            past += input.len();
            self.cached_tokens.extend_from_slice(&input);
            let token = sample(&logits, &generated, &mut rng);
            if first_token.is_some() {
                decoded_after_first += 1;
            } else {
                first_token = Some(Instant::now());
            }
            if token == 50256 {
                break;
            }
            generated.push(token);
            decoded = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // A tokenizer may expose only part of a UTF-8 character or stop marker.
            // Hold that suffix until it is stable so streamed text never grows then shrinks.
            let (visible, stop) = stable_reply(&decoded, false);
            if visible != published {
                published = visible.to_owned();
                emit(Event::Text(published.clone()));
            }
            if stop {
                break;
            }
            input = vec![token];
        }
        if !decoded.is_empty() {
            let visible = stable_reply(&decoded, true).0;
            if visible != published {
                emit(Event::Text(visible.to_owned()));
            }
        }
        let now = Instant::now();
        let first_seconds = first_token
            .map(|t| t.duration_since(started).as_secs_f64())
            .unwrap_or(0.0);
        let decode_seconds = first_token
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        let throughput = if decoded_after_first > 0 {
            format!(
                "{:.1} tokens/s decode",
                decoded_after_first as f64 / decode_seconds.max(0.001)
            )
        } else {
            "single token".into()
        };
        crate::network::log(&format!(
            "Inference {}: first token {first_seconds:.3}s, decode {decoded_after_first} tokens in {decode_seconds:.3}s, KV reused {reused} tokens",
            self.backend
        ));
        emit(Event::Done(format!(
            "{} · {} · first token {:.2}s",
            self.backend, throughput, first_seconds
        )));
        Ok(())
    }
}
fn stable_reply(text: &str, complete: bool) -> (&str, bool) {
    const MARKERS: [&str; 2] = ["\nHuman:", "\nUser:"];
    if let Some(end) = MARKERS.iter().filter_map(|m| text.find(m)).min() {
        return (&text[..end], true);
    }
    let text = text.trim_end_matches('\u{fffd}');
    if complete {
        return (text, false);
    }
    let hold = MARKERS
        .iter()
        .flat_map(|m| (1..m.len()).map(move |n| &m[..n]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    (&text[..text.len() - hold], false)
}

fn sample(logits: &[f32], previous: &[u32], rng: &mut impl Rng) -> u32 {
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (i, v / 0.8))
        .collect();
    // Apply the penalty once per recent token ID. Scanning 64 IDs for every one of
    // 50k logits did millions of needless comparisons on the CPU for each token.
    let mut recent: Vec<_> = previous.iter().rev().take(64).copied().collect();
    recent.sort_unstable();
    recent.dedup();
    for id in recent {
        if let Some((_, v)) = scores.get_mut(id as usize) {
            *v = if *v > 0.0 { *v / 1.15 } else { *v * 1.15 };
        }
    }
    scores.select_nth_unstable_by(39, |a, b| b.1.total_cmp(&a.1));
    scores.truncate(40);
    let max = scores.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = scores.iter().map(|s| ((s.1 - max) as f64).exp()).collect();
    let mut pick = rng.random::<f64>() * weights.iter().sum::<f64>();
    for ((id, _), weight) in scores.iter().zip(weights) {
        pick -= weight;
        if pick <= 0.0 {
            return *id as u32;
        }
    }
    scores[0].0 as u32
}
pub fn start(wake: rune::Waker) -> Worker {
    let (tx, commands) = mpsc::channel();
    let (events, rx) = mpsc::channel();
    thread::spawn(move || {
        let emit = |event| {
            if events.send(event).is_ok() {
                wake();
            }
        };
        let mut engine: Option<Engine> = None;
        for command in commands {
            let result = match command {
                Command::Load(index, cancel, settings) => {
                    engine = None; // release old GPU weights before allocating the next model
                    Engine::load(index, &settings, &cancel, &emit).map(|loaded| {
                        emit(Event::Ready(index, loaded.backend.clone()));
                        engine = Some(loaded);
                    })
                }
                Command::Generate(prompt, cancel) => engine
                    .as_mut()
                    .context("Choose a model first")
                    .and_then(|e| e.generate(&prompt, &cancel, &emit)),
            };
            if let Err(e) = result {
                crate::network::record_error(&e);
                emit(Event::Error(format!("{e:#}")));
            }
        }
    });
    Worker { tx, rx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn streaming_holds_incomplete_stop_markers_and_utf8() {
        for suffix in ["\n", "\nH", "\nHuman", "\nU", "\nUser"] {
            let value = format!("Answer{suffix}");
            assert_eq!(stable_reply(&value, false), ("Answer", false));
        }
        assert_eq!(
            stable_reply("Answer\nHuman: Next question", false),
            ("Answer", true)
        );
        assert_eq!(
            stable_reply("Answer\nUser: earlier\nHuman: later", false),
            ("Answer", true)
        );
        assert_eq!(
            stable_reply("Answer\nHowever", false),
            ("Answer\nHowever", false)
        );
        assert_eq!(
            stable_reply("Answer\nHuman", true),
            ("Answer\nHuman", false)
        );
        assert_eq!(stable_reply("A\u{fffd}", false), ("A", false));
        assert_eq!(stable_reply("A🙂", false), ("A🙂", false));
    }
    #[test]
    fn sampler_excludes_logits_outside_top_k() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let scores: Vec<f32> = (0..100).map(|v| v as f32).collect();
        for _ in 0..100 {
            assert!(sample(&scores, &[], &mut rng) >= 60);
        }
    }
}

/// Exercise real model weights and KV caches on the shipping compute backend, without a window.
pub fn smoke(dir: &std::path::Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    let (device, dtype) = (Device::new_metal(0)?, DType::F16);
    #[cfg(not(target_os = "macos"))]
    let (device, dtype) = (Device::Cpu, DType::F32);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, &device)?
    };
    let mut model = Gpt2::load(vb, models::load_config(dir)?)?;
    let full = model.forward(&[1, 2, 3, 4], 0)?.to_vec1::<f32>()?;
    model.reset();
    let _ = model.forward(&[1, 2, 3], 0)?;
    let cached = model.forward(&[4], 3)?.to_vec1::<f32>()?;
    let max = full
        .iter()
        .zip(&cached)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = full
        .iter()
        .chain(&cached)
        .map(|v| v.abs())
        .fold(0.0, f32::max);
    // FP16 Metal reduces in a different order for a 4-token causal prefill than for a
    // 3+1 cached decode, so a one- or two-ULP spread in the largest logits is expected.
    // Scale the allowance with the logits; fp32 has no such excuse.
    let tolerance = if dtype == DType::F16 {
        (scale * 5e-3).max(0.05)
    } else {
        0.05
    };
    anyhow::ensure!(
        full.iter().all(|v| v.is_finite()) && max <= tolerance,
        "Invalid logits or cache mismatch: {max} exceeds {tolerance} (largest logit {scale})"
    );
    println!(
        "Real-weight inference OK: {dtype:?}, {} logits, max KV difference {max} (tolerance {tolerance})",
        full.len()
    );
    Ok(())
}
