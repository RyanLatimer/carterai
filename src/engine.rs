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
    Load(usize, Cancel),
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
    model: Gpt2,
    tokenizer: Tokenizer,
    backend: String,
}
impl Engine {
    fn load(index: usize, cancel: &Cancel, emit: &impl Fn(Event)) -> Result<Self> {
        let dir = MODELS[index].download(cancel, |s| emit(Event::Status(s)))?;
        anyhow::ensure!(!models::cancelled(cancel), "Cancelled");
        emit(Event::Status(format!("Loading {}…", MODELS[index].name)));
        #[cfg(target_os = "macos")]
        let (device, dtype, backend) = (
            Device::new_metal(0).context("Metal is required. Use an Apple Silicon Mac.")?,
            DType::F16,
            "Metal · FP16",
        );
        #[cfg(not(target_os = "macos"))]
        let (device, dtype, backend) = (Device::Cpu, DType::F32, "CPU · development");
        // Owned, checksum-verified cache files are never modified in place by this app.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], dtype, &device)?
        };
        let model = Gpt2::load(vb, models::load_config(&dir)?)?;
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            model,
            tokenizer,
            backend: backend.into(),
        })
    }
    fn generate(&mut self, prompt: &str, cancel: &Cancel, emit: &impl Fn(Event)) -> Result<()> {
        self.model.reset();
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let all = encoded.get_ids();
        // Reserve 192 tokens for the reply; keep the newest conversation when context fills.
        let limit = self.model.config.n_positions - 192;
        let ids = &all[all.len().saturating_sub(limit)..];
        let mut input = ids.to_vec();
        let mut past = 0;
        let mut generated = Vec::new();
        let mut rng = rand::rng();
        let started = Instant::now();
        for _ in 0..192 {
            if models::cancelled(cancel) {
                break;
            }
            let logits = self.model.forward(&input, past)?.to_vec1::<f32>()?;
            past += input.len();
            let token = sample(&logits, &generated, &mut rng);
            if token == 50256 {
                break;
            }
            generated.push(token);
            let mut text = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if let Some(end) = text.find("\nHuman:").or_else(|| text.find("\nUser:")) {
                text.truncate(end);
                emit(Event::Text(text));
                break;
            }
            // decode the whole prefix so incomplete UTF-8 isn't appended as broken pieces.
            emit(Event::Text(text));
            input = vec![token];
        }
        emit(Event::Done(format!(
            "{} · {:.1} tokens/s",
            self.backend,
            generated.len() as f64 / started.elapsed().as_secs_f64().max(0.001)
        )));
        Ok(())
    }
}
fn sample(logits: &[f32], previous: &[u32], rng: &mut impl Rng) -> u32 {
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, mut v)| {
            if previous.iter().rev().take(64).any(|&t| t as usize == i) {
                v = if v > 0.0 { v / 1.15 } else { v * 1.15 };
            }
            (i, v / 0.8)
        })
        .collect();
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
                Command::Load(index, cancel) => {
                    engine = None; // release old GPU weights before allocating the next model
                    Engine::load(index, &cancel, &emit).map(|loaded| {
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
        .zip(cached)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    anyhow::ensure!(
        full.iter().all(|v| v.is_finite()) && max < 0.05,
        "Invalid logits or cache mismatch: {max}"
    );
    println!(
        "Real-weight inference OK: {dtype:?}, {} logits, max KV difference {max}",
        full.len()
    );
    Ok(())
}
