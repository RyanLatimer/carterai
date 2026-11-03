//! Fixed-workload inference benchmark. This runs no network requests and never reads chat text.
//! Every token includes a logits readback, so asynchronous GPU work cannot inflate throughput.
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Serialize;
use std::{path::Path, process::Command, time::Instant};

use crate::{gpt2::Gpt2, models};

#[derive(Debug, Clone)]
pub struct Options {
    pub output_tokens: usize,
    pub prompt_tokens: usize,
    pub rounds: usize,
    pub cpu: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            output_tokens: 32,
            prompt_tokens: 32,
            rounds: 3,
            cpu: false,
        }
    }
}

#[derive(Serialize)]
struct Machine {
    os: String,
    architecture: String,
    chip: String,
    memory_bytes: Option<u64>,
    virtualization: String,
    github_actions: bool,
}
#[derive(Serialize)]
struct ModelInfo {
    hidden_size: usize,
    layers: usize,
    attention_heads: usize,
    vocabulary: usize,
    context_window: usize,
}
#[derive(Serialize)]
struct Round {
    prefill_ms: f64,
    time_to_first_token_ms: f64,
    decode_tokens: usize,
    decode_ms: f64,
    decode_tokens_per_second: f64,
    decode_latency_p50_ms: f64,
    decode_latency_p95_ms: f64,
    end_to_end_tokens_per_second: f64,
    token_checksum: u64,
}
#[derive(Serialize)]
struct Report {
    schema_version: u32,
    app_version: &'static str,
    backend: &'static str,
    dtype: String,
    machine: Machine,
    model: ModelInfo,
    model_load_ms: f64,
    prompt_tokens: usize,
    output_tokens: usize,
    cold_start_to_first_token_ms: f64,
    cold_first_round: Round,
    rounds: Vec<Round>,
    median_decode_tokens_per_second: f64,
    median_time_to_first_token_ms: f64,
    notes: &'static str,
}

/// CLI: `--benchmark MODEL_DIR [--tokens N] [--prompt-tokens N] [--rounds N] [--cpu]`.
/// JSON goes to stdout, progress to stderr. MODEL_DIR must contain config.json + model.safetensors.
pub fn run_from_args(args: &[String]) -> Result<()> {
    let (path, options) = parse_args(args)?;
    run(Path::new(path), options)
}
fn parse_args(args: &[String]) -> Result<(&str, Options)> {
    let path = args.first().context(
        "Usage: --benchmark MODEL_DIR [--tokens N] [--prompt-tokens N] [--rounds N] [--cpu]",
    )?;
    let mut options = Options::default();
    let mut i = 1;
    while i < args.len() {
        let name = &args[i];
        if name == "--cpu" {
            options.cpu = true;
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .with_context(|| format!("Missing value for {name}"))?
            .parse::<usize>()
            .with_context(|| format!("Invalid integer for {name}"))?;
        match name.as_str() {
            "--tokens" => options.output_tokens = value,
            "--prompt-tokens" => options.prompt_tokens = value,
            "--rounds" => options.rounds = value,
            _ => bail!("Unknown benchmark option: {name}"),
        }
        i += 2;
    }
    ensure!(
        (2..=512).contains(&options.output_tokens),
        "--tokens must be between 2 and 512"
    );
    ensure!(
        options.prompt_tokens > 0,
        "--prompt-tokens must be greater than 0"
    );
    ensure!(
        (1..=20).contains(&options.rounds),
        "--rounds must be between 1 and 20"
    );
    Ok((path, options))
}

pub fn run(path: &Path, options: Options) -> Result<()> {
    let started = Instant::now();
    #[cfg(target_os = "macos")]
    let (device, dtype, backend) = if options.cpu {
        (Device::Cpu, DType::F32, "Candle CPU")
    } else {
        (
            Device::new_metal(0)
                .context("Metal device unavailable; use --cpu only for development comparisons")?,
            DType::F16,
            "Candle Metal",
        )
    };
    #[cfg(not(target_os = "macos"))]
    let (device, dtype, backend) = {
        let _ = options.cpu;
        (Device::Cpu, DType::F32, "Candle CPU")
    };
    let config = models::load_config(path)?;
    ensure!(
        options.prompt_tokens + options.output_tokens - 1 <= config.n_positions,
        "Prompt + output exceed this model's {}-token context",
        config.n_positions
    );
    let model_info = ModelInfo {
        hidden_size: config.n_embd,
        layers: config.n_layer,
        attention_heads: config.n_head,
        vocabulary: config.vocab_size,
        context_window: config.n_positions,
    };
    // The benchmark only maps caller-selected, read-only model files.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path.join("model.safetensors")], dtype, &device)?
    };
    let mut model = Gpt2::load(vb, config)?;
    device.synchronize()?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let prompt: Vec<u32> = (0..options.prompt_tokens)
        .map(|i| (17 + i * 37) as u32 % model.config.vocab_size as u32)
        .collect();
    eprintln!(
        "Cold trial: {} prompt tokens, {} output tokens",
        prompt.len(),
        options.output_tokens
    );
    let cold_first_round = run_round(&mut model, &prompt, options.output_tokens)?;
    let cold_start_to_first_token_ms = load_ms + cold_first_round.time_to_first_token_ms;
    let mut rounds = Vec::with_capacity(options.rounds);
    for i in 0..options.rounds {
        let result = run_round(&mut model, &prompt, options.output_tokens)?;
        eprintln!(
            "Round {}: prefill {:.1}ms, decode {:.2}tokens/s ({} tokens)",
            i + 1,
            result.prefill_ms,
            result.decode_tokens_per_second,
            result.decode_tokens
        );
        rounds.push(result);
    }
    let median_decode = percentile(
        &rounds
            .iter()
            .map(|r| r.decode_tokens_per_second)
            .collect::<Vec<_>>(),
        0.5,
    );
    let median_ttft = percentile(
        &rounds
            .iter()
            .map(|r| r.time_to_first_token_ms)
            .collect::<Vec<_>>(),
        0.5,
    );
    let report = Report {
        schema_version: 1,
        app_version: env!("CARGO_PKG_VERSION"),
        backend,
        dtype: format!("{dtype:?}"),
        machine: machine(),
        model: model_info,
        model_load_ms: load_ms,
        prompt_tokens: options.prompt_tokens,
        output_tokens: options.output_tokens,
        cold_start_to_first_token_ms,
        cold_first_round,
        rounds,
        median_decode_tokens_per_second: median_decode,
        median_time_to_first_token_ms: median_ttft,
        notes: "Synthetic deterministic prompt; fixed-length greedy generation ignores EOS. A fresh KV cache is used for each round. Model load is reported separately. System shader caches are not purged; cold means first execution in this process. cold_first_round captures the first execution and is excluded from warm medians; each following round starts a fresh KV cache on the already-initialized backend. Prefill includes the full prompt forward pass and logits readback; time to first token also includes greedy selection. Decode includes one-token forward passes, logits readback and greedy selection. No tokenization, UI rendering, or text decoding. GitHub macOS results describe the runner and must not be presented as physical-Mac performance.",
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_round(model: &mut Gpt2, prompt: &[u32], output_tokens: usize) -> Result<Round> {
    model.reset();
    let total = Instant::now();
    let logits = model.forward(prompt, 0)?.to_vec1::<f32>()?;
    let prefill_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut token = argmax(&logits)?;
    let first_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut checksum = token as u64;
    let mut latencies = Vec::with_capacity(output_tokens - 1);
    let decode = Instant::now();
    for i in 1..output_tokens {
        let start = Instant::now();
        let logits = model
            .forward(&[token], prompt.len() + i - 1)?
            .to_vec1::<f32>()?;
        token = argmax(&logits)?;
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        checksum = checksum.wrapping_mul(31).wrapping_add(token as u64);
    }
    let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;
    let total_seconds = total.elapsed().as_secs_f64();
    let decode_tokens = output_tokens - 1;
    Ok(Round {
        prefill_ms,
        time_to_first_token_ms: first_ms,
        decode_tokens,
        decode_ms,
        decode_tokens_per_second: decode_tokens as f64 / (decode_ms / 1000.0),
        decode_latency_p50_ms: percentile(&latencies, 0.5),
        decode_latency_p95_ms: percentile(&latencies, 0.95),
        end_to_end_tokens_per_second: output_tokens as f64 / total_seconds,
        token_checksum: checksum,
    })
}
fn argmax(logits: &[f32]) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "Model returned invalid logits"
    );
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32)
}
fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let position = (values.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * (position - lower as f64)
}
fn output(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|r| r.status.success())
        .and_then(|r| String::from_utf8(r.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}
fn machine() -> Machine {
    #[cfg(target_os = "macos")]
    {
        let hypervisor = output("sysctl", &["-n", "kern.hv_vmm_present"]);
        Machine {
            os: format!("macOS {}", output("sw_vers", &["-productVersion"])),
            architecture: std::env::consts::ARCH.into(),
            chip: output("sysctl", &["-n", "machdep.cpu.brand_string"]),
            memory_bytes: output("sysctl", &["-n", "hw.memsize"]).parse().ok(),
            virtualization: match hypervisor.as_str() {
                "1" => "virtual machine",
                "0" => "physical host",
                _ => "unknown",
            }
            .into(),
            github_actions: std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true"),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Machine {
            os: output("uname", &["-s"]),
            architecture: std::env::consts::ARCH.into(),
            chip: "unknown".into(),
            memory_bytes: None,
            virtualization: "unknown".into(),
            github_actions: std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arguments_reject_zero_work_and_missing_values() {
        let args = |items: &[&str]| items.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        assert!(parse_args(&args(&["model", "--tokens", "1"])).is_err());
        assert!(parse_args(&args(&["model", "--prompt-tokens", "0"])).is_err());
        assert!(parse_args(&args(&["model", "--rounds", "0"])).is_err());
        assert!(parse_args(&args(&["model", "--tokens"])).is_err());
        let values = args(&["model", "--tokens", "64", "--rounds", "2", "--cpu"]);
        let (_, options) = parse_args(&values).unwrap();
        assert_eq!(options.output_tokens, 64);
        assert_eq!(options.rounds, 2);
        assert!(options.cpu);
    }
    #[test]
    fn distribution_and_greedy_selection_are_checked() {
        assert_eq!(percentile(&[1.0, 4.0, 2.0, 3.0], 0.5), 2.5);
        assert!((percentile(&[1.0, 4.0, 2.0, 3.0], 0.95) - 3.85).abs() < 1e-12);
        assert_eq!(argmax(&[0.1, -1.0, 4.0]).unwrap(), 2);
        assert!(argmax(&[f32::NAN]).is_err());
        assert!(argmax(&[]).is_err());
    }
}
