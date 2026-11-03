#!/usr/bin/env bash
# Build the original shipping Candle0.9 GPT-2 without making another checkout.
set -euo pipefail
project_root="$(git rev-parse --show-toplevel)"
baseline_ref="${CARTER_BASELINE_REF:-da4fea5}"
baseline_dir="$(mktemp -d "${TMPDIR:-/tmp}/carter-perf-baseline.XXXXXX")"
trap 'rm -rf "$baseline_dir"' EXIT
mkdir -p "$baseline_dir/src"
git -C "$project_root" show "$baseline_ref:src/gpt2.rs" > "$baseline_dir/src/gpt2.rs"
cp "$project_root/src/benchmark.rs" "$baseline_dir/src/benchmark.rs"
cat > "$baseline_dir/Cargo.toml" <<'MANIFEST'
[package]
name = "carter-perf-baseline"
version = "0.1.0"
edition = "2024"
[dependencies]
anyhow = "1"
candle-core = "=0.9.2"
candle-nn = "=0.9.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { version = "=0.9.2", features = ["metal", "accelerate"] }
candle-nn = { version = "=0.9.2", features = ["metal", "accelerate"] }
accelerate-src = "0.3"
[profile.release]
lto = "thin"
codegen-units = 1
strip = true
MANIFEST
cat > "$baseline_dir/src/main.rs" <<'RUST'
mod benchmark;
mod gpt2;
mod models {
    pub fn load_config(path: &std::path::Path) -> anyhow::Result<crate::gpt2::Config> {
        Ok(serde_json::from_reader(std::fs::File::open(path.join("config.json"))?)?)
    }
}
fn main() {
    #[cfg(target_os = "macos")]
    { extern crate accelerate_src; }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = benchmark::run_from_args(&args) {
        eprintln!("{error:#}"); std::process::exit(1);
    }
}
RUST
cargo build --release --manifest-path "$baseline_dir/Cargo.toml" --target-dir "$project_root/target/perf-baseline"
mkdir -p "$project_root/target/perf"
cp "$project_root/target/perf-baseline/release/carter-perf-baseline" "$project_root/target/perf/carter-baseline"
printf 'Baseline %s ready: target/perf/carter-baseline\n' "$baseline_ref"
