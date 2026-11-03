# Carter

A minimalist, local AI chat app for Apple Silicon. Rust, Anomaly's [Rune](https://github.com/anomalyco/rune), native `rune-kit` OpenCode components, and the official `rune-theme-opencode` theme. No webview, cloud inference, API keys, analytics, or accounts.

## Download

Get **Carter-Apple-Silicon.zip** from [Releases](../../releases/latest), unzip, and drag **Carter.app** to Applications. Requires an M1 or newer Mac running macOS 14+. See [installation notes](RELEASE.md) for first-launch Gatekeeper instructions; builds are ad-hoc signed and not notarized.

## Models

- **Dumb** — GPT-2 Large (774M), default; 3.25 GB download.
- **Dumber** — GPT-2 Medium (355M); optional 1.52 GB download.
- **Dumbest** — GPT-2 Small (124M); optional 548 MB download.

Large downloads on first launch. Use the model name in the header to choose another. Download progress, Stop, and Retry are built in. Interrupted downloads restart on retry. Cached models work offline. Weights live in `~/Library/Application Support/CarterAI/models` on macOS; delete a model folder to reclaim space. The app never transmits chat text. New chat clears the conversation; chats are not saved to disk.

GPT-2 completes text rather than following instructions. The chat prompt uses Human/Assistant turns, a sliding 832-token input window, and up to 192 output tokens. Expect strange or inaccurate replies; this is a playful GPT-2 app, not a modern assistant.

## Performance

Metal GPU matmuls, FP16 weights/activations, FP32 attention softmax, per-layer KV caching, tied output embeddings, and Accelerate for CPU operations. Loading and token generation run on a dedicated worker so the UI stays responsive; the worker wakes Rune only when state changes. Top-k sampling (40), temperature 0.8, repetition penalty 1.15. No performance numbers are claimed: the app reports measured generation tokens/s after each reply. Recommend 8 GB or more unified memory. Large uses roughly 1.6 GB for FP16 weights plus runtime, caches, and temporary loading memory.

Model downloads use pinned public `openai-community` Hugging Face revisions and verify weight size and SHA-256 before atomic installation. Original model files stay in FP32 on disk. Optional variants download only when selected.

## Build

On Apple Silicon with Xcode Command Line Tools and current stable Rust:

```sh
cargo test --locked
cargo build --release --locked
swift scripts/icon.swift
scripts/bundle.sh
open dist/Carter.app
```

GitHub Actions builds on the public `macos-14` ARM64 runner, tests, creates an ad-hoc signed `.app`, and uploads a zip. Version tags (`v*`) publish the zip as a release download. Dependencies and Rune revision are pinned by Cargo.lock. On Linux, `cargo test` exercises a CPU backend; the shipping Mac build requires Metal.

## Credits & license

MIT. Rune and its OpenCode theme/components are MIT by Anomaly; embedded Inter and IBM Plex fonts retain their upstream OFL licenses. GPT-2 weights are by OpenAI, distributed by `openai-community` under MIT. See [THIRD_PARTY.md](THIRD_PARTY.md).
