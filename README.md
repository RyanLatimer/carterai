# Carter

Minimal local GPT-2 chat for Apple Silicon, written in Rust using Anomaly's Rune and native OpenCode components. No webview, account, telemetry, or cloud inference.

## Download

Grab the latest build from [Releases](https://github.com/RyanLatimer/carterai/releases). Unzip `Carter-Apple-Silicon.zip` and move `Carter.app` to Applications. Requires an Apple Silicon Mac (M1 or newer) and macOS 14+.

There is no Apple Developer ID for this project, so the download is ad-hoc signed and not notarized. On first launch, use System Settings → Privacy & Security → Open Anyway. If macOS reports the app is damaged, run `xattr -dr com.apple.quarantine /Applications/Carter.app` first, then open it. Only do that for the copy downloaded from this repository.

The empty state shows only “A little less intelligent.” Streaming messages wrap, do not animate their layout, and respect a reader's scroll position.

## Models

- **Dumb** — GPT-2 Large (774M), default; 3.25 GB download.
- **Dumber** — GPT-2 Medium (355M); optional 1.52 GB.
- **Dumbest** — GPT-2 Small (124M); optional 548 MB.

GPT-2 is a 2019 completion model. Expect strange answers. Chats are held in memory only.

## Inference

Two native FP16 GPU implementations: Candle 0.11.1 Metal (dedicated GEMV and fused attention) and MLX. Automatic uses Candle Metal, which measured roughly 4× MLX's decode throughput in the CI benchmark; MLX remains selectable in Settings. Kernels warm on the worker before the composer becomes ready. Both paths use fixed-capacity KV caches, and unchanged conversation prefixes are reused across turns. The status reports first-token time separately from steady decode throughput. Benchmarks report cold first execution and warm trials separately; see [performance tooling](scripts/perf/README.md).

## Network & offline use

Settings provides HTTP/HTTPS proxies, NO_PROXY exceptions, HF_ENDPOINT, an optional HF_TOKEN, offline mode, a model-folder picker, and diagnostic access. On macOS, the token is saved in Keychain, never settings.json. Environment variables override saved values:

- `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, and `NO_PROXY` (lowercase aliases accepted).
- `HF_ENDPOINT`, `HF_TOKEN`, and `HF_HUB_OFFLINE=1`.
- `CARTER_BACKEND=candle` or `mlx`.

With no override, CFNetwork resolves the system route for each URL, including static HTTP CONNECT/SOCKS proxies, exclusions, PAC, and the OS's auto-discovery result. PAC failures are reported rather than silently bypassed. HTTPS uses macOS trust. Authentication is attached only to the configured endpoint origin and stripped on cross-origin redirects.

Downloads follow redirects explicitly and log their paths. Interrupted weight downloads resume with Range/If-Range, validate Content-Range and ETag changes, retry with backoff, and verify pinned SHA-256 before installation. Complete imported models are verified too. Carter's cache is `~/Library/Application Support/CarterAI/models`; partial downloads remain there for retry. Carter does not create Hugging Face `.no_exist` markers. Complete pinned Hugging Face snapshot folders are discovered under HF_HUB_CACHE, HF_HOME/hub, or ~/.cache/huggingface/hub. Existing incomplete blobs belonging to another downloader are not modified.

For offline use, select a folder containing the original config.json, tokenizer.json, and model.safetensors, or use a cached model. Offline mode prevents all app download requests. `network.log` is bounded and credential-redacted; no chats are logged.

## Build & validation

Apple Silicon, current stable Rust, Xcode 26.3 Metal compiler:

```sh
export MLX_RS_METAL_PATH="$PWD/target/mlx-metallib"
cargo test --locked
cargo build --release --locked
swift scripts/icon.swift
DEVELOPMENT_BUILD=1 scripts/bundle.sh
open dist/Carter.app
```

The MLX shader library must be copied into Contents/Resources; the app resolves that bundle path, not a build-machine path. GitHub Actions tests numerical parity, actual Hugging Face access, fixed-workload cold/warm timings, bundle resources, and LaunchServices launches through system-only static and PAC proxies with proxy environment variables removed.

Development builds are ad-hoc signed and **not notarized**. A Developer ID is not available for this project. Development tags (`v*-dev`) run the full test suite, package the app, and publish a prerelease automatically without signing secrets. Normal `v*` releases fail closed unless all five encrypted Actions secrets are configured: DEVELOPER_ID_P12_BASE64, DEVELOPER_ID_P12_PASSWORD, NOTARY_KEY_BASE64 (base64 `.p8`), NOTARY_KEY_ID, and NOTARY_ISSUER_ID. Releases use hardened-runtime Developer ID signatures, notarytool submission, stapling, and spctl assessment before publishing. The bundle identifier remains `app.carter.localchat`.

## Credits

MIT. Rune/OpenCode components by Anomaly; Candle by Hugging Face; MLX and its GPT-2 architecture by Apple; GPT-2 weights by OpenAI. Embedded fonts retain their OFL licenses. [Third-party notices](THIRD_PARTY.md).
