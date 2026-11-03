Fixes model-download connections on macOS: uses system TLS/Keychain trust, honors system proxies (including SOCKS), and displays connection errors without long URLs hiding the cause. The build now checks all three models’ config, tokenizer, and redirected weight endpoints through the app HTTP client.

Download **Carter-Apple-Silicon.zip**, unzip, and move **Carter.app** into Applications.

Requires Apple Silicon (M1 or newer) and macOS 14+. The app uses native Rune/OpenCode UI components and Metal FP16 local inference with KV caching. GPT-2 Large downloads automatically on first launch (3.25 GB). Medium (1.52 GB) and Small (548 MB) are optional via the model selector. Once cached, chat works offline. No accounts or API keys.

This build is ad-hoc signed, not Apple notarized. macOS may block the first launch: open System Settings → Privacy & Security → Open Anyway after trying to launch. If macOS reports a damaged app, remove its quarantine attribute using `xattr -dr com.apple.quarantine /Applications/Carter.app`, then open it again. Only do this for the app downloaded from this repository.

GPT-2 is a 2019 completion model, not an instruction-tuned assistant. Replies may be incoherent or offensive. Chats stay in memory and disappear when the app closes. Downloads are pinned to Hugging Face revisions; model weights are SHA-256 verified before use.
