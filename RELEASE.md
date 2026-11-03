Carter 0.2.0 development build for Apple Silicon.

- Fixes streamed-text flicker, transient role markers, text wrapping, and scrolling while reading older messages.
- Keeps only “A little less intelligent.” in the empty state.
- Adds optimized Candle Metal inference and native MLX, GPU warmup before Ready, cross-turn KV reuse, and separate first-token/decode timing. Automatic uses Candle Metal, which measured about 4× MLX's decode throughput on the CI M1 runner; MLX remains selectable.
- Adds macOS system proxy/PAC resolution, HTTPS Keychain trust, manual proxy/endpoint/token settings, offline mode, and local model-folder import.
- Adds resumable downloads with ETag/Content-Range validation, retries, pinned SHA-256 weight verification, and readable diagnostics.

Download the zip, unzip, and move Carter.app to Applications. Existing model downloads remain usable. Clear the local-folder field to return to downloading models; select a folder containing config.json, tokenizer.json, and the original model.safetensors to import one. Hugging Face pinned snapshot caches are also discovered automatically.

This is a development build: no Developer ID is available, so it is ad-hoc signed and NOT notarized. Gatekeeper may block first launch. Do not mistake this for an Apple-notarized release. The normal versioned-release workflow requires Developer ID signing and successful notarization, stapling, and Gatekeeper verification.

GPT-2 is a completion model, not an instruction-tuned assistant. Chats stay in memory and disappear on exit. No inference requests or chat text are sent to a server. Network diagnostics contain endpoints, selected proxy host, redirect history, and error chains; tokens, proxy credentials, and signed query strings are omitted.
