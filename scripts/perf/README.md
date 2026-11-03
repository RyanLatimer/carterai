# Inference performance checks

Build the current release app, then benchmark a local model folder. This does not
open the chat window or use network access:

```sh
cargo build --release --locked
target/release/carterai --benchmark "$HOME/Library/Application Support/CarterAI/models/gpt2-large" --tokens 32 --prompt-tokens 32 --rounds 3 > current.json
```

Run on the Mac that experienced slow output, with the app closed. The folder must
contain `config.json` and `model.safetensors`. Keep the model, prompt length,
output length, thermal conditions and power mode the same for each comparison.
Run once with a short prompt (32 tokens), then with a long one (512 tokens).

The JSON separates:

- model/device load time, including synchronization after weight upload;
- the first execution, which includes cold kernels/pipelines;
- full prompt prefill and time to first token;
- at least two warm trials (three by default), each with a fresh KV cache;
- sustained one-token decode throughput and p50/p95 token latency.

Every forward pass reads its logits back to CPU before the timer ends. This
prevents asynchronous GPU work from producing inflated speed reports. The
workload uses deterministic synthetic token IDs and greedy fixed-length output,
ignores EOS, and excludes text tokenization, text decoding, sampling and UI work.
It measures the model backend rather than claiming end-to-end app performance.

The original app divided generated tokens by the time for **prefill plus all
decode steps**, including first-use kernel setup. Its reported `0.4 tokens/s`
cannot distinguish a slow first token from slow sustained decoding. The first
and warm trials in this benchmark make that distinction explicit.

For a baseline using the original shipping GPT-2 implementation and Candle0.9.2:

```sh
scripts/perf/build-baseline.sh
target/perf/carter-baseline "$MODEL_DIR" --tokens 32 --prompt-tokens 32 --rounds 3 > baseline.json
```

`CARTER_BASELINE_REF` can select another local Git revision. The script copies the
selected source into a temporary standalone Cargo project; it creates no checkout
and does not modify the app. Alternate baseline/current execution order across
repeated comparisons. Compare warm medians as well as the cold first-trial values.
A GitHub macOS runner result describes that runner; it cannot stand in for the
user's physical Mac. JSON includes machine, virtualization and Actions metadata.
