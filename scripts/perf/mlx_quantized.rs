//! Experimental affine weight-only quantization probes for the native MLX backend.
//! This is deliberately separate from shipping inference until full-model quality and
//! hardware measurements justify an opt-in mode. No downloaded weights are changed.
use anyhow::{Context, Result, ensure};
use mlx_rs::{Array, Device, Dtype, Stream, ops};
use std::{path::Path, time::Instant};

/// An nn::Linear-compatible weight-only projection. Norms, activations, token/position
/// embeddings and KV caches retain FP16 precision. Groupwise scale/bias stay FP16.
pub struct QuantizedLinear {
    packed: Array,
    scales: Array,
    offsets: Array,
    bias: Array,
    bits: i32,
    input: i32,
    output: i32,
}
impl QuantizedLinear {
    pub fn from_weight(weight: &Array, bias: &Array, bits: i32) -> Result<Self> {
        ensure!(
            matches!(bits, 4 | 8),
            "Only experimental 4-bit and 8-bit projections are supported"
        );
        ensure!(
            weight.ndim() == 2 && bias.ndim() == 1,
            "Expected [output,input] linear weights and [output] bias"
        );
        ensure!(
            weight.dim(1) % 64 == 0 && weight.dim(0) == bias.dim(0),
            "Invalid quantized projection dimensions"
        );
        let (packed, scales, offsets) = ops::quantize(weight, 64, bits)?;
        mlx_rs::transforms::eval([&packed, &scales, &offsets])?;
        Ok(Self {
            packed,
            scales,
            offsets,
            bias: bias.clone(),
            bits,
            input: weight.dim(1),
            output: weight.dim(0),
        })
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            x.dim(-1) == self.input,
            "Quantized linear input dimension mismatch"
        );
        let y = ops::quantized_matmul(
            x,
            &self.packed,
            &self.scales,
            &self.offsets,
            true,
            64,
            self.bits,
        )?;
        Ok(ops::add(y, &self.bias)?)
    }
    pub fn storage_bytes(&self) -> usize {
        self.packed.nbytes() + self.scales.nbytes() + self.offsets.nbytes() + self.bias.nbytes()
    }
    fn dequantized(&self) -> Result<Array> {
        Ok(ops::dequantize(
            &self.packed,
            &self.scales,
            &self.offsets,
            64,
            self.bits,
        )?)
    }
}

fn compare(reference: &[f32], candidate: &[f32]) -> Result<serde_json::Value> {
    ensure!(
        reference.len() == candidate.len() && !reference.is_empty(),
        "Projection output shape mismatch"
    );
    ensure!(
        candidate.iter().all(|v| v.is_finite()),
        "Nonfinite quantized projection output"
    );
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut max_error = 0.0f32;
    for (&a, &b) in reference.iter().zip(candidate) {
        squared_error += ((a - b) as f64).powi(2);
        squared_reference += (a as f64).powi(2);
        max_error = max_error.max((a - b).abs());
    }
    Ok(
        serde_json::json!({"relative_l2_error":(squared_error/squared_reference.max(1e-30)).sqrt(),"max_absolute_error":max_error}),
    )
}
fn values(array: Array) -> Result<Vec<f32>> {
    let array = array.as_type::<f32>()?.contiguous()?;
    array.eval()?;
    Ok(array.try_as_slice::<f32>()?.to_vec())
}
fn time_projection(mut call: impl FnMut() -> Result<Array>) -> Result<serde_json::Value> {
    // Discard first execution (including specialization/compilation), then synchronize
    // every sample before ending its timer. This is one matrix, not full-model tokens/s.
    let _ = values(call()?)?;
    let mut timings = Vec::new();
    for _ in 0..32 {
        let started = Instant::now();
        let _ = values(call()?)?;
        timings.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(f64::total_cmp);
    Ok(
        serde_json::json!({"iterations":timings.len(),"p50_milliseconds":timings[timings.len()/2],"p95_milliseconds":timings[(timings.len()-1)*95/100]}),
    )
}

/// Optional microbenchmark: uses real first-layer GPT-2 projection weights and deterministic
/// synthetic vectors, comparing FP16/Q8/Q4 storage, output error, and synchronized latency.
/// It cannot establish chat quality or a full-model decode speedup on its own.
pub fn probe(dir: &Path) -> Result<serde_json::Value> {
    crate::mlx_gpt2::MlxGpt2::configure_library()?;
    let stream = Stream::new_with_device(&Device::gpu());
    mlx_rs::with_stream(&stream, || {
        let mut weights = Array::load_safetensors(dir.join("model.safetensors"))?;
        let prefix = if weights.contains_key("h.0.attn.c_attn.weight") {
            ""
        } else {
            "transformer."
        };
        let mut projections = Vec::new();
        for name in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"] {
            let name = format!("{prefix}h.0.{name}");
            // HF Conv1D weights are [input,output]. Materialize normal Linear order once.
            let weight = weights
                .remove(&format!("{name}.weight"))
                .with_context(|| format!("Missing {name}.weight"))?
                .as_dtype(Dtype::Float16)?
                .t()
                .contiguous()?;
            let bias = weights
                .remove(&format!("{name}.bias"))
                .with_context(|| format!("Missing {name}.bias"))?
                .as_dtype(Dtype::Float16)?;
            mlx_rs::transforms::eval([&weight, &bias])?;
            let input: Vec<f32> = (0..weight.dim(1))
                .map(|i| ((i as f32 + 1.0) * 0.17).sin())
                .collect();
            let x = Array::from_slice(&input, &[1, 1, weight.dim(1)]).as_dtype(Dtype::Float16)?;
            x.eval()?;
            let reference = values(ops::addmm(&bias, &x, weight.t(), None, None)?)?;
            let fp16_timing =
                time_projection(|| Ok(ops::addmm(&bias, &x, weight.t(), None, None)?))?;
            let mut modes = Vec::new();
            for bits in [8, 4] {
                let started = Instant::now();
                let quantized = QuantizedLinear::from_weight(&weight, &bias, bits)?;
                let conversion_seconds = started.elapsed().as_secs_f64();
                let candidate = values(quantized.forward(&x)?)?;
                // First verify that the packed matmul kernel agrees with reconstructed weights.
                // This is separate from expected quantization error against the original weights.
                let reconstructed = quantized.dequantized()?;
                let reconstructed_output =
                    values(ops::addmm(&bias, &x, reconstructed.t(), None, None)?)?;
                let kernel_diff = candidate
                    .iter()
                    .zip(&reconstructed_output)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                ensure!(
                    kernel_diff < 0.1,
                    "Quantized kernel disagrees with reconstructed projection by {kernel_diff}"
                );
                modes.push(serde_json::json!({"bits":bits,"group_size":64,"input":quantized.input,"output":quantized.output,
                    "storage_bytes":quantized.storage_bytes(),"conversion_seconds":conversion_seconds,
                    "output_error":compare(&reference,&candidate)?,"reconstructed_kernel_max_error":kernel_diff,
                    "timing":time_projection(||quantized.forward(&x))?}));
            }
            projections.push(serde_json::json!({"name":name,"fp16_storage_bytes":weight.nbytes()+bias.nbytes(),"fp16_timing":fp16_timing,"modes":modes}));
        }
        Ok(
            serde_json::json!({"benchmark":"experimental_projection_quantization","warning":"Synthetic single-projection microbenchmark; not a full-model speed or quality result","projections":projections}),
        )
    })
}
