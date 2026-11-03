//! Native MLX GPT-2. Uses Apple's fused attention and normalization kernels, lazy execution,
//! GEMV-specialized projections, and fixed-capacity KV caches. No Python runtime is involved.
//! Weight layout and activation follow Apple's mlx-lm/models/gpt2.py (MIT, Apple Inc.).
use crate::gpt2::Config;
use anyhow::{Context, Result, ensure};
use mlx_rs::{
    Array, Device, Dtype, Stream, fast, ops,
    ops::indexing::{TryIndexMutOp, TryIndexOp},
};
use std::{collections::HashMap, path::Path, time::Instant};

// Keep the compiled program alive. mlx-rs's nn::gelu_approximate helper constructs
// and erases a fresh compile-cache entry on each invocation, which is costly in decode.
type Gelu =
    Box<dyn for<'a> FnMut(&'a Array) -> std::result::Result<Array, mlx_rs::error::Exception>>;
fn gelu_program() -> Gelu {
    let function = |x: &Array| -> std::result::Result<Array, mlx_rs::error::Exception> {
        let cube = x.square()?.multiply(x)?;
        let inner = x
            .add(cube.multiply(mlx_rs::array!(0.044715f32))?)?
            .multiply(mlx_rs::array!((2.0f32 / std::f32::consts::PI).sqrt()))?;
        ops::tanh(inner)?
            .add(mlx_rs::array!(1.0f32))?
            .multiply(x)?
            .multiply(mlx_rs::array!(0.5f32))
    };
    Box::new(mlx_rs::transforms::compile::compile(function, true))
}

struct Linear {
    // Store normal nn::Linear [output,input] order once, rather than transposing/copying per token.
    weight: Array,
    bias: Array,
}
impl Linear {
    fn load(weights: &mut Weights, name: &str) -> Result<Self> {
        let weight = weights.take(&format!("{name}.weight"))?.t().contiguous()?;
        let bias = weights.take(&format!("{name}.bias"))?;
        mlx_rs::transforms::eval([&weight, &bias])?;
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(ops::addmm(&self.bias, x, self.weight.t(), None, None)?)
    }
}
struct Norm {
    weight: Array,
    bias: Array,
    eps: f32,
}
impl Norm {
    fn load(weights: &mut Weights, name: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: weights.take(&format!("{name}.weight"))?,
            bias: weights.take(&format!("{name}.bias"))?,
            eps: eps as f32,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(x, &self.weight, &self.bias, self.eps)?)
    }
}
struct Block {
    ln1: Norm,
    ln2: Norm,
    qkv: Linear,
    proj: Linear,
    fc: Linear,
    fc_proj: Linear,
    keys: Option<Array>,
    values: Option<Array>,
    heads: i32,
    capacity: i32,
}
impl Block {
    fn load(weights: &mut Weights, i: usize, c: &Config) -> Result<Self> {
        let prefix = format!("h.{i}");
        Ok(Self {
            ln1: Norm::load(weights, &format!("{prefix}.ln_1"), c.layer_norm_epsilon)?,
            ln2: Norm::load(weights, &format!("{prefix}.ln_2"), c.layer_norm_epsilon)?,
            qkv: Linear::load(weights, &format!("{prefix}.attn.c_attn"))?,
            proj: Linear::load(weights, &format!("{prefix}.attn.c_proj"))?,
            fc: Linear::load(weights, &format!("{prefix}.mlp.c_fc"))?,
            fc_proj: Linear::load(weights, &format!("{prefix}.mlp.c_proj"))?,
            keys: None,
            values: None,
            heads: c.n_head as i32,
            capacity: c.n_positions as i32,
        })
    }
    fn forward(&mut self, x: &Array, past: i32, gelu: &mut Gelu) -> Result<Array> {
        let (batch, t, n) = (x.dim(0), x.dim(1), x.dim(2));
        let head_dim = n / self.heads;
        let qkv = self.qkv.forward(&self.ln1.forward(x)?)?;
        let split = |i: i32| -> Result<Array> {
            Ok(qkv
                .try_index((.., .., i * n..(i + 1) * n))?
                .reshape(&[batch, t, self.heads, head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = split(0)?;
        let k = split(1)?;
        let v = split(2)?;
        let shape = [batch, self.heads, self.capacity, head_dim];
        if self.keys.is_none() {
            self.keys = Some(ops::zeros_dtype(&shape, x.dtype())?);
            self.values = Some(ops::zeros_dtype(&shape, x.dtype())?);
        }
        let keys = self.keys.as_mut().context("Missing key cache")?;
        let values = self.values.as_mut().context("Missing value cache")?;
        keys.try_index_mut((.., .., past..past + t, ..), &k)?;
        values.try_index_mut((.., .., past..past + t, ..), &v)?;
        let k = keys.try_index((.., .., ..past + t, ..))?;
        let v = values.try_index((.., .., ..past + t, ..))?;
        let mask = if t > 1 {
            Some(fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attention = fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            1.0 / (head_dim as f32).sqrt(),
            mask,
            None,
        )?;
        let attention = attention
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, t, n])?;
        let h = ops::add(x, self.proj.forward(&attention)?)?;
        let ff = gelu(&self.fc.forward(&self.ln2.forward(&h)?)?)?;
        Ok(ops::add(&h, self.fc_proj.forward(&ff)?)?)
    }
}
struct Weights {
    arrays: HashMap<String, Array>,
    prefix: &'static str,
    dtype: Dtype,
}
impl Weights {
    fn take(&mut self, name: &str) -> Result<Array> {
        let name = format!("{}{name}", self.prefix);
        let array = self
            .arrays
            .remove(&name)
            .with_context(|| format!("Missing GPT-2 weight {name}"))?;
        Ok(array.as_dtype(self.dtype)?)
    }
}

pub struct MlxGpt2 {
    stream: Stream,
    gelu: Gelu,
    wte: Array,
    wpe: Array,
    blocks: Vec<Block>,
    ln: Norm,
    pub config: Config,
    cached: usize,
}
impl MlxGpt2 {
    /// Called on the inference worker before any MLX operation. Packaged apps use a bundled
    /// metallib and never depend on a build-machine home directory.
    pub fn configure_library() -> Result<()> {
        if let Some(resources) = std::env::current_exe()?
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("Resources"))
        {
            if resources.join("mlx.metallib").is_file() {
                mlx_rs::metal::set_metallib_path(resources.join("mlx.metallib").to_string_lossy())?;
            }
        }
        Ok(())
    }
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_dtype(dir, Dtype::Float16)
    }
    pub fn load_dtype(dir: &Path, dtype: Dtype) -> Result<Self> {
        Self::configure_library()?;
        let stream = Stream::new_with_device(&Device::gpu());
        mlx_rs::with_stream(&stream, || {
            let config = crate::models::load_config(dir)?;
            let arrays = Array::load_safetensors(dir.join("model.safetensors"))?;
            let prefix = if arrays.contains_key("wte.weight") {
                ""
            } else {
                "transformer."
            };
            let mut weights = Weights {
                arrays,
                prefix,
                dtype,
            };
            let wte = weights.take("wte.weight")?;
            let wpe = weights.take("wpe.weight")?;
            let blocks = (0..config.n_layer)
                .map(|i| Block::load(&mut weights, i, &config))
                .collect::<Result<Vec<_>>>()?;
            let ln = Norm::load(&mut weights, "ln_f", config.layer_norm_epsilon)?;
            // Materialize once on load. Decoding then uses only resident FP16 weights.
            mlx_rs::transforms::eval([&wte, &wpe, &ln.weight, &ln.bias])?;
            Ok(Self {
                stream: stream.clone(),
                gelu: gelu_program(),
                wte,
                wpe,
                blocks,
                ln,
                config,
                cached: 0,
            })
        })
    }
    /// Compile and exercise both prefill and single-token decode before showing Ready.
    /// A short warmup also materializes lazy norm weights and allocates KV buffers.
    pub fn warmup(&mut self) -> Result<f64> {
        let started = Instant::now();
        let length = 32.min(self.config.n_positions.saturating_sub(1));
        ensure!(length > 0, "GPT-2 context is too short");
        let ids: Vec<u32> = (0..length)
            .map(|i| ((i + 1) % self.config.vocab_size) as u32)
            .collect();
        self.reset();
        let _ = self.forward(&ids, 0)?;
        let _ = self.forward(&[1], length)?;
        self.reset();
        Ok(started.elapsed().as_secs_f64())
    }
    pub fn reset(&mut self) {
        // Reuse allocated cache storage; the next prefill overwrites all visible positions.
        self.cached = 0;
    }
    pub fn forward(&mut self, ids: &[u32], past: usize) -> Result<Vec<f32>> {
        ensure!(
            !ids.is_empty() && past + ids.len() <= self.config.n_positions,
            "Context window exceeded"
        );
        ensure!(
            past == self.cached,
            "KV cache position mismatch: expected {}, got {past}",
            self.cached
        );
        ensure!(
            ids.iter().all(|&id| (id as usize) < self.config.vocab_size),
            "Token ID outside GPT-2 vocabulary"
        );
        let stream = self.stream.clone();
        mlx_rs::with_stream(&stream, || {
            let t = ids.len() as i32;
            let tokens = Array::from_slice(ids, &[1, t]);
            let positions: Vec<u32> = (past..past + ids.len()).map(|i| i as u32).collect();
            let positions = Array::from_slice(&positions, &[1, t]);
            let mut x = ops::add(
                self.wte.take_axis(&tokens, 0)?,
                self.wpe.take_axis(&positions, 0)?,
            )?;
            for block in &mut self.blocks {
                x = block.forward(&x, past as i32, &mut self.gelu)?;
            }
            // Only the last position needs output projection; do not project the whole prefill.
            let last = self.ln.forward(&x.try_index((0, t - 1, ..))?)?;
            let logits = last
                .reshape(&[1, self.config.n_embd as i32])?
                .matmul(self.wte.t())?
                .reshape(&[self.config.vocab_size as i32])?
                .as_type::<f32>()?
                .contiguous()?;
            // One graph evaluation per token, rather than synchronizing after individual ops.
            logits.eval()?;
            self.cached = past + ids.len();
            Ok(logits.try_as_slice::<f32>()?.to_vec())
        })
    }
}

pub fn smoke(dir: &Path) -> Result<()> {
    use candle_core::{DType, Device as CandleDevice};
    use candle_nn::VarBuilder;
    let mut mlx = MlxGpt2::load(dir)?;
    let full = mlx.forward(&[1, 2, 3, 4], 0)?;
    mlx.reset();
    let _ = mlx.forward(&[1, 2, 3], 0)?;
    let cached = mlx.forward(&[4], 3)?;
    mlx.reset();
    let _ = mlx.forward(&[1, 2], 0)?;
    let chunked = mlx.forward(&[3, 4], 2)?;
    let chunk_diff = full
        .iter()
        .zip(&chunked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    ensure!(
        chunk_diff < 0.05,
        "MLX cached multi-token mask mismatch: {chunk_diff}"
    );
    let cache_diff = full
        .iter()
        .zip(&cached)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    ensure!(
        full.iter().all(|v| v.is_finite()) && cache_diff < 0.05,
        "MLX KV cache mismatch: {cache_diff}"
    );
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[dir.join("model.safetensors")],
            DType::F32,
            &CandleDevice::Cpu,
        )?
    };
    let mut candle = crate::gpt2::Gpt2::load(vb, crate::models::load_config(dir)?)?;
    let reference = candle.forward(&[1, 2, 3, 4], 0)?.to_vec1::<f32>()?;
    let backend_diff = full
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    ensure!(
        backend_diff < 0.15,
        "MLX vs Candle CPU logits differ: {backend_diff}"
    );
    println!(
        "MLX FP16 inference OK: {} logits, max cached/full diff {cache_diff}, max MLX/Candle CPU diff {backend_diff}",
        full.len()
    );
    Ok(())
}

/// Warmed prefill and decode are measured separately; all outputs are evaluated before timing ends.
pub fn benchmark(
    dir: &Path,
    prompt_tokens: usize,
    decode_tokens: usize,
) -> Result<serde_json::Value> {
    let load_started = Instant::now();
    let mut model = MlxGpt2::load(dir)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    ensure!(
        prompt_tokens > 0
            && decode_tokens > 0
            && prompt_tokens + decode_tokens <= model.config.n_positions,
        "Invalid benchmark dimensions"
    );
    let prompt: Vec<u32> = (0..prompt_tokens)
        .map(|i| ((i + 1) % model.config.vocab_size) as u32)
        .collect();
    let cold_started = Instant::now();
    let _ = model.forward(&prompt, 0)?;
    let cold_prefill_seconds = cold_started.elapsed().as_secs_f64();
    let cold_started = Instant::now();
    let _ = model.forward(&[4], prompt_tokens)?;
    let cold_decode_seconds = cold_started.elapsed().as_secs_f64();
    let mut runs = Vec::new();
    for _ in 0..3 {
        model.reset();
        let started = Instant::now();
        let _ = model.forward(&prompt, 0)?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        for i in 0..decode_tokens {
            let logits = model.forward(&[4], prompt_tokens + i)?;
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "Nonfinite decode logits"
            );
        }
        runs.push(serde_json::json!({"prefill_seconds":prefill_seconds,"decode_seconds":started.elapsed().as_secs_f64(),"decode_tokens_per_second":decode_tokens as f64/started.elapsed().as_secs_f64()}));
    }
    Ok(
        serde_json::json!({"backend":"MLX FP16","model_dir":dir.file_name().and_then(|x|x.to_str()),"load_seconds":load_seconds,"cold_prefill_seconds":cold_prefill_seconds,"cold_decode_seconds":cold_decode_seconds,"prompt_tokens":prompt_tokens,"decode_tokens":decode_tokens,"runs":runs}),
    )
}
