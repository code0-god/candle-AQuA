use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Result, Shape};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use std::collections::BTreeMap;

// Reference: ajou-aisa/llama.cpp-gemmini
// commit d5e76be1fca91314c5a0745038b3cedbbdbed13d
const GGML_QNT_VERSION: u32 = 2;

// Reference: ajou-aisa/llama.cpp-gemmini
// commit d5e76be1fca91314c5a0745038b3cedbbdbed13d
const LLAMA_FTYPE_MOSTLY_Q8_H1: u32 = 38;
const LLAMA_FTYPE_MOSTLY_Q8_HP1: u32 = 40;

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum QuantizationMode {
    /// The default quantization includes all 2d tensors, except the output tensor which always
    /// uses Q6_K.
    Llama,
    /// Row-aware Q8_H GGUF conversion matching llama.cpp-gemmini.
    Aqua,
}

impl QuantizationMode {
    fn quantize(&self, name: &str, tensor: QTensor, dtype: GgmlDType) -> Result<QTensor> {
        match self {
            Self::Llama => {
                // Same behavior as the llama.cpp quantization.
                let should_quantize = name.ends_with(".weight") && tensor.rank() == 2;
                if should_quantize {
                    let tensor = tensor.dequantize(&Device::Cpu)?;
                    if name == "output.weight" {
                        QTensor::quantize(&tensor, GgmlDType::Q6K)
                    } else {
                        QTensor::quantize(&tensor, dtype)
                    }
                } else {
                    Ok(tensor)
                }
            }
            Self::Aqua => {
                candle::bail!("Aqua quantization must use the Q8_H GGUF conversion path")
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    #[value(name = "q8_h1")]
    Q8H1,
    #[value(name = "q8_hp1")]
    Q8HP1,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl Quantization {
    fn dtype(&self) -> GgmlDType {
        match self {
            Quantization::Q4_0 => GgmlDType::Q4_0,
            Quantization::Q4_1 => GgmlDType::Q4_1,
            Quantization::Q5_0 => GgmlDType::Q5_0,
            Quantization::Q5_1 => GgmlDType::Q5_1,
            Quantization::Q8_0 => GgmlDType::Q8_0,
            Quantization::Q8_1 => GgmlDType::Q8_1,
            Quantization::Q8H1 => GgmlDType::Q8H1,
            Quantization::Q8HP1 => GgmlDType::Q8HP1,
            Quantization::Q2k => GgmlDType::Q2K,
            Quantization::Q3k => GgmlDType::Q3K,
            Quantization::Q4k => GgmlDType::Q4K,
            Quantization::Q5k => GgmlDType::Q5K,
            Quantization::Q6k => GgmlDType::Q6K,
            Quantization::Q8k => GgmlDType::Q8K,
            Quantization::F16 => GgmlDType::F16,
            Quantization::F32 => GgmlDType::F32,
        }
    }

    fn is_h(self) -> bool {
        matches!(self, Self::Q8H1 | Self::Q8HP1)
    }

    fn file_type(self) -> Option<u32> {
        match self {
            Self::Q8H1 => Some(LLAMA_FTYPE_MOSTLY_Q8_H1),
            Self::Q8HP1 => Some(LLAMA_FTYPE_MOSTLY_Q8_HP1),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AquaCopyReason {
    NotWeight,
    NotMatrix,
    Normalization,
    PositionEmbedding,
    TokenTypeEmbedding,
    ExpertGate,
    SsmConv1d,
    RwkvTimeMix,
    RelativeAttentionBias,
}

impl AquaCopyReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::NotWeight => "not-weight",
            Self::NotMatrix => "not-matrix-weight",
            Self::Normalization => "normalization",
            Self::PositionEmbedding => "position-embedding",
            Self::TokenTypeEmbedding => "token-type-embedding",
            Self::ExpertGate => "expert-gate",
            Self::SsmConv1d => "ssm-conv1d",
            Self::RwkvTimeMix => "rwkv-time-mix",
            Self::RelativeAttentionBias => "relative-attention-bias",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AquaQuantizationAction {
    Quantize,
    Copy(AquaCopyReason),
}

// Reference:
// ajou-aisa/llama.cpp-gemmini
// commit d5e76be1fca91314c5a0745038b3cedbbdbed13d
// src/llama-quant.cpp: llama_model_quantize_impl
fn classify_aqua_quantization(name: &str, shape: &Shape) -> AquaQuantizationAction {
    if !name.ends_with("weight") {
        return AquaQuantizationAction::Copy(AquaCopyReason::NotWeight);
    }
    if shape.rank() < 2 {
        return AquaQuantizationAction::Copy(AquaCopyReason::NotMatrix);
    }
    if name.contains("_norm.weight") {
        return AquaQuantizationAction::Copy(AquaCopyReason::Normalization);
    }
    if name == "position_embd.weight" {
        return AquaQuantizationAction::Copy(AquaCopyReason::PositionEmbedding);
    }
    if name == "token_types.weight" {
        return AquaQuantizationAction::Copy(AquaCopyReason::TokenTypeEmbedding);
    }
    if name.contains("ffn_gate_inp.weight") {
        return AquaQuantizationAction::Copy(AquaCopyReason::ExpertGate);
    }
    if name.contains("ssm_conv1d.weight") {
        return AquaQuantizationAction::Copy(AquaCopyReason::SsmConv1d);
    }
    const RWKV_TIME_MIX_WEIGHTS: [&str; 15] = [
        "time_mix_first.weight",
        "time_mix_w0.weight",
        "time_mix_w1.weight",
        "time_mix_w2.weight",
        "time_mix_v0.weight",
        "time_mix_v1.weight",
        "time_mix_v2.weight",
        "time_mix_a0.weight",
        "time_mix_a1.weight",
        "time_mix_a2.weight",
        "time_mix_g1.weight",
        "time_mix_g2.weight",
        "time_mix_decay_w1.weight",
        "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight",
    ];
    if RWKV_TIME_MIX_WEIGHTS
        .iter()
        .any(|excluded| name.contains(excluded))
    {
        return AquaQuantizationAction::Copy(AquaCopyReason::RwkvTimeMix);
    }
    if name.contains("attn_rel_b.weight") {
        return AquaQuantizationAction::Copy(AquaCopyReason::RelativeAttentionBias);
    }
    AquaQuantizationAction::Quantize
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AquaConversionSummary {
    total_tensors: usize,
    quantized_tensors: usize,
    already_target_tensors: usize,
    copied_tensors: usize,
    copied_by_dtype: BTreeMap<String, usize>,
    skipped_matrix_tensors: Vec<String>,
    input_bytes: u64,
    output_bytes: u64,
}

#[derive(ValueEnum, Debug, Clone)]
enum Format {
    Safetensors,
    Npz,
    Ggml,
    Gguf,
    Pth,
    Pickle,
}

impl Format {
    fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        p.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|e| match e {
                // We don't infer any format for .bin as it can be used for ggml/gguf or pytorch.
                "safetensors" | "safetensor" => Some(Self::Safetensors),
                "npz" => Some(Self::Npz),
                "pth" | "pt" => Some(Self::Pth),
                "ggml" => Some(Self::Ggml),
                "gguf" => Some(Self::Gguf),
                _ => None,
            })
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Ls {
        files: Vec<std::path::PathBuf>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Enable verbose mode.
        #[arg(short, long)]
        verbose: bool,
    },

    Print {
        file: std::path::PathBuf,

        names: Vec<String>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Print the whole content of each tensor.
        #[arg(long)]
        full: bool,

        /// Line width for printing the tensors.
        #[arg(long)]
        line_width: Option<usize>,
    },

    Quantize {
        /// Input file(s). Q8_H Aqua mode requires one GGUF file.
        in_file: Vec<std::path::PathBuf>,

        /// The output file, in gguf format.
        #[arg(long)]
        out_file: std::path::PathBuf,

        /// The quantization schema to apply.
        #[arg(long, value_enum)]
        quantization: Quantization,

        /// Which tensor to quantize.
        #[arg(long, value_enum, default_value_t = QuantizationMode::Llama)]
        mode: QuantizationMode,
    },

    Dequantize {
        /// The input file, in gguf format.
        in_file: std::path::PathBuf,

        /// The output file, in safetensors format.
        #[arg(long)]
        out_file: std::path::PathBuf,
    },
}

#[derive(Parser, Debug, Clone)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn run_print(
    file: &std::path::PathBuf,
    names: Vec<String>,
    format: Option<Format>,
    full: bool,
    line_width: Option<usize>,
    device: &Device,
) -> Result<()> {
    if full {
        candle::display::set_print_options_full();
    }
    if let Some(line_width) = line_width {
        candle::display::set_line_width(line_width)
    }
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle::npy::NpzTensors::new(file)?;
            let names = if names.is_empty() {
                tensors.names().into_iter().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match tensors.get(name)? {
                    Some(tensor) => println!("{tensor}"),
                    None => println!("not found"),
                }
            }
        }
        Format::Safetensors => {
            use candle::safetensors::Load;
            let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(file)? };
            let tensors: std::collections::HashMap<_, _> = tensors.tensors().into_iter().collect();
            let names = if names.is_empty() {
                tensors.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match tensors.get(name) {
                    Some(tensor_view) => {
                        let tensor = tensor_view.load(device)?;
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Pth => {
            let pth_file = candle::pickle::PthTensors::new(file, None)?;
            let names = if names.is_empty() {
                pth_file
                    .tensor_infos()
                    .keys()
                    .map(|v| v.to_string())
                    .collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match pth_file.get(name)? {
                    Some(tensor) => {
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Pickle => {
            candle::bail!("pickle format is not supported for print")
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle::quantized::ggml_file::Content::read(&mut file, device)?;
            let names = if names.is_empty() {
                content.tensors.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match content.tensors.get(name) {
                    Some(tensor) => {
                        let tensor = tensor.dequantize(device)?;
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            let names = if names.is_empty() {
                content.tensor_infos.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match content.tensor(&mut file, name, device) {
                    Ok(tensor) => {
                        let tensor = tensor.dequantize(device)?;
                        println!("{tensor}")
                    }
                    Err(_) => println!("not found"),
                }
            }
        }
    }
    Ok(())
}

fn run_ls(
    file: &std::path::PathBuf,
    format: Option<Format>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle::npy::NpzTensors::new(file)?;
            let mut names = tensors.names();
            names.sort();
            for name in names {
                let shape_dtype = match tensors.get_shape_and_dtype(name) {
                    Ok((shape, dtype)) => format!("[{shape:?}; {dtype:?}]"),
                    Err(err) => err.to_string(),
                };
                println!("{name}: {shape_dtype}")
            }
        }
        Format::Safetensors => {
            let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(file)? };
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        }
        Format::Pth => {
            let mut tensors = candle::pickle::read_pth_tensor_info(file, verbose, None)?;
            tensors.sort_by(|a, b| a.name.cmp(&b.name));
            for tensor_info in tensors.iter() {
                println!(
                    "{}: [{:?}; {:?}]",
                    tensor_info.name,
                    tensor_info.layout.shape(),
                    tensor_info.dtype,
                );
                if verbose {
                    println!("    {tensor_info:?}");
                }
            }
        }
        Format::Pickle => {
            let file = std::fs::File::open(file)?;
            let mut reader = std::io::BufReader::new(file);
            let mut stack = candle::pickle::Stack::empty();
            stack.read_loop(&mut reader)?;
            for (i, obj) in stack.stack().iter().enumerate() {
                println!("{i} {obj:?}");
            }
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle::quantized::ggml_file::Content::read(&mut file, device)?;
            let mut tensors = content.tensors.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, qtensor) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", qtensor.shape(), qtensor.dtype());
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            if verbose {
                let mut metadata = content.metadata.into_iter().collect::<Vec<_>>();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
                println!("metadata entries ({})", metadata.len());
                for (key, value) in metadata.iter() {
                    println!("  {key}: {value:?}");
                }
            }
            let mut tensors = content.tensor_infos.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, info) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", info.shape, info.ggml_dtype);
            }
        }
    }
    Ok(())
}

fn run_quantize_safetensors(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
) -> Result<()> {
    let mut out_file = std::fs::File::create(out_file)?;
    let mut tensors = std::collections::HashMap::new();
    for in_file in in_files.iter() {
        let in_tensors = candle::safetensors::load(in_file, &Device::Cpu)?;
        tensors.extend(in_tensors)
    }
    println!("tensors: {}", tensors.len());

    let dtype = q.dtype();
    let block_size = dtype.block_size();

    let qtensors = tensors
        .into_par_iter()
        .map(|(name, tensor)| {
            let should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
            println!("  quantizing {name} {tensor:?} {should_quantize}");
            let tensor = if should_quantize {
                QTensor::quantize(&tensor, dtype)?
            } else {
                QTensor::quantize(&tensor, GgmlDType::F32)?
            };
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut out_file, &[], &qtensors)?;
    Ok(())
}

fn run_dequantize(
    in_file: std::path::PathBuf,
    out_file: std::path::PathBuf,
    device: &Device,
) -> Result<()> {
    let mut in_file = std::fs::File::open(in_file)?;
    let content = gguf_file::Content::read(&mut in_file)?;
    let mut tensors = std::collections::HashMap::new();
    for tensor_name in content.tensor_infos.keys() {
        let tensor = content.tensor(&mut in_file, tensor_name, device)?;
        let tensor = tensor.dequantize(device)?;
        tensors.insert(tensor_name.to_string(), tensor);
    }
    candle::safetensors::save(&tensors, out_file)?;
    Ok(())
}

fn validate_aqua_gguf_version(version: gguf_file::VersionedMagic) -> Result<()> {
    match version {
        gguf_file::VersionedMagic::GgufV2 | gguf_file::VersionedMagic::GgufV3 => Ok(()),
        gguf_file::VersionedMagic::GgufV1 => {
            candle::bail!("Q8_H conversion supports GGUF V2/V3, not GGUF V1")
        }
    }
}

fn canonical_output_path(path: &std::path::Path) -> Result<std::path::PathBuf> {
    if path.exists() {
        return Ok(std::fs::canonicalize(path)?);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| candle::Error::Msg(format!("invalid output path {path:?}")))?;
    Ok(std::fs::canonicalize(parent)?.join(file_name))
}

fn write_aqua_gguf_atomic(
    path: &std::path::Path,
    version: gguf_file::VersionedMagic,
    metadata: &[(String, gguf_file::Value)],
    tensors: &[(String, QTensor)],
) -> Result<()> {
    let mut temporary_name = path.as_os_str().to_os_string();
    temporary_name.push(".tmp");
    let temporary_path = std::path::PathBuf::from(temporary_name);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)?;
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();
    let write_result = gguf_file::write_with_version(&mut file, version, &metadata, &tensors)
        .and_then(|_| {
            file.sync_all()?;
            Ok(())
        });
    drop(file);
    if let Err(write_error) = write_result {
        if let Err(cleanup_error) = std::fs::remove_file(&temporary_path) {
            candle::bail!(
                "failed writing {temporary_path:?}: {write_error}; cleanup failed: {cleanup_error}"
            )
        }
        return Err(write_error);
    }
    if let Err(rename_error) = std::fs::rename(&temporary_path, path) {
        if let Err(cleanup_error) = std::fs::remove_file(&temporary_path) {
            candle::bail!(
                "failed renaming {temporary_path:?} to {path:?}: {rename_error}; cleanup failed: {cleanup_error}"
            )
        }
        return Err(rename_error.into());
    }
    Ok(())
}

fn run_quantize_aqua_gguf(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    quantization: Quantization,
    device: &Device,
) -> Result<AquaConversionSummary> {
    let target_dtype = quantization.dtype();
    let file_type = quantization.file_type().ok_or_else(|| {
        candle::Error::Msg(format!(
            "Aqua mode requires q8_h1 or q8_hp1, got {quantization:?}"
        ))
    })?;
    if input_path.extension().and_then(|value| value.to_str()) != Some("gguf") {
        candle::bail!("Q8_H conversion requires a single GGUF input")
    }
    if output_path.extension().and_then(|value| value.to_str()) != Some("gguf") {
        candle::bail!("Q8_H conversion output must use the gguf extension")
    }

    let input_path = std::fs::canonicalize(input_path)?;
    let output_path = canonical_output_path(output_path)?;
    if input_path == output_path {
        candle::bail!("Q8_H conversion input and output paths must differ")
    }

    let input_bytes = std::fs::metadata(&input_path)?.len();
    let mut input_file = std::fs::File::open(&input_path)?;
    let content = gguf_file::Content::read(&mut input_file)?;
    validate_aqua_gguf_version(content.magic)?;
    if content.metadata.keys().any(|key| key.starts_with("split.")) {
        candle::bail!("unsupported split GGUF input")
    }

    let mut tensor_names = content.tensor_infos.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    let total_tensors = tensor_names.len();
    let mut tensors = Vec::with_capacity(total_tensors);
    let mut quantized_tensors = 0usize;
    let mut already_target_tensors = 0usize;
    let mut copied_tensors = 0usize;
    let mut copied_by_dtype = BTreeMap::new();
    let mut skipped_matrix_tensors = Vec::new();

    for (index, name) in tensor_names.into_iter().enumerate() {
        let info = content.tensor_infos.get(&name).unwrap();
        let action = classify_aqua_quantization(&name, &info.shape);
        let source_dtype = info.ggml_dtype;
        let shape = info.shape.clone();
        let tensor = content.tensor(&mut input_file, &name, device)?;
        let tensor = match action {
            AquaQuantizationAction::Quantize => {
                let row_width = shape.dims().last().copied().ok_or_else(|| {
                    candle::Error::Msg(format!("tensor {name} has no dimensions"))
                })?;
                if !row_width.is_multiple_of(32) {
                    candle::bail!(
                        "{target_dtype:?} tensor {name} has row width {row_width}, which is not divisible by 32"
                    )
                }
                if source_dtype == target_dtype {
                    already_target_tensors += 1;
                    println!(
                        "[{}/{}] {name} shape={:?} source={source_dtype:?} action=COPY reason=already-target",
                        index + 1,
                        total_tensors,
                        shape.dims()
                    );
                    tensor
                } else {
                    match source_dtype {
                        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {}
                        _ => candle::bail!(
                            "{target_dtype:?} tensor {name} cannot requantize source dtype {source_dtype:?}"
                        ),
                    }
                    let logical = tensor.dequantize(&Device::Cpu)?;
                    let quantized = QTensor::quantize(&logical, target_dtype)?;
                    if quantized.shape().dims() != shape.dims() {
                        candle::bail!(
                            "{target_dtype:?} tensor {name} changed shape from {shape:?} to {:?}",
                            quantized.shape()
                        )
                    }
                    quantized_tensors += 1;
                    println!(
                        "[{}/{}] {name} shape={:?} source={source_dtype:?} action={target_dtype:?}",
                        index + 1,
                        total_tensors,
                        shape.dims()
                    );
                    quantized
                }
            }
            AquaQuantizationAction::Copy(reason) => {
                copied_tensors += 1;
                *copied_by_dtype
                    .entry(format!("{source_dtype:?}"))
                    .or_insert(0) += 1;
                if shape.rank() >= 2 {
                    skipped_matrix_tensors.push(name.clone());
                }
                println!(
                    "[{}/{}] {name} shape={:?} source={source_dtype:?} action=COPY reason={}",
                    index + 1,
                    total_tensors,
                    shape.dims(),
                    reason.as_str()
                );
                tensor
            }
        };
        tensors.push((name, tensor));
    }

    let mut metadata = content.metadata.clone();
    metadata.insert(
        "general.quantization_version".to_string(),
        gguf_file::Value::U32(GGML_QNT_VERSION),
    );
    metadata.insert(
        "general.file_type".to_string(),
        gguf_file::Value::U32(file_type),
    );
    let mut metadata = metadata.into_iter().collect::<Vec<_>>();
    metadata.sort_by(|left, right| left.0.cmp(&right.0));

    write_aqua_gguf_atomic(&output_path, content.magic, &metadata, &tensors)?;
    let output_bytes = std::fs::metadata(&output_path)?.len();
    println!(
        "summary total={} target={target_dtype:?} quantized={} already-target={} copied={} skipped-matrix={} input-bytes={} output-bytes={}",
        total_tensors,
        quantized_tensors,
        already_target_tensors,
        copied_tensors,
        skipped_matrix_tensors.len(),
        input_bytes,
        output_bytes
    );
    for (dtype, count) in &copied_by_dtype {
        println!("summary copied {dtype}={count}");
    }

    Ok(AquaConversionSummary {
        total_tensors,
        quantized_tensors,
        already_target_tensors,
        copied_tensors,
        copied_by_dtype,
        skipped_matrix_tensors,
        input_bytes,
        output_bytes,
    })
}

fn run_quantize(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
    qmode: QuantizationMode,
    device: &Device,
) -> Result<()> {
    if in_files.is_empty() {
        candle::bail!("no specified input files")
    }
    if q.is_h() {
        if qmode != QuantizationMode::Aqua {
            candle::bail!("{:?} quantization requires --mode aqua", q.dtype())
        }
        if in_files.len() != 1 {
            candle::bail!("Q8_H conversion requires a single GGUF input")
        }
        run_quantize_aqua_gguf(&in_files[0], &out_file, q, device)?;
        return Ok(());
    }
    if qmode == QuantizationMode::Aqua {
        candle::bail!("Aqua mode requires q8_h1 or q8_hp1 quantization")
    }
    if let Some(extension) = out_file.extension() {
        if extension == "safetensors" {
            candle::bail!("the generated file cannot use the safetensors extension")
        }
    }
    if let Some(extension) = in_files[0].extension() {
        if extension == "safetensors" {
            return run_quantize_safetensors(in_files, out_file, q);
        }
    }

    if in_files.len() != 1 {
        candle::bail!("only a single in-file can be used when quantizing gguf files")
    }

    // Open the out file early so as to fail directly on missing directories etc.
    let mut out_file = std::fs::File::create(out_file)?;
    let mut in_ = std::fs::File::open(&in_files[0])?;
    let content = gguf_file::Content::read(&mut in_)?;
    println!("tensors: {}", content.tensor_infos.len());

    let dtype = q.dtype();
    let qtensors = content
        .tensor_infos
        .par_iter()
        .map(|(name, _)| {
            println!("  quantizing {name}");
            let mut in_file = std::fs::File::open(&in_files[0])?;
            let tensor = content.tensor(&mut in_file, name, device)?;
            let tensor = qmode.quantize(name, tensor, dtype)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    let metadata = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut out_file, metadata.as_slice(), &qtensors)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    match args.command {
        Command::Ls {
            files,
            format,
            verbose,
        } => {
            let multiple_files = files.len() > 1;
            for file in files.iter() {
                if multiple_files {
                    println!("--- {file:?} ---");
                }
                run_ls(file, format.clone(), verbose, &device)?
            }
        }
        Command::Print {
            file,
            names,
            format,
            full,
            line_width,
        } => run_print(&file, names, format, full, line_width, &device)?,
        Command::Quantize {
            in_file,
            out_file,
            quantization,
            mode,
        } => run_quantize(&in_file, out_file, quantization, mode, &device)?,
        Command::Dequantize { in_file, out_file } => run_dequantize(in_file, out_file, &device)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Shape, Tensor};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP_DIR: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tensor-tools-q8-h-{}-{id}-{name}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn qtensor(values: Vec<f32>, shape: impl Into<Shape>, dtype: GgmlDType) -> Result<QTensor> {
        let tensor = Tensor::from_vec(values, shape, &Device::Cpu)?;
        QTensor::quantize(&tensor, dtype)
    }

    fn write_synthetic_gguf(
        path: &std::path::Path,
        version: gguf_file::VersionedMagic,
    ) -> Result<()> {
        let mut q_values = vec![0.0; 64 * 64];
        for row in q_values.chunks_exact_mut(64) {
            row[..32].fill(1.0);
            row[32..].fill(8.0);
        }
        let tensors = [
            (
                "position_embd.weight".to_string(),
                qtensor(vec![7.0; 128 * 64], (128, 64), GgmlDType::F32)?,
            ),
            (
                "blk.0.attn_v.weight".to_string(),
                qtensor(vec![4.0; 64 * 64], (64, 64), GgmlDType::BF16)?,
            ),
            (
                "blk.0.attn_norm.weight".to_string(),
                qtensor(vec![5.0; 64], (64,), GgmlDType::F32)?,
            ),
            (
                "blk.0.attn_q.weight".to_string(),
                qtensor(q_values, (64, 64), GgmlDType::F32)?,
            ),
            (
                "blk.0.attn_q.bias".to_string(),
                qtensor(vec![6.0; 64], (64,), GgmlDType::F32)?,
            ),
            (
                "blk.0.ffn_up.weight".to_string(),
                qtensor(vec![3.0; 128 * 64], (128, 64), GgmlDType::F32)?,
            ),
            (
                "blk.0.attn_k.weight".to_string(),
                qtensor(vec![2.0; 64 * 64], (64, 64), GgmlDType::F16)?,
            ),
        ];
        let metadata = [
            (
                "custom.tensor-tools.test".to_string(),
                gguf_file::Value::String("preserve-me".to_string()),
            ),
            ("general.alignment".to_string(), gguf_file::Value::U32(64)),
            ("general.file_type".to_string(), gguf_file::Value::U32(1)),
            (
                "general.name".to_string(),
                gguf_file::Value::String("synthetic-q8-h".to_string()),
            ),
            (
                "general.quantization_version".to_string(),
                gguf_file::Value::U32(1),
            ),
            (
                "general.architecture".to_string(),
                gguf_file::Value::String("gpt2".to_string()),
            ),
        ];
        let tensor_refs = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let metadata_refs = metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();
        let mut file = std::fs::File::create(path)?;
        gguf_file::write_with_version(&mut file, version, &metadata_refs, &tensor_refs)
    }

    fn tensor_bytes(path: &std::path::Path, name: &str) -> Result<Vec<u8>> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        Ok(content
            .tensor(&mut file, name, &Device::Cpu)?
            .data()?
            .into_owned())
    }

    fn written_tensor_type_id(dtype: GgmlDType) -> Result<u32> {
        let tensor = qtensor(vec![1.0; 32], (32,), dtype)?;
        let mut cursor = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut cursor, &[], &[("weight", &tensor)])?;
        let bytes = cursor.into_inner();
        let mut offset = 24usize;
        let name_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8 + name_len;
        let dimensions = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4 + dimensions * 8;
        Ok(u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().unwrap(),
        ))
    }

    #[test]
    fn q8_h_cli_and_numeric_ids_match_pinned_reference() -> Result<()> {
        let args = Args::try_parse_from([
            "tensor-tools",
            "quantize",
            "input.gguf",
            "--out-file",
            "output.gguf",
            "--quantization",
            "q8_h1",
            "--mode",
            "aqua",
        ])
        .unwrap();
        assert!(matches!(
            args.command,
            Command::Quantize {
                quantization: Quantization::Q8H1,
                mode: QuantizationMode::Aqua,
                ..
            }
        ));

        assert_eq!(written_tensor_type_id(GgmlDType::Q8H1)?, 39);
        assert_eq!(LLAMA_FTYPE_MOSTLY_Q8_H1, 38);
        assert_eq!(written_tensor_type_id(GgmlDType::Q8HP1)?, 41);
        assert_eq!(LLAMA_FTYPE_MOSTLY_Q8_HP1, 40);
        assert_eq!(GGML_QNT_VERSION, 2);
        Ok(())
    }

    #[test]
    fn aqua_selector_matches_pinned_matrix_policy() {
        let matrix = Shape::from((64, 64));
        let vector = Shape::from((64,));
        for name in [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert_eq!(
                classify_aqua_quantization(name, &matrix),
                AquaQuantizationAction::Quantize,
                "{name}"
            );
        }

        assert_eq!(
            classify_aqua_quantization("blk.0.attn_q.bias", &vector),
            AquaQuantizationAction::Copy(AquaCopyReason::NotWeight)
        );
        assert_eq!(
            classify_aqua_quantization("blk.0.attn_q.weight", &vector),
            AquaQuantizationAction::Copy(AquaCopyReason::NotMatrix)
        );
        for (name, reason) in [
            ("blk.0.attn_norm.weight", AquaCopyReason::Normalization),
            ("position_embd.weight", AquaCopyReason::PositionEmbedding),
            ("token_types.weight", AquaCopyReason::TokenTypeEmbedding),
            ("blk.0.ffn_gate_inp.weight", AquaCopyReason::ExpertGate),
            ("blk.0.ssm_conv1d.weight", AquaCopyReason::SsmConv1d),
            ("blk.0.time_mix_w1.weight", AquaCopyReason::RwkvTimeMix),
            (
                "blk.0.attn_rel_b.weight",
                AquaCopyReason::RelativeAttentionBias,
            ),
        ] {
            assert_eq!(
                classify_aqua_quantization(name, &matrix),
                AquaQuantizationAction::Copy(reason),
                "{name}"
            );
        }
    }

    #[test]
    fn aqua_conversion_is_deterministic_and_preserves_gguf_contracts() -> Result<()> {
        let dir = temp_dir("synthetic");
        for version in [
            gguf_file::VersionedMagic::GgufV2,
            gguf_file::VersionedMagic::GgufV3,
        ] {
            let suffix = match version {
                gguf_file::VersionedMagic::GgufV2 => "v2",
                gguf_file::VersionedMagic::GgufV3 => "v3",
                gguf_file::VersionedMagic::GgufV1 => unreachable!(),
            };
            let input = dir.join(format!("input-{suffix}.gguf"));
            let output1 = dir.join(format!("output-{suffix}-1.gguf"));
            let output2 = dir.join(format!("output-{suffix}-2.gguf"));
            let output_h1 = dir.join(format!("output-{suffix}-h1.gguf"));
            write_synthetic_gguf(&input, version)?;

            let summary1 =
                run_quantize_aqua_gguf(&input, &output1, Quantization::Q8HP1, &Device::Cpu)?;
            let summary2 =
                run_quantize_aqua_gguf(&input, &output2, Quantization::Q8HP1, &Device::Cpu)?;
            assert_eq!(std::fs::read(&output1)?, std::fs::read(&output2)?);
            assert_eq!(summary1.total_tensors, 7);
            assert_eq!(summary1.quantized_tensors, 4);
            assert_eq!(summary1.copied_tensors, 3);
            assert_eq!(summary1, summary2);

            let mut output_file = std::fs::File::open(&output1)?;
            let output = gguf_file::Content::read(&mut output_file)?;
            assert_eq!(output.magic, version);
            let mut input_file = std::fs::File::open(&input)?;
            let input_content = gguf_file::Content::read(&mut input_file)?;
            for (name, input_info) in &input_content.tensor_infos {
                assert_eq!(
                    output.tensor_infos.get(name).unwrap().shape.dims(),
                    input_info.shape.dims()
                );
            }
            assert!(matches!(
                output.metadata.get("general.alignment"),
                Some(gguf_file::Value::U32(64))
            ));
            assert!(matches!(
                output.metadata.get("general.file_type"),
                Some(gguf_file::Value::U32(LLAMA_FTYPE_MOSTLY_Q8_HP1))
            ));
            assert!(matches!(
                output.metadata.get("general.quantization_version"),
                Some(gguf_file::Value::U32(GGML_QNT_VERSION))
            ));
            assert!(matches!(
                output.metadata.get("custom.tensor-tools.test"),
                Some(gguf_file::Value::String(value)) if value == "preserve-me"
            ));
            for name in [
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.ffn_up.weight",
            ] {
                let info = output.tensor_infos.get(name).unwrap();
                assert_eq!(info.ggml_dtype, GgmlDType::Q8HP1);
            }
            for name in [
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.bias",
                "position_embd.weight",
            ] {
                assert_eq!(tensor_bytes(&input, name)?, tensor_bytes(&output1, name)?);
            }

            let q_bytes = tensor_bytes(&output1, "blk.0.attn_q.weight")?;
            assert_eq!(&q_bytes[0..32], &[64; 32]);
            assert_eq!(i16::from_ne_bytes(q_bytes[32..34].try_into().unwrap()), 0);
            assert_eq!(&q_bytes[34..36], &[0, 0]);
            assert_eq!(
                f32::from_ne_bytes(q_bytes[36..40].try_into().unwrap()),
                1.0 / 64.0
            );
            assert_eq!(&q_bytes[40..72], &[64; 32]);
            assert_eq!(i16::from_ne_bytes(q_bytes[72..74].try_into().unwrap()), 3);
            assert_eq!(
                f32::from_ne_bytes(q_bytes[76..80].try_into().unwrap()),
                1.0 / 64.0
            );

            run_quantize_aqua_gguf(&input, &output_h1, Quantization::Q8H1, &Device::Cpu)?;
            let mut h1_file = std::fs::File::open(&output_h1)?;
            let h1 = gguf_file::Content::read(&mut h1_file)?;
            assert!(matches!(
                h1.metadata.get("general.file_type"),
                Some(gguf_file::Value::U32(LLAMA_FTYPE_MOSTLY_Q8_H1))
            ));
            assert_eq!(
                h1.tensor_infos
                    .get("blk.0.attn_q.weight")
                    .unwrap()
                    .ggml_dtype,
                GgmlDType::Q8H1
            );
            assert!(!std::path::PathBuf::from(format!("{}.tmp", output1.display())).exists());
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn aqua_conversion_rejects_unsafe_or_unsupported_inputs() -> Result<()> {
        let dir = temp_dir("contracts");
        let input = dir.join("input.gguf");
        let output = dir.join("output.gguf");
        write_synthetic_gguf(&input, gguf_file::VersionedMagic::GgufV2)?;

        assert!(run_quantize(
            std::slice::from_ref(&input),
            output.clone(),
            Quantization::Q8HP1,
            QuantizationMode::Llama,
            &Device::Cpu,
        )
        .is_err());
        assert!(run_quantize(
            &[input.clone(), input.clone()],
            output.clone(),
            Quantization::Q8HP1,
            QuantizationMode::Aqua,
            &Device::Cpu,
        )
        .is_err());
        assert!(run_quantize_aqua_gguf(&input, &input, Quantization::Q8HP1, &Device::Cpu).is_err());
        assert!(validate_aqua_gguf_version(gguf_file::VersionedMagic::GgufV1).is_err());

        let split = dir.join("split.gguf");
        let split_output = dir.join("split-output.gguf");
        let split_tensor = qtensor(vec![1.0; 64 * 64], (64, 64), GgmlDType::F32)?;
        let split_metadata = gguf_file::Value::U16(2);
        let mut file = std::fs::File::create(&split)?;
        gguf_file::write(
            &mut file,
            &[("split.count", &split_metadata)],
            &[("blk.0.attn_q.weight", &split_tensor)],
        )?;
        assert!(
            run_quantize_aqua_gguf(&split, &split_output, Quantization::Q8HP1, &Device::Cpu)
                .is_err()
        );

        let bad_width = dir.join("bad-width.gguf");
        let bad_width_output = dir.join("bad-width-output.gguf");
        let bad_width_tensor = qtensor(vec![1.0; 64 * 33], (64, 33), GgmlDType::F32)?;
        let mut file = std::fs::File::create(&bad_width)?;
        gguf_file::write(
            &mut file,
            &[],
            &[("blk.0.attn_q.weight", &bad_width_tensor)],
        )?;
        assert!(run_quantize_aqua_gguf(
            &bad_width,
            &bad_width_output,
            Quantization::Q8HP1,
            &Device::Cpu
        )
        .is_err());

        let quantized = dir.join("quantized.gguf");
        let quantized_output = dir.join("quantized-output.gguf");
        let q8_0 = qtensor(vec![1.0; 64 * 64], (64, 64), GgmlDType::Q8_0)?;
        let mut file = std::fs::File::create(&quantized)?;
        gguf_file::write(&mut file, &[], &[("blk.0.attn_q.weight", &q8_0)])?;
        assert!(run_quantize_aqua_gguf(
            &quantized,
            &quantized_output,
            Quantization::Q8HP1,
            &Device::Cpu
        )
        .is_err());

        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn aqua_conversion_preserves_already_target_raw_tensor() -> Result<()> {
        let dir = temp_dir("already-target");
        let input = dir.join("input.gguf");
        let output = dir.join("output.gguf");
        let target = qtensor(vec![8.0; 64 * 64], (64, 64), GgmlDType::Q8HP1)?;
        let expected = target.data()?.into_owned();
        let mut file = std::fs::File::create(&input)?;
        gguf_file::write(&mut file, &[], &[("output.weight", &target)])?;
        drop(file);

        run_quantize_aqua_gguf(&input, &output, Quantization::Q8HP1, &Device::Cpu)?;
        assert_eq!(tensor_bytes(&output, "output.weight")?, expected);
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }
}
