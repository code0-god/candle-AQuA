use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::{borrow::Cow, sync::OnceLock};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes};

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

fn as_t_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
    /// Lazily initialized storage for repacked quantized data. Currently raw bits, could be `QStorage` in the future.
    /// Not always used.
    #[allow(dead_code)]
    repacked_qs: OnceLock<Option<Vec<u8>>>,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => Ok(dtype.cpu_zeros(elem_count)),
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
            #[cfg(feature = "aqua")]
            Device::Aqua(_) => crate::bail!("quantized storage is not supported on Aqua devices"),
        }
    }
}

pub trait RawQuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn block_size(&self) -> usize;
    fn data(&self) -> &[u8];
}

struct RawBlockStorage<T> {
    blocks: Vec<T>,
    dtype: GgmlDType,
    block_size: usize,
}

impl<T> RawQuantizedType for RawBlockStorage<T>
where
    T: IntoBytes + Immutable + Send + Sync,
{
    fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn data(&self) -> &[u8] {
        self.blocks.as_bytes()
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    CpuRaw(Box<dyn RawQuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

fn raw_storage_from_blocks<T>(blocks: Vec<T>, dtype: GgmlDType, block_size: usize) -> QStorage
where
    T: IntoBytes + Immutable + Send + Sync + 'static,
{
    QStorage::CpuRaw(Box::new(RawBlockStorage {
        blocks,
        dtype,
        block_size,
    }))
}

fn raw_storage_from_data<T>(data: &[u8], dtype: GgmlDType, block_size: usize) -> Result<QStorage>
where
    T: FromBytes + IntoBytes + Immutable + Send + Sync + 'static,
{
    // Keep raw block bytes unchanged; scalar endian interpretation belongs to later math support.
    let type_size = std::mem::size_of::<T>();
    if !data.len().is_multiple_of(type_size) {
        crate::bail!(
            "{dtype:?} raw data length {} is not divisible by block byte size {type_size}",
            data.len()
        )
    }
    let blocks = data
        .chunks_exact(type_size)
        .map(|bytes| {
            T::read_from_bytes(bytes)
                .map_err(|_| crate::Error::Msg(format!("failed to copy one {dtype:?} raw block")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(raw_storage_from_blocks(blocks, dtype, block_size))
}

fn raw_zero_storage<T>(block_count: usize, dtype: GgmlDType, block_size: usize) -> QStorage
where
    T: FromZeros + IntoBytes + Immutable + Clone + Send + Sync + 'static,
{
    raw_storage_from_blocks(vec![T::new_zeroed(); block_count], dtype, block_size)
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        let data: &[u8] = &data;
        match device {
            Device::Cpu => dtype.from_data(Cow::Borrowed(data)),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(data)),
                GgmlDType::Q8H1 | GgmlDType::Q8HP1 => {
                    crate::bail!("{dtype:?} raw storage is not supported on Metal")
                }
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),
                GgmlDType::Q8H1 | GgmlDType::Q8HP1 => {
                    crate::bail!("{dtype:?} raw storage is not supported on CUDA")
                }
            },
            #[cfg(feature = "aqua")]
            Device::Aqua(_) => crate::bail!("quantized storage is not supported on Aqua devices"),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::CpuRaw(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::CpuRaw(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_) | QStorage::CpuRaw(_) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::CpuRaw(storage) => storage.data().len(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::CpuRaw(storage) => {
                crate::bail!(
                    "dequantization is not implemented for {:?}",
                    storage.dtype()
                )
            }
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::CpuRaw(storage) => Ok(Cow::Borrowed(storage.data())),
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::CpuRaw(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr_with_guard(stream),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::CpuRaw(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

/// Logical element count for Q8_H GGUF blocks.
pub const QK8_H: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Q8H1,
    Q8HP1,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            // https://github.com/ajou-aisa/llama.cpp-gemmini/blob/d5e76be1fca91314c5a0745038b3cedbbdbed13d/ggml/include/ggml.h#L391-L393
            39 => Self::Q8H1,
            41 => Self::Q8HP1,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
            Self::Q8H1 => 39,
            Self::Q8HP1 => 41,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> QStorage {
        let storage: Box<dyn QuantizedType> = match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::Q8H1 => return raw_zero_storage::<BlockQ8H1>(elem_count / QK8_H, *self, QK8_H),
            Self::Q8HP1 => return raw_zero_storage::<BlockQ8HP1>(elem_count / QK8_H, *self, QK8_H),
        };
        QStorage::Cpu(storage)
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Result<QStorage> {
        let data: &[u8] = &data;
        let storage: Box<dyn QuantizedType> = match self {
            Self::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
            Self::Q8H1 => return raw_storage_from_data::<BlockQ8H1>(data, *self, QK8_H),
            Self::Q8HP1 => return raw_storage_from_data::<BlockQ8HP1>(data, *self, QK8_H),
        };
        Ok(QStorage::Cpu(storage))
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
            Self::Q8H1 => std::mem::size_of::<BlockQ8H1>(),
            Self::Q8HP1 => std::mem::size_of::<BlockQ8HP1>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q8H1 | Self::Q8HP1 => QK8_H,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_h1_fixture() -> [u8; 44] {
        let mut bytes = [0u8; 44];
        for (index, byte) in bytes[..32].iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(17).wrapping_add(3);
        }
        bytes[32] = 0x7d;
        bytes[33..36].copy_from_slice(&[0xa1, 0xb2, 0xc3]);
        bytes[36..40].copy_from_slice(&f32::from_bits(0x3eaaaaab).to_ne_bytes());
        bytes[40..42].copy_from_slice(&0xbeefu16.to_ne_bytes());
        bytes[42..44].copy_from_slice(&[0xd4, 0xe5]);
        bytes
    }

    fn q8_hp1_fixture() -> [u8; 40] {
        let mut bytes = [0u8; 40];
        for (index, byte) in bytes[..32].iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(29).wrapping_add(11);
        }
        bytes[32..34].copy_from_slice(&(-1234i16).to_ne_bytes());
        bytes[34..36].copy_from_slice(&[0xc7, 0xd8]);
        bytes[36..40].copy_from_slice(&f32::from_bits(0x3f123456).to_ne_bytes());
        bytes
    }

    fn q8_h_tensor_blocks<T: FromBytes>(tensor: &QTensor) -> Result<Vec<T>> {
        let data = tensor.data()?;
        data.chunks_exact(std::mem::size_of::<T>())
            .map(|bytes| {
                T::read_from_bytes(bytes)
                    .map_err(|_| crate::Error::Msg("invalid Q8_H test block size".to_string()))
            })
            .collect()
    }

    #[test]
    fn q8_h1_dtype_id_round_trip() -> Result<()> {
        assert_eq!(GgmlDType::from_u32(39)?, GgmlDType::Q8H1);
        assert_eq!(GgmlDType::Q8H1.to_u32(), 39);
        Ok(())
    }

    #[test]
    fn q8_hp1_dtype_id_round_trip() -> Result<()> {
        assert_eq!(GgmlDType::from_u32(41)?, GgmlDType::Q8HP1);
        assert_eq!(GgmlDType::Q8HP1.to_u32(), 41);
        Ok(())
    }

    #[test]
    fn q8_h1_block_geometry() {
        assert_eq!(GgmlDType::Q8H1.block_size(), 32);
        assert_eq!(GgmlDType::Q8H1.type_size(), 44);
    }

    #[test]
    fn q8_hp1_block_geometry() {
        assert_eq!(GgmlDType::Q8HP1.block_size(), 32);
        assert_eq!(GgmlDType::Q8HP1.type_size(), 40);
    }

    #[test]
    fn unsupported_q8_h_custom_ids() {
        assert!(GgmlDType::from_u32(40).is_err());
        assert!(GgmlDType::from_u32(42).is_err());
    }

    #[test]
    fn q8_h1_layout_matches_gguf_abi() {
        assert_eq!(std::mem::size_of::<BlockQ8H1>(), 44);
        assert_eq!(std::mem::align_of::<BlockQ8H1>(), 4);
        assert_eq!(std::mem::offset_of!(BlockQ8H1, qs), 0);
        assert_eq!(std::mem::offset_of!(BlockQ8H1, c_b), 32);
        assert_eq!(std::mem::offset_of!(BlockQ8H1, s_rf), 36);
        assert_eq!(std::mem::offset_of!(BlockQ8H1, r), 40);
    }

    #[test]
    fn q8_hp1_layout_matches_gguf_abi() {
        assert_eq!(std::mem::size_of::<BlockQ8HP1>(), 40);
        assert_eq!(std::mem::align_of::<BlockQ8HP1>(), 4);
        assert_eq!(std::mem::offset_of!(BlockQ8HP1, qs), 0);
        assert_eq!(std::mem::offset_of!(BlockQ8HP1, m), 32);
        assert_eq!(std::mem::offset_of!(BlockQ8HP1, padding), 34);
        assert_eq!(std::mem::offset_of!(BlockQ8HP1, channel_scale), 36);
    }

    #[test]
    fn q8_h1_cpu_from_data_preserves_bytes() -> Result<()> {
        let bytes = q8_h1_fixture();
        let mut unaligned = vec![0xff];
        unaligned.extend_from_slice(&bytes);
        let raw = &unaligned[1..];
        assert_ne!(raw.as_ptr() as usize % std::mem::align_of::<BlockQ8H1>(), 0);
        let storage = QStorage::from_data(Cow::Borrowed(raw), &Device::Cpu, GgmlDType::Q8H1)?;
        let tensor = QTensor::new(storage, (QK8_H,))?;
        assert_eq!(tensor.dtype(), GgmlDType::Q8H1);
        assert_eq!(tensor.storage_size_in_bytes(), bytes.len());
        assert_eq!(tensor.data()?.as_ref(), bytes);
        Ok(())
    }

    #[test]
    fn q8_hp1_cpu_from_data_preserves_bytes() -> Result<()> {
        let bytes = q8_hp1_fixture();
        let storage =
            QStorage::from_data(Cow::Owned(bytes.to_vec()), &Device::Cpu, GgmlDType::Q8HP1)?;
        let tensor = QTensor::new(storage, (QK8_H,))?;
        assert_eq!(tensor.dtype(), GgmlDType::Q8HP1);
        assert_eq!(tensor.storage_size_in_bytes(), bytes.len());
        assert_eq!(tensor.data()?.as_ref(), bytes);
        Ok(())
    }

    #[test]
    fn q8_h1_cpu_zero_storage_geometry() -> Result<()> {
        let storage = Device::Cpu.qzeros(2 * QK8_H, GgmlDType::Q8H1)?;
        let tensor = QTensor::new(storage, (2 * QK8_H,))?;
        assert_eq!(tensor.storage_size_in_bytes(), 2 * 44);
        assert_eq!(tensor.data()?.as_ref(), [0u8; 88]);
        Ok(())
    }

    #[test]
    fn q8_hp1_cpu_zero_storage_geometry() -> Result<()> {
        let storage = Device::Cpu.qzeros(2 * QK8_H, GgmlDType::Q8HP1)?;
        let tensor = QTensor::new(storage, (2 * QK8_H,))?;
        assert_eq!(tensor.storage_size_in_bytes(), 2 * 40);
        assert_eq!(tensor.data()?.as_ref(), [0u8; 80]);
        Ok(())
    }

    #[test]
    fn q8_h_float_helpers_match_c_semantics() {
        assert_eq!(k_quants::ilogb_positive_f32(1.0), 0);
        assert_eq!(k_quants::ilogb_positive_f32(2.0), 1);
        assert_eq!(k_quants::ilogb_positive_f32(0.5), -1);
        assert_eq!(k_quants::ilogb_positive_f32(f32::MIN_POSITIVE), -126);
        assert_eq!(k_quants::ilogb_positive_f32(f32::from_bits(1)), -149);
        assert_eq!(
            k_quants::ilogb_positive_f32(f32::from_bits(0x007f_ffff)),
            -127
        );

        assert_eq!(k_quants::pow2_f32(0), 1.0);
        assert_eq!(k_quants::pow2_f32(-126), f32::MIN_POSITIVE);
        assert_eq!(k_quants::pow2_f32(-149).to_bits(), 1);
        assert_eq!(k_quants::pow2_f32(-150), 0.0);
        assert_eq!(k_quants::pow2_f32(127).to_bits(), 0x7f00_0000);
        assert!(k_quants::pow2_f32(128).is_infinite());
    }

    #[test]
    fn q8_h_row_shape_validation() {
        let mut h1 = vec![BlockQ8H1::new_zeroed()];
        assert!(k_quants::quantize_row_q8_h1_ref(&[], &mut []).is_err());
        assert!(k_quants::quantize_row_q8_h1_ref(&[0.0; 31], &mut h1).is_err());
        assert!(k_quants::quantize_row_q8_h1_ref(&[0.0; 32], &mut []).is_err());

        let mut hp1 = vec![BlockQ8HP1::new_zeroed()];
        assert!(k_quants::quantize_row_q8_hp1_ref(&[], &mut []).is_err());
        assert!(k_quants::quantize_row_q8_hp1_ref(&[0.0; 31], &mut hp1).is_err());
        assert!(k_quants::quantize_row_q8_hp1_ref(&[0.0; 32], &mut []).is_err());
    }

    #[test]
    fn q8_h1_quantizes_all_zero_row_like_c_reference() -> Result<()> {
        let mut blocks = vec![BlockQ8H1::new_zeroed(); 2];
        k_quants::quantize_row_q8_h1_ref(&[0.0; 64], &mut blocks)?;
        for block in blocks {
            assert_eq!(block.qs, [0; QK8_H]);
            assert_eq!(block.c_b, 0);
            assert_eq!(block._padding, [0; 3]);
            assert_eq!(block.s_rf, 0.0);
            assert_eq!(block.r, 0);
            assert_eq!(block._tail_padding, [0; 2]);
        }
        Ok(())
    }

    #[test]
    fn q8_h1_quantizes_equal_magnitude_blocks_like_c_reference() -> Result<()> {
        let mut values = vec![1.0; QK8_H];
        values.extend(vec![-1.0; QK8_H]);
        let mut blocks = vec![BlockQ8H1::new_zeroed(); 2];
        k_quants::quantize_row_q8_h1_ref(&values, &mut blocks)?;

        let expected_scale = f16::from_f32(1.0 / 127.0).to_f32();
        assert_eq!(blocks[0].qs, [127; QK8_H]);
        assert_eq!(blocks[1].qs, [-127; QK8_H]);
        for block in blocks {
            assert_eq!(block.c_b, 0);
            assert_eq!(block.s_rf.to_bits(), expected_scale.to_bits());
            assert_eq!(block.r, 1);
            assert_eq!(block._padding, [0; 3]);
            assert_eq!(block._tail_padding, [0; 2]);
        }
        Ok(())
    }

    #[test]
    fn q8_h1_zero_block_participates_in_scale_range() -> Result<()> {
        let mut values = vec![0.0; QK8_H];
        values.extend(vec![1.0; QK8_H]);
        let mut blocks = vec![BlockQ8H1::new_zeroed(); 2];
        k_quants::quantize_row_q8_h1_ref(&values, &mut blocks)?;

        let block_scale = f16::from_f32(1.0 / 127.0).to_f32();
        let expected_row_scale = block_scale / 255.0;
        assert_eq!(blocks[0].qs, [0; QK8_H]);
        assert_eq!(blocks[1].qs, [127; QK8_H]);
        assert_eq!(blocks[0].s_rf.to_bits(), expected_row_scale.to_bits());
        assert_eq!(blocks[1].s_rf.to_bits(), expected_row_scale.to_bits());
        assert_eq!(blocks[0].r, 0);
        assert_eq!(blocks[1].r, 0);
        assert_eq!(blocks[0].c_b, 0);
        assert_eq!(blocks[1].c_b, 255);
        Ok(())
    }

    #[test]
    fn q8_hp1_quantizes_constant_one_like_c_reference() -> Result<()> {
        let mut blocks = vec![BlockQ8HP1::new_zeroed()];
        k_quants::quantize_row_q8_hp1_ref(&[1.0; QK8_H], &mut blocks)?;
        assert_eq!(blocks[0].qs, [64; QK8_H]);
        assert_eq!(blocks[0].m, 0);
        assert_eq!(blocks[0].padding, [0; 2]);
        assert_eq!(blocks[0].channel_scale, 0.015625);
        Ok(())
    }

    #[test]
    fn q8_hp1_quantizes_left_shift_hierarchy_like_c_reference() -> Result<()> {
        let mut values = vec![1.0; QK8_H];
        values.extend(vec![8.0; QK8_H]);
        let mut blocks = vec![BlockQ8HP1::new_zeroed(); 2];
        k_quants::quantize_row_q8_hp1_ref(&values, &mut blocks)?;
        assert_eq!(blocks[0].qs, [64; QK8_H]);
        assert_eq!(blocks[1].qs, [64; QK8_H]);
        assert_eq!([blocks[0].m, blocks[1].m], [0, 3]);
        assert!(blocks.iter().all(|block| block.m >= 0));
        assert!(blocks.iter().all(|block| block.channel_scale == 0.015625));
        assert!(blocks.iter().all(|block| block.padding == [0; 2]));
        Ok(())
    }

    #[test]
    fn q8_hp1_zero_block_keeps_shared_row_scale() -> Result<()> {
        let mut values = vec![0.0; QK8_H];
        values.extend(vec![8.0; QK8_H]);
        let mut blocks = vec![BlockQ8HP1::new_zeroed(); 2];
        k_quants::quantize_row_q8_hp1_ref(&values, &mut blocks)?;
        assert_eq!(blocks[0].qs, [0; QK8_H]);
        assert_eq!(blocks[0].m, i16::MIN);
        assert_eq!(blocks[0].channel_scale, 0.125);
        assert_eq!(blocks[1].qs, [64; QK8_H]);
        assert_eq!(blocks[1].m, 0);
        assert_eq!(blocks[1].channel_scale, 0.125);
        Ok(())
    }

    #[test]
    fn q8_hp1_quantizes_all_zero_row_like_c_reference() -> Result<()> {
        let mut blocks = vec![BlockQ8HP1::new_zeroed(); 2];
        k_quants::quantize_row_q8_hp1_ref(&[0.0; 64], &mut blocks)?;
        for block in blocks {
            assert_eq!(block.qs, [0; QK8_H]);
            assert_eq!(block.m, i16::MIN);
            assert_eq!(block.padding, [0; 2]);
            assert_eq!(block.channel_scale, 0.0);
        }
        Ok(())
    }

    #[test]
    fn q8_hp1_rejects_invalid_c_reference_inputs() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [0.0; QK8_H];
            values[0] = invalid;
            let mut blocks = vec![BlockQ8HP1::new_zeroed()];
            assert!(k_quants::quantize_row_q8_hp1_ref(&values, &mut blocks).is_err());
        }

        let mut blocks = vec![BlockQ8HP1::new_zeroed()];
        assert!(
            k_quants::quantize_row_q8_hp1_ref(&[f32::from_bits(1); QK8_H], &mut blocks).is_err()
        );
    }

    #[test]
    fn q8_hp1_rounds_half_away_from_zero() -> Result<()> {
        let mut values = [0.0; QK8_H];
        values[0] = 1.0;
        values[1] = 1.0 / 128.0;
        values[2] = -1.0 / 128.0;
        let mut blocks = vec![BlockQ8HP1::new_zeroed()];
        k_quants::quantize_row_q8_hp1_ref(&values, &mut blocks)?;
        assert_eq!(blocks[0].qs[0], 64);
        assert_eq!(blocks[0].qs[1], 1);
        assert_eq!(blocks[0].qs[2], -1);
        Ok(())
    }

    #[test]
    fn q8_h1_qtensor_quantize_isolates_rows() -> Result<()> {
        let mut values = Vec::new();
        for (first, second) in [(1.0, -1.0), (8.0, -8.0)] {
            values.extend(vec![first; QK8_H]);
            values.extend(vec![second; QK8_H]);
        }
        let tensor = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
        let quantized = QTensor::quantize(&tensor, GgmlDType::Q8H1)?;
        let blocks = q8_h_tensor_blocks::<BlockQ8H1>(&quantized)?;

        assert_eq!(quantized.shape().dims(), [2, 64]);
        assert_eq!(quantized.storage_size_in_bytes(), 4 * 44);
        assert_eq!(blocks.len(), 4);
        let row0_scale = f16::from_f32(1.0 / 127.0).to_f32();
        let row1_scale = f16::from_f32(8.0 / 127.0).to_f32();
        for block in &blocks[..2] {
            assert_eq!(block.s_rf.to_bits(), row0_scale.to_bits());
            assert_eq!(block.r, 1);
        }
        for block in &blocks[2..] {
            assert_eq!(block.s_rf.to_bits(), row1_scale.to_bits());
            assert_eq!(block.r, 1);
        }
        assert_ne!(blocks[0].s_rf.to_bits(), blocks[2].s_rf.to_bits());
        Ok(())
    }

    #[test]
    fn q8_hp1_qtensor_quantize_isolates_rows() -> Result<()> {
        let mut values = Vec::new();
        for (first, second) in [(1.0, 8.0), (4.0, 32.0)] {
            values.extend(vec![first; QK8_H]);
            values.extend(vec![second; QK8_H]);
        }
        let tensor = Tensor::from_vec(values, (2, 64), &Device::Cpu)?;
        let quantized = QTensor::quantize(&tensor, GgmlDType::Q8HP1)?;
        let blocks = q8_h_tensor_blocks::<BlockQ8HP1>(&quantized)?;

        assert_eq!(quantized.shape().dims(), [2, 64]);
        assert_eq!(quantized.storage_size_in_bytes(), 4 * 40);
        assert_eq!(blocks.len(), 4);
        assert_eq!([blocks[0].m, blocks[1].m], [0, 3]);
        assert_eq!([blocks[2].m, blocks[3].m], [0, 3]);
        assert!(blocks[..2]
            .iter()
            .all(|block| block.channel_scale == 1.0 / 64.0));
        assert!(blocks[2..]
            .iter()
            .all(|block| block.channel_scale == 1.0 / 16.0));
        assert!(blocks.iter().all(|block| block.qs == [64; QK8_H]));
        Ok(())
    }

    #[test]
    fn q8_hp1_qtensor_quantize_uses_higher_rank_rows() -> Result<()> {
        let mut values = Vec::new();
        for base in [1.0, 2.0, 4.0, 8.0] {
            values.extend(vec![base; QK8_H]);
            values.extend(vec![base * 8.0; QK8_H]);
        }
        let tensor = Tensor::from_vec(values, (2, 2, 64), &Device::Cpu)?;
        let quantized = QTensor::quantize(&tensor, GgmlDType::Q8HP1)?;
        let blocks = q8_h_tensor_blocks::<BlockQ8HP1>(&quantized)?;

        assert_eq!(quantized.shape().dims(), [2, 2, 64]);
        assert_eq!(quantized.storage_size_in_bytes(), 8 * 40);
        assert_eq!(blocks.len(), 8);
        for (row, expected_scale) in [1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0]
            .into_iter()
            .enumerate()
        {
            let row_blocks = &blocks[row * 2..row * 2 + 2];
            assert_eq!([row_blocks[0].m, row_blocks[1].m], [0, 3]);
            assert!(row_blocks
                .iter()
                .all(|block| block.channel_scale == expected_scale));
        }
        Ok(())
    }

    #[test]
    fn q8_h_qtensor_quantize_zeroes_padding_bytes() -> Result<()> {
        for (dtype, block_size, padding_ranges) in [
            (GgmlDType::Q8H1, 44, &[(33, 36), (42, 44)][..]),
            (GgmlDType::Q8HP1, 40, &[(34, 36)][..]),
        ] {
            let tensor = Tensor::from_vec(vec![1.0; 128], (2, 64), &Device::Cpu)?;
            let quantized = QTensor::quantize(&tensor, dtype)?;
            let data = quantized.data()?;
            for block in data.chunks_exact(block_size) {
                for &(start, end) in padding_ranges {
                    assert_eq!(&block[start..end], vec![0; end - start]);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn q8_h_reference_golden_bytes_match_pinned_c() -> Result<()> {
        // ajou-aisa/llama.cpp-gemmini@d5e76be1fca91314c5a0745038b3cedbbdbed13d
        let mut h1_values = vec![1.0; QK8_H];
        h1_values.extend(vec![-1.0; QK8_H]);
        let mut h1_blocks = vec![BlockQ8H1::new_zeroed(); 2];
        k_quants::quantize_row_q8_h1_ref(&h1_values, &mut h1_blocks)?;
        let h1_scale = f16::from_f32(1.0 / 127.0).to_f32();
        let mut expected_h1 = Vec::new();
        for q in [127i8, -127i8] {
            expected_h1.extend(vec![q as u8; QK8_H]);
            expected_h1.push(0);
            expected_h1.extend([0; 3]);
            expected_h1.extend(h1_scale.to_ne_bytes());
            expected_h1.extend(1u16.to_ne_bytes());
            expected_h1.extend([0; 2]);
        }
        assert_eq!(h1_blocks.as_bytes(), expected_h1);

        let mut hp1_values = vec![1.0; QK8_H];
        hp1_values.extend(vec![8.0; QK8_H]);
        let mut hp1_blocks = vec![BlockQ8HP1::new_zeroed(); 2];
        k_quants::quantize_row_q8_hp1_ref(&hp1_values, &mut hp1_blocks)?;
        let mut expected_hp1 = Vec::new();
        for m in [0i16, 3i16] {
            expected_hp1.extend([64; QK8_H]);
            expected_hp1.extend(m.to_ne_bytes());
            expected_hp1.extend([0; 2]);
            expected_hp1.extend(0.015625f32.to_ne_bytes());
        }
        assert_eq!(hp1_blocks.as_bytes(), expected_hp1);
        Ok(())
    }

    #[test]
    fn q8_h_qtensor_dequantization_remains_unsupported() -> Result<()> {
        for dtype in [GgmlDType::Q8H1, GgmlDType::Q8HP1] {
            let tensor = Tensor::from_vec(vec![1.0; QK8_H], (QK8_H,), &Device::Cpu)?;
            let quantized = QTensor::quantize(&tensor, dtype)?;
            assert!(quantized.dequantize(&Device::Cpu).is_err());
        }
        Ok(())
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(T::BLCK_SIZE) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                T::BLCK_SIZE
            )
        }
        let row_blocks = hidden / T::BLCK_SIZE;
        if self.len() != rows * row_blocks {
            crate::bail!(
                "quantized tensor has {} blocks, expected {}",
                self.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                crate::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &self[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            T::to_float(src, dst);
        }
        Ok(CpuStorage::F32(out))
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

fn quantize_h_cpu(values: &[f32], shape: &Shape, dtype: GgmlDType) -> Result<QStorage> {
    let Some(&n_per_row) = shape.dims().last() else {
        crate::bail!("{dtype:?} cannot quantize a scalar tensor")
    };
    if n_per_row == 0 {
        crate::bail!("{dtype:?} row width cannot be zero")
    }
    if !n_per_row.is_multiple_of(QK8_H) {
        crate::bail!("{dtype:?} row width {n_per_row} is not divisible by block size {QK8_H}")
    }
    if values.len() != shape.elem_count() || !values.len().is_multiple_of(n_per_row) {
        crate::bail!(
            "{dtype:?} flat value count {} is incompatible with shape {shape:?}",
            values.len()
        )
    }

    let blocks_per_row = n_per_row / QK8_H;
    let block_count = values.len() / QK8_H;
    match dtype {
        GgmlDType::Q8H1 => {
            let mut blocks = vec![BlockQ8H1::new_zeroed(); block_count];
            for (row, row_blocks) in values
                .chunks_exact(n_per_row)
                .zip(blocks.chunks_exact_mut(blocks_per_row))
            {
                k_quants::quantize_row_q8_h1_ref(row, row_blocks)?;
            }
            Ok(raw_storage_from_blocks(blocks, dtype, QK8_H))
        }
        GgmlDType::Q8HP1 => {
            let mut blocks = vec![BlockQ8HP1::new_zeroed(); block_count];
            for (row, row_blocks) in values
                .chunks_exact(n_per_row)
                .zip(blocks.chunks_exact_mut(blocks_per_row))
            {
                k_quants::quantize_row_q8_hp1_ref(row, row_blocks)?;
            }
            Ok(raw_storage_from_blocks(blocks, dtype, QK8_H))
        }
        _ => crate::bail!("internal misuse of H-family quantizer with {dtype:?}"),
    }
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self {
            storage,
            shape,
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        if matches!(dtype, GgmlDType::Q8H1 | GgmlDType::Q8HP1) {
            if !src.device().is_cpu() {
                crate::bail!("{dtype:?} quantization is currently CPU-only")
            }
            let shape = shape.clone();
            let values = src
                .to_dtype(crate::DType::F32)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let storage = quantize_h_cpu(&values, &shape, dtype)?;
            return Self::new(storage, shape);
        }
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, hidden) = self.shape.dims2()?;
        if !hidden.is_multiple_of(self.dtype().block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype().block_size()
            )
        }
        let mut out_shape = ids.dims().to_vec();
        out_shape.push(hidden);
        let device = self.device();
        let ids = ids
            .to_device(&device)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .contiguous()?;
        let storage = match &self.storage {
            QStorage::Cpu(storage) => {
                let ids = ids.to_vec1::<u32>()?;
                Storage::Cpu(storage.embedding(&ids, rows, hidden)?)
            }
            QStorage::CpuRaw(storage) => {
                crate::bail!("embedding is not implemented for {:?}", storage.dtype())
            }
            QStorage::Metal(storage) => match &*ids.storage() {
                Storage::Metal(ids_storage) => {
                    Storage::Metal(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
            QStorage::Cuda(storage) => match &*ids.storage() {
                Storage::Cuda(ids_storage) => {
                    Storage::Cuda(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
        };
        let none = crate::op::BackpropOp::none();
        Ok(crate::tensor::from_storage(storage, out_shape, none, false))
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => {
                    panic!("Non-cuda indexed_moe_forward is not implemented!");
                }
            },
            _ => {
                panic!("indexed_moe_forward is not implemented in this platform!");
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::CpuRaw(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.storage.device_ptr_with_guard(stream)
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            _ => {
                panic!("Not implemented!")
            }
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.embedding(ids),
            Self::Tensor(w) | Self::TensorF16(w) => {
                let mut final_dims = ids.dims().to_vec();
                final_dims.push(w.dim(D::Minus1)?);
                let ids = ids.to_device(w.device())?.flatten_all()?;
                w.index_select(&ids, 0)?.reshape(final_dims)
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::CpuRaw(storage) => {
                crate::bail!("matmul is not implemented for {:?}", storage.dtype())
            }
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];

                // Try the 8-column BlockQ4Kx8 repacked path.
                #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
                if self_storage.dtype() == GgmlDType::Q4K && n.is_multiple_of(8) {
                    use zerocopy::{FromBytes, IntoBytes};

                    let total_blocks =
                        self_storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ4K>();
                    let repacked = self.repacked_qs.get_or_init(|| {
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                self_storage.as_ptr() as *const BlockQ4K,
                                total_blocks,
                            )
                        };
                        let packed = k_quants::pack_to_q4kx8(blocks, n);
                        Some(packed.as_bytes().to_vec())
                    });
                    if let Some(repacked_bytes) = repacked {
                        let block_x8: &[BlockQ4Kx8] =
                            <[BlockQ4Kx8]>::ref_from_bytes(repacked_bytes).map_err(|_| {
                                crate::Error::Msg(
                                    "repacked_qs alignment invariant violated".to_string(),
                                )
                            })?;

                        k_quants::matmul_q4k_x8(
                            (dst_shape.elem_count() / n, k, n),
                            slice,
                            block_x8,
                            &mut dst_storage,
                        )?;
                        return Ok((crate::CpuStorage::F32(dst_storage), dst_shape));
                    }
                }

                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
