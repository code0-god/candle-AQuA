use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, DeviceLocation, Error, Layout, Result, Shape, WithDType};
use std::{fmt, sync::Arc};

/// Result of asking the injected AQuA executor to handle an operation.
///
/// `Executed` means that the external executor produced the result.
/// `Fallback` asks `AquaStorage` to execute the ordinary Candle CPU path.
#[derive(Debug)]
pub enum AquaDispatch<T> {
    Executed(T),
    Fallback,
}

/// Candle matmul request delivered to an injected AQuA executor.
///
/// Both operands are currently represented by CPU-shadow storage. Layouts
/// retain Candle's logical shape, offset, and stride semantics.
#[derive(Debug)]
pub struct AquaMatMulRequest<'a> {
    pub lhs: &'a CpuStorage,
    pub rhs: &'a CpuStorage,
    pub bmnk: (usize, usize, usize, usize),
    pub lhs_layout: &'a Layout,
    pub rhs_layout: &'a Layout,
}

/// Execution interface defined by Candle and implemented by AQuA.
///
/// This trait intentionally contains no ExSIA, RaCo, BSV, or AQuA repository
/// types. Candle only provides storage and operation metadata.
pub trait AquaExecutor: Send + Sync + fmt::Debug {
    fn name(&self) -> &'static str;

    fn matmul(&self, _request: AquaMatMulRequest<'_>) -> Result<AquaDispatch<CpuStorage>> {
        Ok(AquaDispatch::Fallback)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Default executor used while validating Device::Aqua plumbing.
///
/// Every operation falls back to Candle's CPU implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuFallbackAquaExecutor;

impl AquaExecutor for CpuFallbackAquaExecutor {
    fn name(&self) -> &'static str {
        "cpu-fallback"
    }
}

/// Candle device handle for AQuA.
///
/// The executor identity is part of the device identity. Two separately
/// constructed devices with the same numeric id are not considered identical
/// unless they share the same executor Arc.
#[derive(Clone)]
pub struct AquaDevice {
    device_id: usize,
    executor: Arc<dyn AquaExecutor>,
}

impl AquaDevice {
    pub fn cpu_fallback(device_id: usize) -> Self {
        Self {
            device_id,
            executor: Arc::new(CpuFallbackAquaExecutor),
        }
    }

    pub fn with_executor(device_id: usize, executor: Arc<dyn AquaExecutor>) -> Self {
        Self {
            device_id,
            executor,
        }
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn executor(&self) -> &Arc<dyn AquaExecutor> {
        &self.executor
    }

    pub fn same_instance(&self, other: &Self) -> bool {
        self.device_id == other.device_id && Arc::ptr_eq(&self.executor, &other.executor)
    }

    pub fn wrap_cpu_storage(&self, cpu: CpuStorage) -> AquaStorage {
        AquaStorage::from_cpu(self.clone(), cpu)
    }
}

impl fmt::Debug for AquaDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AquaDevice")
            .field("device_id", &self.device_id)
            .field("executor", &self.executor.name())
            .finish()
    }
}

/// Initial Candle storage for AQuA.
///
/// This phase deliberately uses CPU-shadow storage. It allows the complete
/// Candle model/runtime path to run on Device::Aqua before accelerator-resident
/// memory is introduced.
#[derive(Clone, Debug)]
pub struct AquaStorage {
    device: AquaDevice,
    cpu: CpuStorage,
}

impl AquaStorage {
    pub fn from_cpu(device: AquaDevice, cpu: CpuStorage) -> Self {
        Self { device, cpu }
    }

    pub fn cpu_storage(&self) -> &CpuStorage {
        &self.cpu
    }

    pub(crate) fn cpu_storage_mut(&mut self) -> &mut CpuStorage {
        &mut self.cpu
    }

    pub(crate) fn replace_cpu_storage(&self, cpu: CpuStorage) -> Self {
        Self {
            device: self.device.clone(),
            cpu,
        }
    }

    fn ensure_same_device(&self, other: &Self, op: &'static str) -> Result<()> {
        if self.device.same_instance(&other.device) {
            return Ok(());
        }

        Err(Error::DeviceMismatchBinaryOp {
            lhs: self.device.location(),
            rhs: other.device.location(),
            op,
        }
        .bt())
    }

    fn validate_matmul_output(
        &self,
        output: &CpuStorage,
        bmnk: (usize, usize, usize, usize),
    ) -> Result<()> {
        if output.dtype() != self.dtype() {
            crate::bail!(
                "Aqua executor matmul returned dtype {:?}, expected {:?}",
                output.dtype(),
                self.dtype()
            )
        }
        let (b, m, n, _) = bmnk;
        let expected_len = b
            .checked_mul(m)
            .and_then(|len| len.checked_mul(n))
            .ok_or_else(|| Error::Msg("Aqua executor matmul output size overflow".to_owned()))?;
        let actual_len = match output {
            CpuStorage::U8(values) => values.len(),
            CpuStorage::U32(values) => values.len(),
            CpuStorage::I16(values) => values.len(),
            CpuStorage::I32(values) => values.len(),
            CpuStorage::I64(values) => values.len(),
            CpuStorage::BF16(values) => values.len(),
            CpuStorage::F16(values) => values.len(),
            CpuStorage::F32(values) => values.len(),
            CpuStorage::F64(values) => values.len(),
            CpuStorage::F8E4M3(values) => values.len(),
            CpuStorage::F6E2M3(values) => values.len(),
            CpuStorage::F6E3M2(values) => values.len(),
            CpuStorage::F4(values) => values.len(),
            CpuStorage::F8E8M0(values) => values.len(),
        };
        if actual_len != expected_len {
            crate::bail!(
                "Aqua executor matmul returned {actual_len} elements, expected {expected_len}"
            )
        }
        Ok(())
    }
}

impl BackendDevice for AquaDevice {
    type Storage = AquaStorage;

    fn new(device_id: usize) -> Result<Self> {
        Ok(Self::cpu_fallback(device_id))
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Aqua {
            device_id: self.device_id,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.same_instance(other)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let cpu = CpuDevice.zeros_impl(shape, dtype)?;
        Ok(self.wrap_cpu_storage(cpu))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // CPU-shadow bring-up favors initialized memory over allocation speed.
        // Returning zeros is stricter than BackendDevice's caller contract.
        let cpu = CpuDevice.zeros_impl(shape, dtype)?;
        Ok(self.wrap_cpu_storage(cpu))
    }

    fn storage_from_slice<T: WithDType>(&self, values: &[T]) -> Result<Self::Storage> {
        let cpu = CpuDevice.storage_from_slice(values)?;
        Ok(self.wrap_cpu_storage(cpu))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(self.wrap_cpu_storage(storage.clone()))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        Ok(self.wrap_cpu_storage(storage))
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
        let cpu = CpuDevice.rand_uniform(shape, dtype, min, max)?;
        Ok(self.wrap_cpu_storage(cpu))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let cpu = CpuDevice.rand_normal(shape, dtype, mean, std)?;
        Ok(self.wrap_cpu_storage(cpu))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        CpuDevice.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        CpuDevice.get_current_seed()
    }

    fn synchronize(&self) -> Result<()> {
        self.executor.synchronize()
    }
}

mod storage;
