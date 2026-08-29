#![cfg(feature = "aqua")]

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{
    backend::{BackendDevice, BackendStorage},
    AquaDevice, AquaDispatch, AquaExecutor, AquaMatMulRequest, CpuStorage, DType, Device,
    DeviceLocation, Result, Shape, Tensor,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Default)]
struct RecordingExecutor {
    matmuls: AtomicUsize,
}

impl AquaExecutor for RecordingExecutor {
    fn name(&self) -> &'static str {
        "recording"
    }

    fn matmul(&self, request: AquaMatMulRequest<'_>) -> Result<AquaDispatch<CpuStorage>> {
        self.matmuls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request.bmnk, (1, 2, 2, 2));
        Ok(AquaDispatch::Executed(CpuStorage::F32(vec![
            19.0, 22.0, 43.0, 50.0,
        ])))
    }
}

#[derive(Clone, Copy, Debug)]
enum MalformedOutputExecutor {
    DType,
    Length,
}

impl AquaExecutor for MalformedOutputExecutor {
    fn name(&self) -> &'static str {
        "malformed-output"
    }

    fn matmul(&self, _request: AquaMatMulRequest<'_>) -> Result<AquaDispatch<CpuStorage>> {
        let output = match self {
            Self::DType => CpuStorage::I64(vec![0; 4]),
            Self::Length => CpuStorage::F32(vec![0.0; 3]),
        };
        Ok(AquaDispatch::Executed(output))
    }
}

#[test]
fn creates_aqua_device() -> Result<()> {
    let device = Device::new_aqua(0)?;

    assert!(device.is_aqua());
    assert_eq!(device.location(), DeviceLocation::Aqua { device_id: 0 });
    Ok(())
}

#[test]
fn aqua_uninitialized_allocation_uses_initialized_cpu_shadow() -> Result<()> {
    let device = AquaDevice::cpu_fallback(0);
    let shape = Shape::from((2, 2));

    // SAFETY: AquaDevice's CPU-shadow implementation returns initialized zero
    // storage, which is stricter than BackendDevice's caller contract.
    let storage = unsafe { device.alloc_uninit(&shape, DType::F32)? };

    match storage.to_cpu_storage()? {
        CpuStorage::F32(values) => assert_eq!(values, vec![0.0; 4]),
        other => panic!("expected F32 CPU shadow, got {other:?}"),
    }
    Ok(())
}

#[test]
fn moves_tensor_between_cpu_and_aqua() -> Result<()> {
    let cpu = Device::Cpu;
    let aqua = Device::new_aqua(0)?;
    let original = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &cpu)?;

    let on_aqua = original.to_device(&aqua)?;
    assert!(on_aqua.device().is_aqua());
    assert_eq!(
        on_aqua.to_vec2::<f32>()?,
        vec![vec![1.0, 2.0], vec![3.0, 4.0]]
    );

    let restored = on_aqua.to_device(&cpu)?;
    assert!(restored.device().is_cpu());
    assert_eq!(restored.to_vec2::<f32>()?, original.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn cpu_fallback_matmul_matches_cpu_exactly() -> Result<()> {
    let cpu = Device::Cpu;
    let aqua = Device::new_aqua(0)?;
    let lhs_data = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rhs_data = [[5.0_f32, 6.0], [7.0, 8.0]];

    let cpu_result = Tensor::new(&lhs_data, &cpu)?.matmul(&Tensor::new(&rhs_data, &cpu)?)?;
    let aqua_result = Tensor::new(&lhs_data, &aqua)?.matmul(&Tensor::new(&rhs_data, &aqua)?)?;

    assert!(aqua_result.device().is_aqua());
    assert_eq!(aqua_result.to_vec2::<f32>()?, cpu_result.to_vec2::<f32>()?);
    Ok(())
}

#[test]
fn cloned_device_is_same_but_separate_device_is_not() -> Result<()> {
    let first = Device::new_aqua(0)?;
    let cloned = first.clone();
    let separate = Device::new_aqua(0)?;

    assert!(first.same_device(&cloned));
    assert!(!first.same_device(&separate));
    Ok(())
}

#[test]
fn cpu_shadow_tensor_ops_match_cpu() -> Result<()> {
    let aqua = Device::new_aqua(0)?;
    let lhs = Tensor::new(&[[1.0f32, -2.0], [3.0, 4.0]], &aqua)?;
    let rhs = Tensor::new(&[[2.0f32, 1.0], [0.5, 3.0]], &aqua)?;

    let result = (&lhs + &rhs)?.relu()?.matmul(&rhs.t()?)?.to_vec2::<f32>()?;
    assert_eq!(result, vec![vec![6.0, 1.5], vec![14.0, 22.75]]);
    assert!(lhs.device().is_aqua());
    assert_eq!(format!("{lhs:?}"), "Tensor[dims 2, 2; f32, aqua:0]");
    Ok(())
}

#[test]
fn injected_executor_dispatches_matmul_and_preserves_arc_identity() -> Result<()> {
    let concrete = Arc::new(RecordingExecutor::default());
    let executor: Arc<dyn AquaExecutor> = concrete.clone();
    let aqua = Device::new_aqua_with_executor(7, executor.clone())?;
    let aqua_device = aqua.as_aqua_device()?;
    assert!(Arc::ptr_eq(aqua_device.executor(), &executor));

    let lhs = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &aqua)?;
    let rhs = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &aqua)?;
    assert_eq!(
        lhs.matmul(&rhs)?.to_vec2::<f32>()?,
        vec![vec![19.0, 22.0], vec![43.0, 50.0],]
    );
    assert_eq!(concrete.matmuls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn rejects_malformed_executor_matmul_outputs() -> Result<()> {
    let cases = [
        (MalformedOutputExecutor::DType, "returned dtype"),
        (MalformedOutputExecutor::Length, "returned 3 elements"),
    ];

    for (executor, expected_message) in cases {
        let aqua = Device::new_aqua_with_executor(0, Arc::new(executor))?;
        let lhs = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &aqua)?;
        let rhs = Tensor::new(&[[5.0_f32, 6.0], [7.0, 8.0]], &aqua)?;
        let error = match lhs.matmul(&rhs) {
            Ok(_) => panic!("malformed executor output unexpectedly succeeded"),
            Err(error) => error,
        };

        assert!(error.to_string().contains(expected_message));
    }
    Ok(())
}

#[test]
fn executor_arc_identity_defines_device_identity() -> Result<()> {
    let shared: Arc<dyn AquaExecutor> = Arc::new(RecordingExecutor::default());
    let first = Device::new_aqua_with_executor(3, shared.clone())?;
    let same = Device::new_aqua_with_executor(3, shared)?;
    let distinct = Device::new_aqua_with_executor(3, Arc::new(RecordingExecutor::default()))?;

    assert!(first.same_device(&same));
    assert!(!first.same_device(&distinct));

    let lhs = Tensor::ones(1, DType::F32, &first)?;
    let rhs = Tensor::ones(1, DType::F32, &distinct)?;
    let error = match &lhs + &rhs {
        Ok(_) => panic!("different executor Arcs unexpectedly shared a device"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("device mismatch"));
    Ok(())
}

#[test]
fn quantized_storage_is_explicitly_unsupported() -> Result<()> {
    let aqua = Device::new_aqua(0)?;
    let tensor = Tensor::ones((1, 32), DType::F32, &aqua)?;
    let error = match QTensor::quantize(&tensor, GgmlDType::Q4_0) {
        Ok(_) => panic!("Aqua unexpectedly created quantized storage"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "quantized storage is not supported on Aqua devices"
    );
    Ok(())
}
