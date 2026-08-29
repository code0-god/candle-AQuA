use candle_core::quantized::{
    gguf_file::{Content, GgufTypeProfile, Value},
    GgmlDType, QTensor,
};
use candle_core::{Device, Result, Tensor};
use std::io::Cursor;

const PROFILE_KEY: &str = "aqua.gguf.profile";
const PROFILE_VERSION_KEY: &str = "aqua.gguf.profile_version";
const QUANTIZATION_VERSION_KEY: &str = "general.quantization_version";
const FILE_TYPE_KEY: &str = "general.file_type";

fn qtensor(dtype: GgmlDType) -> Result<QTensor> {
    let tensor = Tensor::from_vec(vec![1.0_f32; 32], (1, 32), &Device::Cpu)?;
    QTensor::quantize(&tensor, dtype)
}

fn gguf(metadata: &[(&str, Value)], tensors: &[(&str, GgmlDType)]) -> Result<Cursor<Vec<u8>>> {
    let tensors = tensors
        .iter()
        .map(|(name, dtype)| Ok((*name, qtensor(*dtype)?)))
        .collect::<Result<Vec<_>>>()?;
    let tensor_refs = tensors
        .iter()
        .map(|(name, tensor)| (*name, tensor))
        .collect::<Vec<_>>();
    let metadata_refs = metadata
        .iter()
        .map(|(key, value)| (*key, value))
        .collect::<Vec<_>>();
    let mut cursor = Cursor::new(Vec::new());
    candle_core::quantized::gguf_file::write(&mut cursor, &metadata_refs, &tensor_refs)?;
    cursor.set_position(0);
    Ok(cursor)
}

fn explicit_profile(name: &str, file_type: u32, version: u32) -> Vec<(&str, Value)> {
    vec![
        (PROFILE_KEY, Value::String(name.to_owned())),
        (PROFILE_VERSION_KEY, Value::U32(version)),
        (QUANTIZATION_VERSION_KEY, Value::U32(2)),
        (FILE_TYPE_KEY, Value::U32(file_type)),
    ]
}

fn legacy_profile(file_type: u32) -> Vec<(&'static str, Value)> {
    vec![
        (QUANTIZATION_VERSION_KEY, Value::U32(2)),
        (FILE_TYPE_KEY, Value::U32(file_type)),
    ]
}

fn read_error(metadata: &[(&str, Value)], tensors: &[(&str, GgmlDType)]) -> String {
    let mut cursor = gguf(metadata, tensors).expect("synthetic GGUF");
    Content::read(&mut cursor)
        .expect_err("profile must be rejected")
        .to_string()
}

#[test]
fn explicit_hp1_profile_decodes_41_as_q8_hp1() -> Result<()> {
    // Given
    let mut cursor = gguf(
        &explicit_profile("q8_hp1", 40, 1),
        &[("weight", GgmlDType::Q8HP1)],
    )?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::AquaQ8Hp1);
    assert_eq!(content.tensor_infos["weight"].ggml_dtype, GgmlDType::Q8HP1);
    Ok(())
}

#[test]
fn explicit_h1_profile_decodes_39_as_q8_h1() -> Result<()> {
    // Given
    let mut cursor = gguf(
        &explicit_profile("q8_h1", 38, 1),
        &[("weight", GgmlDType::Q8H1)],
    )?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::AquaQ8H1);
    assert_eq!(content.tensor_infos["weight"].ggml_dtype, GgmlDType::Q8H1);
    Ok(())
}

#[test]
fn legacy_hp1_fingerprint_is_accepted() -> Result<()> {
    // Given
    let mut cursor = gguf(&legacy_profile(40), &[("weight", GgmlDType::Q8HP1)])?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::AquaQ8Hp1);
    Ok(())
}

#[test]
fn legacy_h1_fingerprint_is_accepted() -> Result<()> {
    // Given
    let mut cursor = gguf(&legacy_profile(38), &[("weight", GgmlDType::Q8H1)])?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::AquaQ8H1);
    Ok(())
}

#[test]
fn file_type_40_without_dtype_41_is_not_hp1() -> Result<()> {
    // Given
    let mut cursor = gguf(&legacy_profile(40), &[("weight", GgmlDType::F32)])?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::Standard);
    Ok(())
}

#[test]
fn file_type_38_without_dtype_39_is_not_h1() -> Result<()> {
    // Given
    let mut cursor = gguf(&legacy_profile(38), &[("weight", GgmlDType::F32)])?;

    // When
    let content = Content::read(&mut cursor)?;

    // Then
    assert_eq!(content.profile, GgufTypeProfile::Standard);
    Ok(())
}

#[test]
fn standard_profile_does_not_interpret_41_as_hp1() {
    // Given
    let metadata = [];

    // When
    let error = read_error(&metadata, &[("weight", GgmlDType::Q8HP1)]);

    // Then
    assert!(error.contains("Standard profile"));
    assert!(error.contains("41"));
}

#[test]
fn standard_profile_does_not_interpret_39_as_h1() {
    // Given
    let metadata = [];

    // When
    let error = read_error(&metadata, &[("weight", GgmlDType::Q8H1)]);

    // Then
    assert!(error.contains("Standard profile"));
    assert!(error.contains("39"));
}

#[test]
fn rejects_mixed_h_types() {
    // Given
    let metadata = explicit_profile("q8_hp1", 40, 1);

    // When
    let error = read_error(
        &metadata,
        &[
            ("hp1.weight", GgmlDType::Q8HP1),
            ("h1.weight", GgmlDType::Q8H1),
        ],
    );

    // Then
    assert!(error.contains("mixed Q8_H"));
}

#[test]
fn rejects_profile_file_type_mismatch() {
    // Given
    let metadata = explicit_profile("q8_hp1", 38, 1);

    // When
    let error = read_error(&metadata, &[("weight", GgmlDType::Q8HP1)]);

    // Then
    assert!(error.contains("invalid AQuA GGUF profile metadata"));
}

#[test]
fn rejects_unknown_aqua_profile_version() {
    // Given
    let metadata = explicit_profile("q8_hp1", 40, 2);

    // When
    let error = read_error(&metadata, &[("weight", GgmlDType::Q8HP1)]);

    // Then
    assert!(error.contains("invalid AQuA GGUF profile metadata"));
}
