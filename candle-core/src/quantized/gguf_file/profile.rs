use super::Value;
use crate::quantized::GgmlDType;
use crate::{Result, Shape};
use std::collections::HashMap;

const PROFILE_KEY: &str = "aqua.gguf.profile";
const PROFILE_VERSION_KEY: &str = "aqua.gguf.profile_version";
const QUANTIZATION_VERSION_KEY: &str = "general.quantization_version";
const FILE_TYPE_KEY: &str = "general.file_type";
const PROFILE_VERSION: u32 = 1;
const QUANTIZATION_VERSION: u32 = 2;
const Q8_H1_DTYPE: u32 = 39;
const Q8_HP1_DTYPE: u32 = 41;
const Q8_H1_FILE_TYPE: u32 = 38;
const Q8_HP1_FILE_TYPE: u32 = 40;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufTypeProfile {
    Standard,
    AquaQ8H1,
    AquaQ8Hp1,
}

impl GgufTypeProfile {
    const fn other_h_dtype_id(self) -> Option<u32> {
        match self {
            Self::Standard => None,
            Self::AquaQ8H1 => Some(Q8_HP1_DTYPE),
            Self::AquaQ8Hp1 => Some(Q8_H1_DTYPE),
        }
    }

    const fn file_type(self) -> Option<u32> {
        match self {
            Self::Standard => None,
            Self::AquaQ8H1 => Some(Q8_H1_FILE_TYPE),
            Self::AquaQ8Hp1 => Some(Q8_HP1_FILE_TYPE),
        }
    }
}

#[derive(Debug)]
pub(super) struct RawTensorInfo {
    pub(super) shape: Shape,
    pub(super) offset: u64,
    pub(super) raw_dtype: u32,
}

pub(super) fn detect_type_profile(
    metadata: &HashMap<String, Value>,
    tensors: &HashMap<String, RawTensorInfo>,
) -> Result<GgufTypeProfile> {
    let profile = match (metadata.get(PROFILE_KEY), metadata.get(PROFILE_VERSION_KEY)) {
        (None, None) => detect_legacy_profile(metadata, tensors),
        (Some(Value::String(name)), Some(Value::U32(PROFILE_VERSION))) => {
            let profile = match name.as_str() {
                "q8_h1" => GgufTypeProfile::AquaQ8H1,
                "q8_hp1" => GgufTypeProfile::AquaQ8Hp1,
                _ => invalid_profile(format!("unknown profile '{name}'"))?,
            };
            validate_explicit_profile(metadata, profile)?;
            profile
        }
        _ => {
            return invalid_profile(
                "profile must be String and profile_version must be U32 value 1",
            )
        }
    };
    validate_h_dtype_set(profile, tensors)?;
    Ok(profile)
}

fn detect_legacy_profile(
    metadata: &HashMap<String, Value>,
    tensors: &HashMap<String, RawTensorInfo>,
) -> GgufTypeProfile {
    let quantization_version = metadata_u32(metadata, QUANTIZATION_VERSION_KEY);
    let file_type = metadata_u32(metadata, FILE_TYPE_KEY);
    let has_h1 = has_dtype(tensors, Q8_H1_DTYPE);
    let has_hp1 = has_dtype(tensors, Q8_HP1_DTYPE);

    match (quantization_version, file_type, has_h1, has_hp1) {
        (Some(QUANTIZATION_VERSION), Some(Q8_H1_FILE_TYPE), true, _) => GgufTypeProfile::AquaQ8H1,
        (Some(QUANTIZATION_VERSION), Some(Q8_HP1_FILE_TYPE), _, true) => GgufTypeProfile::AquaQ8Hp1,
        _ => GgufTypeProfile::Standard,
    }
}

fn validate_explicit_profile(
    metadata: &HashMap<String, Value>,
    profile: GgufTypeProfile,
) -> Result<()> {
    let expected_file_type = profile
        .file_type()
        .ok_or_else(|| crate::Error::Msg("missing AQuA profile file type".to_owned()))?;
    if metadata_u32(metadata, QUANTIZATION_VERSION_KEY) != Some(QUANTIZATION_VERSION) {
        return invalid_profile("general.quantization_version must be U32 value 2");
    }
    if metadata_u32(metadata, FILE_TYPE_KEY) != Some(expected_file_type) {
        return invalid_profile(format!(
            "general.file_type must be U32 value {expected_file_type}"
        ));
    }
    Ok(())
}

fn validate_h_dtype_set(
    profile: GgufTypeProfile,
    tensors: &HashMap<String, RawTensorInfo>,
) -> Result<()> {
    if let Some(other_dtype) = profile.other_h_dtype_id() {
        if has_dtype(tensors, other_dtype) {
            crate::bail!("mixed Q8_H tensor types are not supported by the {profile:?} profile")
        }
    }
    Ok(())
}

pub(super) fn decode_gguf_dtype(raw: u32, profile: GgufTypeProfile) -> Result<GgmlDType> {
    match (profile, raw) {
        (GgufTypeProfile::AquaQ8H1, Q8_H1_DTYPE) => Ok(GgmlDType::Q8H1),
        (GgufTypeProfile::AquaQ8Hp1, Q8_HP1_DTYPE) => Ok(GgmlDType::Q8HP1),
        (GgufTypeProfile::Standard, Q8_H1_DTYPE | Q8_HP1_DTYPE) => {
            let custom_name = if raw == Q8_H1_DTYPE {
                "Q8_H1"
            } else {
                "Q8_HP1"
            };
            crate::bail!(
                "GGUF dtype id {raw} is not supported by the Standard profile; \
                 it is only interpreted as {custom_name} in an AQuA Q8_H GGUF profile"
            )
        }
        _ => GgmlDType::from_u32(raw),
    }
}

fn metadata_u32(metadata: &HashMap<String, Value>, key: &str) -> Option<u32> {
    match metadata.get(key) {
        Some(Value::U32(value)) => Some(*value),
        _ => None,
    }
}

fn has_dtype(tensors: &HashMap<String, RawTensorInfo>, raw_dtype: u32) -> bool {
    tensors.values().any(|tensor| tensor.raw_dtype == raw_dtype)
}

fn invalid_profile<T>(reason: impl std::fmt::Display) -> Result<T> {
    crate::bail!("invalid AQuA GGUF profile metadata: {reason}")
}
