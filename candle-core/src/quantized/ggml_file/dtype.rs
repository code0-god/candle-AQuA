use super::GgmlDType;
use crate::Result;

pub(super) fn decode_ggml_dtype(raw: u32) -> Result<GgmlDType> {
    match raw {
        39 | 41 => crate::bail!(
            "Q8_H1/Q8_HP1 custom types are only supported in profile-identified GGUF files"
        ),
        _ => GgmlDType::from_u32(raw),
    }
}
