use crate::quantized::{gguf_file::GgufTypeProfile, GgmlDType};
use crate::Shape;

/// GGUF tensor source delivered during model materialization.
#[derive(Debug)]
pub struct AquaGgufTensorRequest<'a> {
    name: &'a str,
    shape: &'a Shape,
    dtype: GgmlDType,
    raw_data: &'a [u8],
    profile: GgufTypeProfile,
}

impl<'a> AquaGgufTensorRequest<'a> {
    pub(crate) const fn new(
        name: &'a str,
        shape: &'a Shape,
        dtype: GgmlDType,
        raw_data: &'a [u8],
        profile: GgufTypeProfile,
    ) -> Self {
        Self {
            name,
            shape,
            dtype,
            raw_data,
            profile,
        }
    }

    pub const fn name(&self) -> &'a str {
        self.name
    }

    pub const fn shape(&self) -> &'a Shape {
        self.shape
    }

    pub const fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub const fn raw_data(&self) -> &'a [u8] {
        self.raw_data
    }

    pub const fn profile(&self) -> GgufTypeProfile {
        self.profile
    }
}
