use super::decode_ggml_dtype;

#[test]
fn ggml_legacy_41_is_not_hp1() {
    // Given
    let raw_dtype = 41;

    // When
    let error = decode_ggml_dtype(raw_dtype).expect_err("legacy GGML must reject Q8_HP1");

    // Then
    assert_eq!(
        error.to_string(),
        "Q8_H1/Q8_HP1 custom types are only supported in profile-identified GGUF files"
    );
}

#[test]
fn ggml_legacy_39_is_not_h1() {
    // Given
    let raw_dtype = 39;

    // When
    let error = decode_ggml_dtype(raw_dtype).expect_err("legacy GGML must reject Q8_H1");

    // Then
    assert_eq!(
        error.to_string(),
        "Q8_H1/Q8_HP1 custom types are only supported in profile-identified GGUF files"
    );
}
