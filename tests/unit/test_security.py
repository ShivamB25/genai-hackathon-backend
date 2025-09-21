import pytest

from src.core.security import (
    EncryptionError,
    decrypt_secret,
    encrypt_secret,
    hash_api_key,
    sanitize_string,
    verify_api_key,
)


def test_encrypt_decrypt_secret():
    """Test that data can be encrypted and decrypted successfully."""
    original_data = "my_super_secret_data"
    key = "a_strong_key_for_testing"

    encrypted_data = encrypt_secret(original_data, key)
    assert encrypted_data != original_data

    decrypted_data = decrypt_secret(encrypted_data, key)
    assert decrypted_data == original_data


def test_decryption_with_wrong_key():
    """Test that decryption fails with an incorrect key."""
    original_data = "my_super_secret_data"
    correct_key = "a_strong_key_for_testing"
    wrong_key = "a_different_key"

    encrypted_data = encrypt_secret(original_data, correct_key)

    with pytest.raises(EncryptionError):
        decrypt_secret(encrypted_data, wrong_key)


def test_hash_and_verify_api_key():
    """Test that an API key can be hashed and verified."""
    api_key = "my-test-api-key"
    hashed_key = hash_api_key(api_key)

    assert hashed_key != api_key
    assert verify_api_key(api_key, hashed_key) is True


def test_verify_incorrect_api_key():
    """Test that an incorrect API key fails verification."""
    api_key = "my-test-api-key"
    incorrect_key = "wrong-api-key"
    hashed_key = hash_api_key(api_key)

    assert verify_api_key(incorrect_key, hashed_key) is False


def test_sanitize_string_removes_scripts():
    """Test that sanitize_string removes script tags."""
    dirty_string = '<script>alert("xss")</script>Hello'
    clean_string = sanitize_string(dirty_string)
    assert "<script>" not in clean_string
    assert "Hello" in clean_string


def test_sanitize_string_handles_null_bytes():
    """Test that sanitize_string removes null bytes."""
    dirty_string = "hello\0world"
    clean_string = sanitize_string(dirty_string)
    assert "\0" not in clean_string
    assert "helloworld" in clean_string
