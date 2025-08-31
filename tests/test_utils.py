"""Tests for utility functions."""
import pytest
from utils import (
    sanitize_user_input,
    validate_search_query,
    validate_environment_variable,
    validate_api_key,
    validate_url,
    validate_model_name,
    validate_temperature,
    InputValidationError,
    SimpleRateLimiter,
    safe_str_conversion
)


class TestInputSanitization:
    """Test input sanitization functions."""
    
    def test_sanitize_user_input_normal(self):
        """Test normal input sanitization."""
        result = sanitize_user_input("Hello world!")
        assert result == "Hello world!"
    
    def test_sanitize_user_input_html_escape(self):
        """Test HTML escaping."""
        result = sanitize_user_input("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result
    
    def test_sanitize_user_input_too_long(self):
        """Test input length limits."""
        long_input = "a" * 10001
        with pytest.raises(InputValidationError):
            sanitize_user_input(long_input)
    
    def test_sanitize_user_input_empty(self):
        """Test empty input handling."""
        with pytest.raises(InputValidationError):
            sanitize_user_input("")
    
    def test_sanitize_user_input_whitespace_only(self):
        """Test whitespace-only input."""
        with pytest.raises(InputValidationError):
            sanitize_user_input("   \n\t  ")
    
    def test_sanitize_user_input_control_chars(self):
        """Test removal of control characters."""
        result = sanitize_user_input("Hello\x00\x01world")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Helloworld" in result


class TestSearchQueryValidation:
    """Test search query validation."""
    
    def test_validate_search_query_normal(self):
        """Test normal search query."""
        result = validate_search_query("python programming")
        assert result == "python programming"
    
    def test_validate_search_query_too_short(self):
        """Test too short query."""
        with pytest.raises(InputValidationError):
            validate_search_query("a")
    
    def test_validate_search_query_repeated_chars(self):
        """Test repeated character handling."""
        result = validate_search_query("helllllllllllllllo world")
        assert result.count("l") <= 5  # Should be reduced


class TestEnvironmentValidation:
    """Test environment variable validation."""
    
    def test_validate_environment_variable_required_present(self):
        """Test required variable present."""
        result = validate_environment_variable(
            "TEST_VAR", "test_value", required=True
        )
        assert result == "test_value"
    
    def test_validate_environment_variable_required_missing(self):
        """Test required variable missing."""
        with pytest.raises(InputValidationError):
            validate_environment_variable("TEST_VAR", None, required=True)
    
    def test_validate_environment_variable_optional_missing(self):
        """Test optional variable missing."""
        result = validate_environment_variable(
            "TEST_VAR", None, required=False
        )
        assert result is None
    
    def test_validate_environment_variable_too_long(self):
        """Test variable too long."""
        long_value = "a" * 10001
        with pytest.raises(InputValidationError):
            validate_environment_variable("TEST_VAR", long_value)
    
    def test_validate_environment_variable_null_bytes(self):
        """Test null bytes in variable."""
        with pytest.raises(InputValidationError):
            validate_environment_variable("TEST_VAR", "test\x00value")


class TestAPIKeyValidation:
    """Test API key validation."""
    
    def test_validate_api_key_valid(self):
        """Test valid API key."""
        api_key = "sk-1234567890abcdef"
        result = validate_api_key(api_key)
        assert result == api_key
    
    def test_validate_api_key_too_short(self):
        """Test API key too short."""
        with pytest.raises(InputValidationError):
            validate_api_key("short")
    
    def test_validate_api_key_too_long(self):
        """Test API key too long."""
        long_key = "a" * 501
        with pytest.raises(InputValidationError):
            validate_api_key(long_key)
    
    def test_validate_api_key_empty(self):
        """Test empty API key."""
        with pytest.raises(InputValidationError):
            validate_api_key("")
    
    def test_validate_api_key_many_spaces(self):
        """Test API key with many spaces."""
        with pytest.raises(InputValidationError):
            validate_api_key("sk-abc def ghi jkl mno pqr stu vwx yz")


class TestURLValidation:
    """Test URL validation."""
    
    def test_validate_url_https(self):
        """Test HTTPS URL."""
        url = "https://api.example.com"
        result = validate_url(url)
        assert result == url
    
    def test_validate_url_http(self):
        """Test HTTP URL."""
        url = "http://api.example.com"
        result = validate_url(url)
        assert result == url
    
    def test_validate_url_no_scheme(self):
        """Test URL without scheme."""
        with pytest.raises(InputValidationError):
            validate_url("api.example.com")
    
    def test_validate_url_invalid_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(InputValidationError):
            validate_url("ftp://api.example.com")
    
    def test_validate_url_invalid_chars(self):
        """Test URL with invalid characters."""
        with pytest.raises(InputValidationError):
            validate_url("https://api.example.com\x00")


class TestModelNameValidation:
    """Test model name validation."""
    
    def test_validate_model_name_valid(self):
        """Test valid model name."""
        model = "azure/gpt-4-turbo"
        result = validate_model_name(model)
        assert result == model
    
    def test_validate_model_name_empty(self):
        """Test empty model name."""
        with pytest.raises(InputValidationError):
            validate_model_name("")
    
    def test_validate_model_name_too_long(self):
        """Test model name too long."""
        long_name = "a" * 101
        with pytest.raises(InputValidationError):
            validate_model_name(long_name)
    
    def test_validate_model_name_invalid_chars(self):
        """Test model name with invalid characters."""
        with pytest.raises(InputValidationError):
            validate_model_name("azure/gpt-4@turbo")


class TestTemperatureValidation:
    """Test temperature validation."""
    
    def test_validate_temperature_valid_float(self):
        """Test valid temperature as float."""
        result = validate_temperature(0.7)
        assert result == 0.7
    
    def test_validate_temperature_valid_string(self):
        """Test valid temperature as string."""
        result = validate_temperature("0.7")
        assert result == 0.7
    
    def test_validate_temperature_too_low(self):
        """Test temperature too low."""
        with pytest.raises(InputValidationError):
            validate_temperature(-0.1)
    
    def test_validate_temperature_too_high(self):
        """Test temperature too high."""
        with pytest.raises(InputValidationError):
            validate_temperature(2.1)
    
    def test_validate_temperature_invalid_string(self):
        """Test invalid temperature string."""
        with pytest.raises(InputValidationError):
            validate_temperature("not_a_number")


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_allows_initial_requests(self):
        """Test rate limiter allows initial requests."""
        limiter = SimpleRateLimiter(max_requests=3, window_seconds=60)
        
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests."""
        limiter = SimpleRateLimiter(max_requests=2, window_seconds=60)
        
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
    
    def test_rate_limiter_different_users(self):
        """Test rate limiter handles different users separately."""
        limiter = SimpleRateLimiter(max_requests=1, window_seconds=60)
        
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True
        assert limiter.is_allowed("user1") is False
        assert limiter.is_allowed("user2") is False


class TestSafeStringConversion:
    """Test safe string conversion."""
    
    def test_safe_str_conversion_string(self):
        """Test string conversion."""
        result = safe_str_conversion("test")
        assert result == "test"
    
    def test_safe_str_conversion_none(self):
        """Test None conversion."""
        result = safe_str_conversion(None)
        assert result == ""
    
    def test_safe_str_conversion_number(self):
        """Test number conversion."""
        result = safe_str_conversion(42)
        assert result == "42"
    
    def test_safe_str_conversion_long_string(self):
        """Test long string truncation."""
        long_string = "a" * 1500
        result = safe_str_conversion(long_string, max_length=100)
        assert len(result) <= 120  # 100 + "... (truncated)"
        assert "truncated" in result


if __name__ == "__main__":
    pytest.main([__file__])