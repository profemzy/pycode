"""
Utility functions for input validation, security, and error handling.
"""
import re
import html
import logging
from typing import Optional, Union, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Custom exception for input validation errors."""
    pass


def sanitize_user_input(user_input: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent injection attacks and excessive content.
    
    Args:
        user_input: Raw user input string
        max_length: Maximum allowed length for input
        
    Returns:
        Sanitized input string
        
    Raises:
        InputValidationError: If input is invalid or exceeds limits
    """
    if not isinstance(user_input, str):
        raise InputValidationError("Input must be a string")
    
    # Check length limits
    if len(user_input) > max_length:
        raise InputValidationError(
            f"Input too long: {len(user_input)} chars, max {max_length}"
        )
    
    # Remove null bytes and control characters (except common whitespace)
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', user_input)
    
    # HTML escape to prevent XSS (even though we're not serving web content)
    sanitized = html.escape(sanitized)
    
    # Strip excessive whitespace
    sanitized = sanitized.strip()
    
    if not sanitized:
        raise InputValidationError("Input cannot be empty after sanitization")
    
    return sanitized


def validate_search_query(query: str) -> str:
    """
    Validate and sanitize search query input.
    
    Args:
        query: Search query string
        
    Returns:
        Validated and sanitized query
        
    Raises:
        InputValidationError: If query is invalid
    """
    # Basic sanitization
    sanitized_query = sanitize_user_input(query, max_length=500)
    
    # Check for minimum meaningful length
    if len(sanitized_query.replace(' ', '')) < 2:
        raise InputValidationError("Search query too short")
    
    # Remove excessive repeated characters (possible spam/abuse)
    sanitized_query = re.sub(r'(.)\1{10,}', r'\1\1\1', sanitized_query)
    
    # Log potentially suspicious queries (without logging the actual content)
    if len(sanitized_query) > 200 or sanitized_query.count(' ') > 50:
        logger.warning("Received unusually long or complex search query")
    
    return sanitized_query


def validate_environment_variable(
    var_name: str, 
    value: Optional[str], 
    required: bool = True,
    validator_func: Optional[callable] = None
) -> Optional[str]:
    """
    Validate environment variable values.
    
    Args:
        var_name: Name of the environment variable
        value: Value to validate
        required: Whether the variable is required
        validator_func: Optional custom validation function
        
    Returns:
        Validated value or None if not required and missing
        
    Raises:
        InputValidationError: If validation fails
    """
    if value is None or value.strip() == "":
        if required:
            raise InputValidationError(
                f"Required environment variable {var_name} is missing or empty"
            )
        return None
    
    # Basic security checks
    if len(value) > 10000:  # Reasonable limit for env vars
        raise InputValidationError(
            f"Environment variable {var_name} is too long"
        )
    
    # Check for null bytes
    if '\x00' in value:
        raise InputValidationError(
            f"Environment variable {var_name} contains null bytes"
        )
    
    # Apply custom validation if provided
    if validator_func:
        try:
            return validator_func(value)
        except Exception as e:
            raise InputValidationError(
                f"Validation failed for {var_name}: {e}"
            )
    
    return value.strip()


def validate_api_key(api_key: str) -> str:
    """
    Validate API key format and basic security checks.
    
    Args:
        api_key: API key to validate
        
    Returns:
        Validated API key
        
    Raises:
        InputValidationError: If API key is invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise InputValidationError("API key must be a non-empty string")
    
    api_key = api_key.strip()
    
    # Check minimum length (most API keys are at least 20 chars)
    if len(api_key) < 10:
        raise InputValidationError("API key too short")
    
    # Check maximum reasonable length
    if len(api_key) > 500:
        raise InputValidationError("API key too long")
    
    # Check for suspicious patterns
    if api_key.count(' ') > 5:  # API keys shouldn't have many spaces
        raise InputValidationError("API key format appears invalid")
    
    # Check for null bytes or control characters
    if re.search(r'[\x00-\x1F\x7F]', api_key):
        raise InputValidationError("API key contains invalid characters")
    
    return api_key


def validate_url(url: str) -> str:
    """
    Validate URL format and basic security checks.
    
    Args:
        url: URL to validate
        
    Returns:
        Validated URL
        
    Raises:
        InputValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise InputValidationError("URL must be a non-empty string")
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
        
        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            raise InputValidationError("URL must include scheme and domain")
        
        # Only allow http/https
        if parsed.scheme not in ['http', 'https']:
            raise InputValidationError("URL must use http or https scheme")
        
        # Basic security checks
        if any(char in url for char in ['\x00', '\r', '\n']):
            raise InputValidationError("URL contains invalid characters")
        
        return url
        
    except Exception as e:
        raise InputValidationError(f"Invalid URL format: {e}")


def safe_str_conversion(value: Any, max_length: int = 1000) -> str:
    """
    Safely convert any value to string with length limits.
    
    Args:
        value: Value to convert
        max_length: Maximum allowed string length
        
    Returns:
        String representation of the value
    """
    try:
        if value is None:
            return ""
        
        str_value = str(value)
        
        if len(str_value) > max_length:
            return str_value[:max_length] + "... (truncated)"
        
        return str_value
        
    except Exception as e:
        logger.warning(f"Failed to convert value to string: {e}")
        return f"<conversion_error: {type(value).__name__}>"


def validate_model_name(model_name: str) -> str:
    """
    Validate model name format.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        Validated model name
        
    Raises:
        InputValidationError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise InputValidationError("Model name must be a non-empty string")
    
    model_name = model_name.strip()
    
    # Check length
    if len(model_name) < 1 or len(model_name) > 100:
        raise InputValidationError(
            "Model name length must be between 1 and 100 characters"
        )
    
    # Check for valid characters (alphanumeric, hyphens, underscores, etc.)
    if not re.match(r'^[a-zA-Z0-9\-_/.]+$', model_name):
        raise InputValidationError("Model name contains invalid characters")
    
    return model_name


def validate_temperature(temperature: Union[str, float, int]) -> float:
    """
    Validate temperature value.
    
    Args:
        temperature: Temperature value to validate
        
    Returns:
        Validated temperature as float
        
    Raises:
        InputValidationError: If temperature is invalid
    """
    try:
        temp_float = float(temperature)
        
        if temp_float < 0.0 or temp_float > 2.0:
            raise InputValidationError(
                "Temperature must be between 0.0 and 2.0"
            )
        
        return temp_float
        
    except (ValueError, TypeError) as e:
        raise InputValidationError(f"Invalid temperature value: {e}")


# Rate limiting utilities
class SimpleRateLimiter:
    """Simple in-memory rate limiter for basic protection."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for the given identifier.
        
        Args:
            identifier: Unique identifier (e.g., user ID, IP address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        import time
        
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            k: v for k, v in self.requests.items() 
            if current_time - v['first_request'] < self.window_seconds
        }
        
        if identifier not in self.requests:
            self.requests[identifier] = {
                'count': 1,
                'first_request': current_time
            }
            return True
        
        request_data = self.requests[identifier]
        
        if current_time - request_data['first_request'] >= self.window_seconds:
            # Reset window
            self.requests[identifier] = {
                'count': 1,
                'first_request': current_time
            }
            return True
        
        if request_data['count'] >= self.max_requests:
            return False
        
        request_data['count'] += 1
        return True