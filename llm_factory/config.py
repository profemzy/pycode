"""Configuration management for LLM Factory."""
import os
import warnings
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv, find_dotenv

try:
    from utils import (
        validate_environment_variable,
        validate_api_key,
        validate_url,
        validate_model_name,
        validate_temperature,
        InputValidationError
    )
except ImportError:
    # Fallback validation functions if utils module is not available
    def validate_environment_variable(var_name, value, required=True,
                                      validator_func=None):
        if required and not value:
            raise ValueError(
                f"Required environment variable {var_name} is missing"
            )
        return value
    
    def validate_api_key(api_key):
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid API key")
        return api_key
    
    def validate_url(url):
        if url and not (url.startswith('http://') or
                        url.startswith('https://')):
            raise ValueError("Invalid URL format")
        return url
    
    def validate_model_name(model_name):
        if not model_name:
            raise ValueError("Model name cannot be empty")
        return model_name
    
    def validate_temperature(temperature):
        temp = float(temperature)
        if temp < 0.0 or temp > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return temp
    
    class InputValidationError(ValueError):
        pass

# Load environment variables
load_dotenv(find_dotenv(), override=False)

# Model temperature constraints (configurable via environment)
MODEL_TEMPERATURE_CONSTRAINTS = {
    # Default model constraints - azure/model-router
    'azure/model-router': {
        'min': float(os.getenv('MODEL_TEMPERATURE_MIN', '0.0')),
        'max': float(os.getenv('MODEL_TEMPERATURE_MAX', '2.0'))
    }
}

# Model pattern-based constraints for common cases
MODEL_PATTERN_CONSTRAINTS = {
    # Models that require minimum temperature 1.0
    'azure/gpt-5': {'min': 1.0, 'max': 2.0},
    'azure/gpt-4-mini': {'min': 1.0, 'max': 2.0},
    'azure/gpt-4-nano': {'min': 1.0, 'max': 2.0},
    # Models with relaxed constraints
    'gpt-3.5': {'min': 0.0, 'max': 2.0},
    'claude': {'min': 0.0, 'max': 2.0},
}


def validate_and_adjust_temperature(model: str, temperature: float) -> float:
    """
    Validate temperature for a specific model and adjust if necessary.

    Args:
        model: The model name to validate temperature for
        temperature: The requested temperature value

    Returns:
        Adjusted temperature value that meets model requirements
    """
    # First check for exact model match in MODEL_TEMPERATURE_CONSTRAINTS
    constraints = MODEL_TEMPERATURE_CONSTRAINTS.get(model)
    
    # If no exact match, check for pattern-based constraints
    if not constraints:
        model_lower = model.lower()
        for pattern, constraint in MODEL_PATTERN_CONSTRAINTS.items():
            if pattern in model_lower:
                constraints = constraint
                break

    # Fall back to default constraints
    if not constraints:
        constraints = MODEL_TEMPERATURE_CONSTRAINTS['azure/model-router']

    # Validate and adjust temperature within constraints
    if temperature < constraints['min']:
        original_temp = temperature
        temperature = constraints['min']
        warnings.warn(
            f"Temperature {original_temp} below minimum {constraints['min']} "
            f"for model '{model}', adjusted to {temperature}"
        )
    elif temperature > constraints['max']:
        original_temp = temperature
        temperature = constraints['max']
        warnings.warn(
            f"Temperature {original_temp} above maximum {constraints['max']} "
            f"for model '{model}', adjusted to {temperature}"
        )

    return temperature


def set_temperature_constraints(model: str, min_temp: float,
                                max_temp: float) -> None:
    """Dynamically set temperature constraints for a specific model.

    Args:
        model: The model name to set constraints for
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature
    """
    MODEL_TEMPERATURE_CONSTRAINTS[model] = {
        'min': float(min_temp),
        'max': float(max_temp)
    }


@dataclass
class LLMConfig:
    """Configuration for ChatOpenAI instances."""
    api_key: str
    base_url: Optional[str] = None
    model: str = "azure/model-router"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30

    def __post_init__(self):
        """Validate and adjust temperature after initialization."""
        self.temperature = validate_and_adjust_temperature(
            self.model, self.temperature
        )

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables with validation."""
        try:
            # Validate API key
            api_key = validate_environment_variable(
                "OPENAI_API_KEY",
                os.getenv("OPENAI_API_KEY"),
                required=True,
                validator_func=validate_api_key
            )
            
            # Validate base URL if provided
            base_url = validate_environment_variable(
                "OPENAI_API_BASE",
                os.getenv("OPENAI_API_BASE"),
                required=False,
                validator_func=validate_url
            )
            
            # Validate model name
            model = validate_environment_variable(
                "OPENAI_MODEL",
                os.getenv("OPENAI_MODEL", "azure/model-router"),
                required=True,
                validator_func=validate_model_name
            )
            
            # Validate temperature
            raw_temperature = validate_environment_variable(
                "OPENAI_TEMPERATURE",
                os.getenv("OPENAI_TEMPERATURE", "0.7"),
                required=True,
                validator_func=validate_temperature
            )
            
            # Validate timeout
            timeout_str = os.getenv("OPENAI_TIMEOUT", "30")
            try:
                timeout = int(timeout_str)
                if timeout <= 0 or timeout > 300:  # Max 5 minutes
                    raise ValueError(
                        "Timeout must be between 1 and 300 seconds"
                    )
            except ValueError as e:
                raise InputValidationError(f"Invalid timeout value: {e}")
            
            # Validate max_tokens
            max_tokens_str = os.getenv("OPENAI_MAX_TOKENS", "0")
            try:
                max_tokens = int(max_tokens_str) or None
                if max_tokens and (max_tokens < 1 or max_tokens > 100000):
                    raise ValueError("Max tokens must be between 1 and 100000")
            except ValueError as e:
                raise InputValidationError(f"Invalid max_tokens value: {e}")

            return cls(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=raw_temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
        except (InputValidationError, ValueError) as e:
            raise ValueError(f"Configuration validation failed: {e}")


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAIEmbeddings instances."""
    api_key: str
    base_url: Optional[str] = None
    model: str = "azure/text-embedding-ada-002"
    dimensions: Optional[int] = None

    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create config from environment variables with validation."""
        try:
            # Support both OPENAI_ and EMBEDDING_ prefixed env vars
            api_key = (os.getenv("OPENAI_API_KEY") or
                       os.getenv("EMBEDDING_API_KEY"))
            
            api_key = validate_environment_variable(
                "OPENAI_API_KEY or EMBEDDING_API_KEY",
                api_key,
                required=True,
                validator_func=validate_api_key
            )
            
            # Validate base URL if provided
            base_url = (os.getenv("OPENAI_API_BASE") or
                        os.getenv("EMBEDDING_BASE_URL"))
            
            if base_url:
                base_url = validate_environment_variable(
                    "OPENAI_API_BASE or EMBEDDING_BASE_URL",
                    base_url,
                    required=False,
                    validator_func=validate_url
                )
            
            # Validate model name
            model = validate_environment_variable(
                "EMBEDDING_MODEL",
                os.getenv("EMBEDDING_MODEL", "azure/text-embedding-ada-002"),
                required=True,
                validator_func=validate_model_name
            )
            
            # Validate dimensions
            dimensions_str = os.getenv("EMBEDDING_DIMENSIONS", "0")
            try:
                dimensions = int(dimensions_str) or None
                if dimensions and (dimensions < 1 or dimensions > 10000):
                    raise ValueError("Dimensions must be between 1 and 10000")
            except ValueError as e:
                raise InputValidationError(f"Invalid dimensions value: {e}")

            return cls(
                api_key=api_key,
                base_url=base_url,
                model=model,
                dimensions=dimensions
            )
            
        except (InputValidationError, ValueError) as e:
            raise ValueError(f"Embedding configuration validation failed: {e}")


def get_llm_config(model: Optional[str] = None) -> LLMConfig:
    """Get LLM configuration, optionally overriding the model."""
    config = LLMConfig.from_env()
    if model:
        config.model = model
    return config


def get_embedding_config(model: Optional[str] = None) -> EmbeddingConfig:
    """Get embedding configuration, optionally overriding the model."""
    config = EmbeddingConfig.from_env()
    if model:
        config.model = model
    return config


def validate_environment() -> None:
    """Validate that all required environment variables are set and secure."""
    try:
        # Test creating configurations to validate all settings
        LLMConfig.from_env()
        
        # Additional security checks
        api_key = os.getenv("OPENAI_API_KEY", "")
        has_base_url = os.getenv("OPENAI_API_BASE")
        
        # Warn about potential configuration issues
        if not has_base_url and api_key.startswith("sk-proj-"):
            print("Warning: You appear to be using a project API key "
                  "but no custom base_url is set.")
        
        # Check for common insecure patterns
        if api_key and len(api_key) < 20:
            warnings.warn("API key appears to be too short, please verify")
        
        # Check log level configuration
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            warnings.warn(f"Invalid LOG_LEVEL '{log_level}', using INFO")
            
    except Exception as e:
        raise ValueError(f"Environment validation failed: {e}")
