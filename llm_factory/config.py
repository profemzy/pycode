"""Configuration management for LLM Factory."""
import os
import warnings
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv, find_dotenv

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
    # Check if this is our default model first
    if model == 'azure/model-router':
        constraints = MODEL_TEMPERATURE_CONSTRAINTS.get(model)

    # Then check for configured model patterns
    else:
        constraints = None
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
        """Create config from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        raw_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        return cls(
            api_key=api_key,
            base_url=os.getenv("OPENAI_API_BASE"),
            model=os.getenv("OPENAI_MODEL", "azure/model-router"),
            temperature=raw_temperature,  # Let __post_init__ validate
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "0")) or None,
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30"))
        )


@dataclass
class EmbeddingConfig:
    """Configuration for OpenAIEmbeddings instances."""
    api_key: str
    base_url: Optional[str] = None
    model: str = "azure/text-embedding-ada-002"
    dimensions: Optional[int] = None

    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create config from environment variables."""
        # Support both OPENAI_ and EMBEDDING_ prefixed environment variables
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
        if not api_key:
            msg = ("OPENAI_API_KEY or EMBEDDING_API_KEY environment "
                   "variable is required")
            raise ValueError(msg)

        return cls(
            api_key=api_key,
            base_url=(os.getenv("OPENAI_API_BASE") or
                      os.getenv("EMBEDDING_BASE_URL")),
            model=os.getenv("EMBEDDING_MODEL", "azure/text-embedding-ada-002"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "0")) or None
        )


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
    """Validate that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        vars_str = ', '.join(missing)
        raise ValueError(f"Missing required environment variables: {vars_str}")

    # Optional warning for common configuration issues
    api_key = os.getenv("OPENAI_API_KEY", "")
    has_base_url = os.getenv("OPENAI_API_BASE")
    if not has_base_url and api_key.startswith("sk-proj-"):
        warning_msg = ("Warning: You appear to be using a project API key "
                       "but no custom base_url is set.")
        print(warning_msg)
