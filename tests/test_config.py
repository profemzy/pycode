"""Tests for configuration management."""
import os
import pytest
from unittest.mock import patch
from llm_factory.config import (
    LLMConfig,
    EmbeddingConfig,
    validate_and_adjust_temperature,
    set_temperature_constraints,
    validate_environment
)


class TestTemperatureConstraints:
    """Test temperature constraint functionality."""
    
    def test_validate_and_adjust_temperature_within_range(self):
        """Test temperature validation within acceptable range."""
        result = validate_and_adjust_temperature("azure/model-router", 0.7)
        assert result == 0.7
    
    def test_validate_and_adjust_temperature_too_low(self):
        """Test temperature adjustment when too low."""
        with pytest.warns(UserWarning):
            result = validate_and_adjust_temperature("azure/gpt-4-mini", 0.5)
            assert result == 1.0  # Should be adjusted to minimum
    
    def test_validate_and_adjust_temperature_too_high(self):
        """Test temperature adjustment when too high."""
        with pytest.warns(UserWarning):
            result = validate_and_adjust_temperature("azure/model-router", 2.5)
            assert result == 2.0  # Should be adjusted to maximum
    
    def test_set_temperature_constraints(self):
        """Test setting custom temperature constraints."""
        # Save original constraints
        from llm_factory.config import MODEL_TEMPERATURE_CONSTRAINTS
        original = MODEL_TEMPERATURE_CONSTRAINTS.copy()
        
        try:
            set_temperature_constraints("custom-model", 0.5, 1.5)
            result = validate_and_adjust_temperature("custom-model", 0.3)
            assert result == 0.5  # Should be adjusted to custom minimum
        finally:
            # Restore original constraints
            MODEL_TEMPERATURE_CONSTRAINTS.clear()
            MODEL_TEMPERATURE_CONSTRAINTS.update(original)


class TestLLMConfig:
    """Test LLM configuration management."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'OPENAI_MODEL': 'azure/gpt-4-turbo',
        'OPENAI_TEMPERATURE': '0.8',
        'OPENAI_TIMEOUT': '60',
        'OPENAI_MAX_TOKENS': '2000'
    })
    def test_llm_config_from_env_valid(self):
        """Test LLM config creation from valid environment variables."""
        config = LLMConfig.from_env()
        
        assert config.api_key == 'sk-test123456789012345'
        assert config.model == 'azure/gpt-4-turbo'
        assert config.timeout == 60
        assert config.max_tokens == 2000
    
    @patch.dict(os.environ, {}, clear=True)
    def test_llm_config_from_env_missing_api_key(self):
        """Test LLM config creation with missing API key."""
        with pytest.raises(ValueError,
                           match="Configuration validation failed"):
            LLMConfig.from_env()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'OPENAI_TIMEOUT': 'invalid'
    })
    def test_llm_config_from_env_invalid_timeout(self):
        """Test LLM config creation with invalid timeout."""
        with pytest.raises(ValueError,
                           match="Configuration validation failed"):
            LLMConfig.from_env()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'OPENAI_TIMEOUT': '500'  # Too high
    })
    def test_llm_config_from_env_timeout_too_high(self):
        """Test LLM config creation with timeout too high."""
        with pytest.raises(ValueError,
                           match="Configuration validation failed"):
            LLMConfig.from_env()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'OPENAI_MAX_TOKENS': '200000'  # Too high
    })
    def test_llm_config_from_env_max_tokens_too_high(self):
        """Test LLM config creation with max_tokens too high."""
        with pytest.raises(ValueError,
                           match="Configuration validation failed"):
            LLMConfig.from_env()


class TestEmbeddingConfig:
    """Test embedding configuration management."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'EMBEDDING_MODEL': 'text-embedding-ada-002',
        'EMBEDDING_DIMENSIONS': '1536'
    })
    def test_embedding_config_from_env_valid(self):
        """Test embedding config creation from valid environment variables."""
        config = EmbeddingConfig.from_env()
        
        assert config.api_key == 'sk-test123456789012345'
        assert config.model == 'text-embedding-ada-002'
        assert config.dimensions == 1536
    
    @patch.dict(os.environ, {
        'EMBEDDING_API_KEY': 'sk-embedding123456789012345',
        'EMBEDDING_MODEL': 'text-embedding-3-small'
    }, clear=True)
    def test_embedding_config_from_env_embedding_specific(self):
        """Test embedding config with embedding-specific env vars."""
        config = EmbeddingConfig.from_env()
        
        assert config.api_key == 'sk-embedding123456789012345'
        assert config.model == 'text-embedding-3-small'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_config_from_env_missing_api_key(self):
        """Test embedding config creation with missing API key."""
        with pytest.raises(ValueError,
                           match="Embedding configuration validation failed"):
            EmbeddingConfig.from_env()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'EMBEDDING_DIMENSIONS': '15000'  # Too high
    })
    def test_embedding_config_from_env_dimensions_too_high(self):
        """Test embedding config with dimensions too high."""
        with pytest.raises(ValueError,
                           match="Embedding configuration validation failed"):
            EmbeddingConfig.from_env()


class TestEnvironmentValidation:
    """Test environment validation functionality."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test123456789012345',
        'LOG_LEVEL': 'INFO'
    })
    def test_validate_environment_success(self):
        """Test successful environment validation."""
        # Should not raise any exceptions
        validate_environment()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_missing_required(self):
        """Test environment validation with missing required variables."""
        with pytest.raises(ValueError, match="Environment validation failed"):
            validate_environment()
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-proj-test123456789',  # Project key
    })
    @patch('builtins.print')
    def test_validate_environment_project_key_warning(self, mock_print):
        """Test warning for project API key without base URL."""
        validate_environment()
        # Should print warning
        mock_print.assert_called()
        warning_text = str(mock_print.call_args[0][0])
        assert "project API key" in warning_text
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-test-valid-key-1234567890',  # Valid API key
        'LOG_LEVEL': 'INVALID'  # Invalid log level
    }, clear=True)
    def test_validate_environment_warnings(self):
        """Test environment validation with warning conditions."""
        with pytest.warns(UserWarning):
            validate_environment()


class TestConfigDataclasses:
    """Test configuration dataclass functionality."""
    
    def test_llm_config_post_init_temperature_adjustment(self):
        """Test temperature adjustment in LLM config post_init."""
        with pytest.warns(UserWarning):
            config = LLMConfig(
                api_key="sk-test123456789012345",
                model="azure/gpt-4-mini",
                temperature=0.5  # Should be adjusted to 1.0
            )
            assert config.temperature == 1.0
    
    def test_llm_config_defaults(self):
        """Test LLM config defaults."""
        config = LLMConfig(api_key="sk-test123456789012345")
        
        assert config.model == "azure/model-router"
        assert config.temperature == 0.7
        assert config.base_url is None
        assert config.max_tokens is None
        assert config.timeout == 30
    
    def test_embedding_config_defaults(self):
        """Test embedding config defaults."""
        config = EmbeddingConfig(api_key="sk-test123456789012345")
        
        assert config.model == "azure/text-embedding-ada-002"
        assert config.base_url is None
        assert config.dimensions is None


if __name__ == "__main__":
    pytest.main([__file__])