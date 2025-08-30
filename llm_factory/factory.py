"""LLM factory module for creating LLM and embedding instances."""
import logging
from typing import Optional, Any, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from .config import (
    get_llm_config,
    get_embedding_config,
    validate_environment
)

# Validate environment on import
try:
    validate_environment()
except ValueError as e:
    logging.warning(f"Configuration validation failed: {e}")


def get_llm(model: Optional[str] = None) -> ChatOpenAI:
    """
    Create and return an instance of ChatOpenAI with preset configuration.

    Args:
        model: The model to use for the LLM. If None, uses environment default.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    try:
        config = get_llm_config(model)
        return ChatOpenAI(
            model=config.model,
            api_key=SecretStr(config.api_key),
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
    except ValueError as e:
        logging.error(f"Failed to create ChatOpenAI instance: {e}")
        raise


def get_llm_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Create and return an OpenAI embeddings instance with preset configuration.

    Args:
        model: The model to use for embeddings. Uses environment
               default if None.
        api_key: Custom API key. If None, uses environment configuration.
        base_url: Custom base URL. If None, uses environment configuration.

    Returns:
        Configured OpenAIEmbeddings instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    try:
        if api_key or base_url:
            # Use custom configuration
            config = get_embedding_config(model)
            if api_key:
                config.api_key = api_key
            if base_url:
                config.base_url = base_url
        else:
            config = get_embedding_config(model)

        kwargs: Dict[str, Any] = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.base_url
        }
        if config.dimensions:
            kwargs["dimensions"] = config.dimensions

        return OpenAIEmbeddings(**kwargs)
    except ValueError as e:
        logging.error(f"Failed to create OpenAIEmbeddings instance: {e}")
        raise


async def get_llm_async(model: Optional[str] = None) -> ChatOpenAI:
    """
    Async version of get_llm.

    Create and return an instance of ChatOpenAI with preset configuration.

    Args:
        model: The model to use for the LLM. If None, uses environment default.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    try:
        config = get_llm_config(model)
        return ChatOpenAI(
            model=config.model,
            api_key=SecretStr(config.api_key),
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
    except ValueError as e:
        logging.error(f"Failed to create async ChatOpenAI instance: {e}")
        raise


async def get_llm_embeddings_async(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Async version of get_llm_embeddings.

    Create and return an OpenAI embeddings instance with preset configuration.

    Args:
        model: The model to use for embeddings. Uses environment
               default if None.
        api_key: Custom API key. If None, uses environment configuration.
        base_url: Custom base URL. If None, uses environment configuration.

    Returns:
        Configured OpenAIEmbeddings instance.

    Raises:
        ValueError: If required environment variables are not set.
    """
    try:
        if api_key or base_url:
            # Use custom configuration
            config = get_embedding_config(model)
            if api_key:
                config.api_key = api_key
            if base_url:
                config.base_url = base_url
        else:
            config = get_embedding_config(model)

        kwargs: Dict[str, Any] = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.base_url
        }
        if config.dimensions:
            kwargs["dimensions"] = config.dimensions

        return OpenAIEmbeddings(**kwargs)
    except ValueError as e:
        logging.error(f"Failed to create async OpenAIEmbeddings instance: {e}")
        raise
