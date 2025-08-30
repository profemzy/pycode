# LLM Factory

A robust, type-safe Python library for creating and managing Large Language Model (LLM) instances with LangChain and OpenAI integration. Features comprehensive configuration management, temperature constraints, and both synchronous and asynchronous support.

## 🌟 Features

- **Type-Safe Factory Pattern**: Strongly typed LLM and embedding instance creation
- **Flexible Configuration**: Environment-based configuration with intelligent defaults
- **Temperature Management**: Automatic temperature constraint validation and adjustment by model
- **Async Support**: Full async/await support for all LLM operations
- **Multiple LLM Providers**: Support for various OpenAI-compatible endpoints
- **Smart Model Detection**: Pattern-based temperature constraints for different model families
- **Production Ready**: Comprehensive error handling and logging

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- pipenv (recommended) or pip

### Install with pipenv (Recommended)
```bash
pipenv install
pipenv install --dev flake8
```

### Install with pip
```bash
pip install langchain==0.3.27
pip install langchain-openai==0.3.32
pip install python-dotenv==1.1.1
pip install flake8  # For development/linting
```

## 🚀 Quick Start

1. **Set up your environment variables:**
```bash
# Copy the example file and edit it
cp .env.example .env
# Edit .env with your actual API keys and configuration
```

2. **Basic usage:**
```python
from llm_factory import get_llm

# Create an LLM instance
llm = get_llm()

# Use it
response = llm.invoke("What are Large Language Models?")
print(response.content)
```

## 📋 Environment Variables

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_BASE` | Custom OpenAI API base URL | None |
| `OPENAI_MODEL` | Default model to use | `azure/model-router` |
| `OPENAI_TEMPERATURE` | Default temperature | `0.7` |
| `OPENAI_MAX_TOKENS` | Maximum tokens per request | Unlimited |
| `OPENAI_TIMEOUT` | Request timeout in seconds | `30` |
| `MODEL_TEMPERATURE_MIN` | Minimum temperature override | `0.0` |
| `MODEL_TEMPERATURE_MAX` | Maximum temperature override | `2.0` |

### Embedding-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_API_KEY` | Alternative API key for embeddings | Uses OPENAI_API_KEY |
| `EMBEDDING_BASE_URL` | Alternative base URL for embeddings | Uses OPENAI_API_BASE |
| `EMBEDDING_MODEL` | Default embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | Embedding dimensions | Auto-detected |

## 💻 Usage

### Synchronous Usage

#### Basic LLM Operations
```python
from llm_factory import get_llm, get_llm_embeddings

# Create LLM instance
llm = get_llm()

# Ask a question
response = llm.invoke("What are the benefits of artificial intelligence?")
print(response.content)

# Customize model and temperature
custom_llm = get_llm(model="azure/gpt-4-turbo", temperature=0.8)
response = custom_llm.invoke("Explain quantum computing simply")
```

#### Embedding Operations
```python
from llm_factory import get_llm_embeddings

# Create embedding instance
embeddings = get_llm_embeddings()

# Generate embeddings
vectors = embeddings.embed_query("Your text here")
print(f"Embedding dimensions: {len(vectors)}")
```

### Asynchronous Usage

```python
import asyncio
from llm_factory import get_llm_async, get_llm_embeddings_async

async def main():
    # Create async LLM instance
    llm = await get_llm_async()

    # Ask questions asynchronously
    response = await llm.ainvoke("What future AI advancements interest you?")
    print(response.content)

    # Multiple async operations
    tasks = [
        llm.ainvoke("Explain machine learning"),
        llm.ainvoke("What ethical considerations exist for AI?")
    ]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

```python
from llm_factory import (
    get_llm_config,
    set_temperature_constraints,
    validate_and_adjust_temperature
)

# Get configuration object
config = get_llm_config(model="azure/gpt-4-turbo")
print(f"Model: {config.model}")
print(f"Temperature: {config.temperature}")

# Set custom temperature constraints for specific models
set_temperature_constraints("custom-model", min_temp=1.0, max_temp=1.5)

# Validate temperature for any model
valid_temp = validate_and_adjust_temperature("azure/gpt-4-turbo", 0.5)
print(f"Adjusted temperature: {valid_temp}")  # Will be 1.0 for GPT-4
```

## 🔧 API Reference

### Core Functions

#### `get_llm(model=None) -> ChatOpenAI`
Creates and returns a configured ChatOpenAI instance.

**Parameters:**
- `model` (str, optional): Model identifier to override default

**Returns:** Configured ChatOpenAI instance

#### `get_llm_async(model=None) -> ChatOpenAI`
Async version of `get_llm()`.

#### `get_llm_embeddings(model=None, api_key=None, base_url=None) -> OpenAIEmbeddings`
Creates and returns a configured OpenAIEmbeddings instance.

**Parameters:**
- `model` (str, optional): Embedding model to use
- `api_key` (str, optional): Override API key
- `base_url` (str, optional): Override base URL

#### `get_llm_embeddings_async(...) -> OpenAIEmbeddings`
Async version of `get_llm_embeddings()`.

### Configuration Functions

#### `get_llm_config(model=None) -> LLMConfig`
Get LLM configuration object with optional model override.

#### `get_embedding_config(model=None) -> EmbeddingConfig`
Get embedding configuration object with optional model override.

#### `set_temperature_constraints(model, min_temp, max_temp)`
Dynamically set temperature constraints for a specific model.

#### `validate_and_adjust_temperature(model, temperature) -> float`
Validate and adjust temperature according to model constraints.

## ⚙️ Temperature Management

### Model-Agnostic Temperature Constraints

The library automatically adjusts temperatures based on model capabilities:

- **GPT-4 Series** (`gpt-4*`, `grok`): Min 1.0, Max 2.0
- **GPT-3.5 Series** (`gpt-3.5*`): Min 0.0, Max 2.0
- **Claude Models** (`claude`): Min 0.0, Max 2.0
- **Azure Router** (`azure/model-router`): Configurable via environment

### Temperature Validation Examples

```python
from llm_factory import get_llm

# Automatically adjusted to 1.0 (GPT-4 minimum)
llm1 = get_llm(model="azure/gpt-4-turbo", temperature=0.5)

# Kept at 0.7 (within GPT-3.5 range)
llm2 = get_llm(model="azure/gpt-3.5-turbo", temperature=0.7)

# Custom model with runtime constraints
from llm_factory import set_temperature_constraints
set_temperature_constraints("custom-model", 0.8, 1.2)
```

## 🔍 Troubleshooting

### Common Issues

#### "Invalid model name" Error
```python
# Solution: Use correct model identifier
llm = get_llm(model="azure/model-router")  # Instead of "gpt-3.5-turbo"
```

#### Temperature Warning
```python
# Temperature automatically adjusted with warning
llm = get_llm(model="azure/gpt-4-turbo", temperature=0.5)
# Output: Warning: Temperature 0.5 below minimum 1.0 for model azure/gpt-4-turbo, adjusted to 1.0
```

#### Missing Environment Variables
```python
# Ensure .env file exists with proper values
echo "OPENAI_API_KEY=your_key_here" > .env
echo "OPENAI_API_BASE=your_base_url" >> .env
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see detailed logs about temperature adjustments and API calls
```

## 🔐 Security

This project includes comprehensive security measures:

### Credential Protection
- `.env` files are automatically git-ignored
- `.env.example` provides a template with placeholder values
- `.gitignore` includes patterns for common credential files
- No hardcoded API keys or secrets

### Best Practices
```bash
# ✅ Good: Use environment variables
OPENAI_API_KEY=your_actual_key

# ❌ Bad: Never hardcode in source code
api_key = "sk-proj-abc123"  # Don't do this!
```

## 🧪 Testing

Run the included example:
```bash
# Default: runs async version (production-optimal)
python main.py

# Run synchronous version for debugging
python main.py --sync-only
```

Expected output (default async):
```
=== Asynchronous LLM Demo ===
Prompt: Explain the benefits of asynchronous programming.
-------------------------------------------------------
[LLM Response Here]
```

## 🛠️ Development

### Code Quality
```bash
# Run linting
pipenv run flake8 . --max-line-length=88

# Install development dependencies
pipenv install --dev flake8
```

### Project Structure
```
├── main.py                    # Example usage and entry point (async-first)
├── llm_factory/              # Core library package
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   └── factory.py           # Factory functions
├── Pipfile                   # Dependencies
├── Pipfile.lock             # Dependency lock file
├── .env.example              # Environment template (commit to version control)
├── .env                      # Your actual environment variables (git ignored)
├── .gitignore               # Git ignore file with security patterns
└── README.md                # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and run tests
4. Run linting: `flake8 . --max-line-length=88`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature/your-feature`
7. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the API documentation above

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Powered by [OpenAI](https://openai.com/) compatible APIs
- Configured with [python-dotenv](https://github.com/theskumar/python-dotenv)

Happy prompting! 🚀