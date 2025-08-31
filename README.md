# LLM Factory Chatbot

A production-ready, enterprise-grade chatbot powered by LangGraph and OpenAI integration. Features comprehensive security, monitoring, testing, and deployment capabilities designed for scalable production environments.

## 🌟 Key Features

### Core Functionality
- **LangGraph-Based Architecture**: Advanced conversation flow management with memory
- **Web Search Integration**: Tavily and DuckDuckGo search capabilities
- **Streaming Support**: Real-time response streaming for better UX
- **Smart Tool Binding**: Contextual tool activation based on user intent

### Production Enhancements
- **🔒 Security**: Input validation, sanitization, rate limiting, API key protection
- **📊 Monitoring**: Comprehensive health checks, performance metrics, system monitoring
- **🧪 Testing**: Full test suite with 60 tests and 100% pass rate
- **🐳 Containerization**: Docker support with production-ready configurations
- **⚡ Performance**: Async operations, caching, connection pooling
- **🔧 Configuration**: Environment-based config with validation and security checks
- **🚨 Error Handling**: Robust error recovery and graceful degradation

## 📦 Quick Start

### Prerequisites
- Python 3.12 or higher
- pipenv (recommended)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd pycode

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Option 1: Use the quick start script (recommended)
./quick_start.sh

# Option 2: Manual setup
pipenv install --dev
pipenv run python main.py
```

### Docker Deployment
```bash
# Simple deployment
cd docker
docker-compose up -d

# With monitoring stack
docker-compose --profile monitoring up -d
```

## 📋 Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Configuration
```bash
# LLM Settings
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_TIMEOUT=30

# Search Configuration
TAVILY_API_KEY=your_tavily_api_key_here
SEARCH_TIMEOUT=6

# Production Settings
LOG_LEVEL=WARNING  # INFO for dev, WARNING for production
```

## 💻 Usage Examples

### Interactive Chat
```bash
pipenv run python main.py
```

### Health Monitoring
```bash
# Quick health check
pipenv run python health_check.py

# Returns exit codes:
# 0 = healthy, 1 = unhealthy, 2 = degraded
```

### Programmatic Usage
```python
from llm_factory import get_llm
from tools import get_search_tools

# Create LLM with search capabilities
llm = get_llm()
tools = get_search_tools()
llm_with_tools = llm.bind_tools(tools)

# Process user input
response = llm_with_tools.invoke([
    {"role": "user", "content": "Search for Python best practices"}
])
print(response.content)
```

## 🧪 Testing

### Running Tests
```bash
# All tests (60 tests, 100% pass rate)
pipenv run pytest tests/ -v

# With coverage
pipenv run pytest tests/ --cov=.

# Specific test categories
pipenv run pytest tests/test_utils.py -v      # Security & validation
pipenv run pytest tests/test_config.py -v    # Configuration
```

### Code Quality
```bash
# Linting
pipenv run flake8 . --max-line-length=79

# Type checking
pipenv run mypy . --ignore-missing-imports

# Formatting
pipenv run black .
```

## 🐳 Production Deployment

### Docker
```bash
# Production deployment
cd docker
docker-compose up -d

# Check health
docker exec llm-chatbot python health_check.py
```

### Environment Configuration
```yaml
# docker-compose.yml
environment:
  - LOG_LEVEL=WARNING           # Clean production logs
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - PIPENV_VERBOSITY=-1         # Suppress pipenv warnings
```

### Cloud Deployment
```bash
# Heroku
git push heroku main

# Railway
railway up

# Google Cloud Run
gcloud run deploy --source .
```

## 🔒 Security Features

### Input Protection
- **Sanitization**: HTML escaping, control character removal
- **Validation**: Length limits, format validation, content filtering
- **Rate Limiting**: Per-user request throttling (configurable)
- **Error Handling**: Secure error messages without information leakage

### Production Security
- **API Key Validation**: Secure format verification
- **Environment Isolation**: No hardcoded secrets
- **Container Security**: Non-root users, minimal attack surface
- **Vulnerability Scanning**: Automated security scanning in CI/CD

## 📊 Monitoring & Health Checks

### System Health
```bash
# Manual health check
python health_check.py

# Automated monitoring (returns appropriate exit codes)
if ! python health_check.py; then
    case $? in
        1) echo "CRITICAL: System unhealthy" ;;
        2) echo "WARNING: System degraded" ;;
    esac
fi
```

### Health Check Components
- **LLM Connectivity**: Response time and functionality
- **Search Tools**: Tavily/DuckDuckGo availability
- **System Resources**: CPU, memory, disk usage
- **Configuration**: Environment validation
- **Embeddings**: Vector generation capability

### Production Logging
- **Default Level**: WARNING (minimal output)
- **Third-party Suppression**: ERROR level for libraries (httpx, openai, etc.)
- **Debug Mode**: Set `LOG_LEVEL=DEBUG` for troubleshooting
- **Structured Format**: Timestamp, level, message

## 🏗️ Architecture

### System Components
```
User Input → Input Validation → Rate Limiting → LangGraph Workflow
    ↓               ↓               ↓               ↓
Security        Sanitization   Throttling    LLM Processing
                                                    ↓
                                             Search Tools
                                                    ↓
                                           Response Generation
                                                    ↓
                                          Output Sanitization
                                                    ↓
                                            User Response
```

### Key Modules
- **[`main.py`](main.py)**: LangGraph workflow orchestration
- **[`llm_factory/`](llm_factory/)**: LLM configuration and instance management
- **[`tools.py`](tools.py)**: Web search tool integration
- **[`utils.py`](utils.py)**: Security, validation, and utility functions
- **[`health_check.py`](health_check.py)**: Comprehensive system monitoring
- **[`tests/`](tests/)**: Complete test suite (60 tests, 100% pass)

## 🔧 Development

### Dependency Management
This project uses **Pipenv** for modern Python dependency management:

```bash
# Install dependencies
pipenv install --dev

# Add new dependency
pipenv install requests

# Update dependencies
pipenv update

# Security check
pipenv check
```

### Project Structure
```
├── main.py                    # Main application entry point
├── health_check.py           # System health monitoring
├── utils.py                  # Security and utility functions
├── tools.py                  # Web search tool integration
├── llm_factory/              # LLM configuration and management
├── tests/                    # Comprehensive test suite
├── docker/                   # Container configuration
├── .github/workflows/        # CI/CD pipeline
├── Pipfile                   # Dependency management
├── Pipfile.lock              # Locked dependency versions
├── .env.example              # Environment template
├── quick_start.sh           # Automated setup script
└── README.md                # This file
```

### Adding New Features
1. Write code with proper type hints
2. Add comprehensive tests
3. Update configuration if needed
4. Run quality checks
5. Test in Docker environment

## 🚀 Performance & Scaling

### Optimizations
- **Async Operations**: Non-blocking I/O for better concurrency
- **Caching**: Search results cached for 5 minutes (configurable)
- **Connection Pooling**: Efficient HTTP client reuse
- **Resource Limits**: Configurable CPU/memory constraints

### Scaling Guidelines
- **Horizontal Scaling**: Stateless design enables easy scaling
- **Load Balancing**: Use session persistence for conversation continuity
- **Auto-scaling**: Configure based on CPU/memory usage
- **Resource Monitoring**: Built-in health checks and metrics

## 🔍 Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# Check environment variables
pipenv run python -c "from llm_factory.config import validate_environment; validate_environment()"
```

#### Docker Issues
```bash
# If BuildKit errors occur
export DOCKER_BUILDKIT=0
docker-compose up -d --build

# Or use legacy build
cd docker && docker build --no-cache -t llm-chatbot .
```

#### Performance Issues
```bash
# Check system health
pipenv run python health_check.py

# Enable debug logging temporarily
export LOG_LEVEL=DEBUG
pipenv run python main.py
```

### Debug Mode
```bash
# Verbose logging for troubleshooting
export LOG_LEVEL=DEBUG
pipenv run python main.py

# Health check with details
LOG_LEVEL=INFO pipenv run python health_check.py
```

## 📈 CI/CD Pipeline

The project includes automated CI/CD with:
- **Testing**: All 60 tests run on every push
- **Quality Checks**: Linting, type checking, security scanning
- **Security**: Vulnerability scanning with Trivy and Bandit
- **Docker**: Automated container building and publishing
- **Deployment**: Multi-environment deployment support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with tests: `pipenv run pytest tests/`
4. Run quality checks: `pipenv run flake8 . && pipenv run mypy .`
5. Test in Docker: `cd docker && docker-compose up -d`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Production Ready** ✅ | **Enterprise Grade** ✅ | **Fully Tested** ✅

This chatbot is designed and tested for production environments with enterprise-grade security, monitoring, and reliability features.