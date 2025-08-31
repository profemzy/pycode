#!/bin/bash
# Quick start script for LLM Factory Chatbot
set -e

echo "🚀 LLM Factory Chatbot - Quick Start Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.12+ is available
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Found Python $PYTHON_VERSION"
        if [[ $(echo "$PYTHON_VERSION 3.10" | awk '{print ($1 >= $2)}') == 1 ]]; then
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Found Python $PYTHON_VERSION"
        if [[ $(echo "$PYTHON_VERSION 3.10" | awk '{print ($1 >= $2)}') == 1 ]]; then
            PYTHON_CMD="python"
            return 0
        fi
    fi
    
    print_error "Python 3.10+ is required but not found"
    exit 1
}

# Check if .env file exists
check_env() {
    print_status "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning ".env file not found. Copying from .env.example"
            cp .env.example .env
            print_warning "Please edit .env file with your actual API keys before proceeding"
            echo
            echo "Required: OPENAI_API_KEY=your_openai_api_key_here"
            echo "Optional: TAVILY_API_KEY=your_tavily_api_key_here"
            echo
            read -p "Press Enter after you've configured the .env file..."
        else
            print_error ".env.example file not found. Please create .env with your configuration"
            exit 1
        fi
    else
        print_success ".env file found"
    fi
    
    # Check if OPENAI_API_KEY is set
    if [ -f ".env" ]; then
        source .env
        if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
            print_error "OPENAI_API_KEY not configured in .env file"
            exit 1
        fi
        print_success "API key configuration verified"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Try pipenv first
    if command -v pipenv &> /dev/null; then
        print_status "Using pipenv for dependency management"
        
        # Try to install with pipenv, fall back to pip if it fails
        if pipenv install; then
            print_success "Dependencies installed with pipenv"
            RUNNER="pipenv run"
        else
            print_warning "pipenv install failed, falling back to pip"
            install_with_pip
        fi
    else
        print_status "pipenv not found, using pip with virtual environment"
        install_with_pip
    fi
}

# Install with pip and virtual environment
install_with_pip() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    fi
    
    print_status "Activating virtual environment and installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    # Generate requirements from Pipfile if available
    if [ -f "Pipfile" ]; then
        pipenv requirements > temp_requirements.txt
        pip install -r temp_requirements.txt
        rm temp_requirements.txt
    else
        print_error "No Pipfile found for dependency installation"
        exit 1
    fi
    print_success "Dependencies installed with pip"
    RUNNER="venv/bin/python"
}

# Run health check
run_health_check() {
    print_status "Running health check..."
    if $RUNNER python health_check.py; then
        print_success "Health check passed!"
    else
        print_warning "Health check failed, but continuing..."
        print_warning "Some components may not be fully functional"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    if $RUNNER pytest tests/ -v --tb=short; then
        print_success "All tests passed!"
    else
        print_warning "Some tests failed, but continuing..."
    fi
}

# Start application
start_application() {
    print_success "Setup complete! Starting the chatbot..."
    echo
    echo "🤖 LangGraph Chatbot will start in interactive mode"
    echo "📝 Type your messages and press Enter"
    echo "🔍 Use phrases like 'search online for...' to trigger web search"
    echo "💬 Type 'quit', 'exit', or 'bye' to end the conversation"
    echo
    print_status "Starting application..."
    
    $RUNNER python main.py
}

# Alternative deployment options
show_alternatives() {
    echo
    print_status "Alternative deployment options:"
    echo
    echo "1. Docker (if BuildKit issues are resolved):"
    echo "   cd docker && docker-compose up -d"
    echo
    echo "2. Direct Python execution:"
    echo "   python main.py"
    echo
    echo "3. Cloud deployment:"
    echo "   See PRODUCTION_GUIDE.md for detailed instructions"
    echo
    echo "4. Docker troubleshooting:"
    echo "   See DOCKER_TROUBLESHOOTING.md for solutions"
}

# Main execution
main() {
    echo
    print_status "Starting setup process..."
    
    # Run checks and setup
    check_python
    check_env
    install_dependencies

    # Present a persistent menu that returns after each action until the user quits
    while true; do
        echo
        echo "What would you like to do?"
        echo "1. Run health check only"
        echo "2. Run tests only"
        echo "3. Start the chatbot application"
        echo "4. Run health check, tests, then start chatbot"
        echo "5. Show alternative deployment options"
        echo "6. Quit / Exit"
        echo
        read -p "Enter your choice (1-6): " choice

        case $choice in
            1)
                run_health_check
                ;;
            2)
                run_tests
                ;;
            3)
                start_application
                ;;
            4)
                run_health_check
                echo
                run_tests
                echo
                start_application
                ;;
            5)
                show_alternatives
                ;;
            6|q|Q|quit|exit)
                print_status "Exiting quick start. Goodbye."
                break
                ;;
            *)
                print_warning "Invalid choice. Please select a valid option (1-6)."
                ;;
        esac

        # Small pause to make output readable before showing the menu again
        echo
        read -p "Press Enter to return to the menu..."
    done
}
 
# Run main function
main "$@"