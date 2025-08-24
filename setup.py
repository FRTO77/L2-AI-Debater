"""
Haresha - Advanced AI Debate Platform
Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from template."""
    env_content = """# Haresha - Advanced AI Debate Platform
# Environment Variables Configuration

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct

# Application Settings
SESSION_DIRECTORY=debate_sessions
MAX_SESSIONS_PER_USER=100
SESSION_RETENTION_DAYS=90

# Analytics Settings
ENABLE_ANALYTICS=true
SAVE_ANALYTICS=true
ANALYTICS_RETENTION_DAYS=30

# UI Settings
THEME=Light
ANIMATION_SPEED=0.05
SHOW_METRICS=true
SHOW_FALLACIES=true

# Advanced Settings
LOG_LEVEL=INFO
DEBUG_MODE=false
ENABLE_LOGGING=true
REAL_TIME_ANALYSIS=true
AUTO_SAVE=true
ENABLE_MODERATOR=true
"""
    
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists")
        return True
    
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        print("📝 Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "debate_sessions",
        "logs",
        "exports"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")
            return False
    
    return True


def check_streamlit():
    """Check if Streamlit is properly installed."""
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False


def check_langchain():
    """Check if LangChain is properly installed."""
    try:
        import langchain
        print(f"✅ LangChain version: {langchain.__version__}")
        return True
    except ImportError:
        print("❌ LangChain not found")
        return False


def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic tests...")
    
    # Test imports
    try:
        from debate_engine import DebateEngine, DebateSession
        from argument_analyzer import ArgumentAnalyzer
        from config import Config, load_config
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test configuration
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    return True


def main():
    """Main setup function."""
    print("🚀 Haresha Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Check installations
    if not check_streamlit():
        print("⚠️  Streamlit installation issue")
    
    if not check_langchain():
        print("⚠️  LangChain installation issue")
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: streamlit run streamlit_app.py")
    print("3. Open http://localhost:8501 in your browser")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()

