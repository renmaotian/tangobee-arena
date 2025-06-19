#!/usr/bin/env python3
"""
TangoBee Local Arena Setup Script
Easy setup for local AI model evaluation
"""

import os
import sys
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).parent
    directories = [
        'data',
        'reports',
        'reports/visualizations',
        'config'
    ]
    
    print("Creating directories...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"  ✓ {directory}")

def setup_config():
    """Setup configuration files"""
    base_dir = Path(__file__).parent
    config_dir = base_dir / 'config'
    
    # Create API key template
    api_key_file = config_dir / 'api_key.txt.template'
    if not api_key_file.exists():
        with open(api_key_file, 'w') as f:
            f.write("# Replace this line with your OpenRouter API key\n")
            f.write("# Get your API key from: https://openrouter.ai/keys\n")
            f.write("# Then rename this file to api_key.txt\n")
            f.write("your_openrouter_api_key_here\n")
        print("  ✓ Created config/api_key.txt.template")
    
    # Create models config
    models_file = config_dir / 'models.txt'
    if not models_file.exists():
        with open(models_file, 'w') as f:
            f.write("# Default models for evaluation (cheap models)\n")
            f.write("openai/gpt-4o-mini\n")
            f.write("anthropic/claude-3-5-haiku\n")
            f.write("deepseek/deepseek-r1\n")
            f.write("\n# Add more models here if desired\n")
            f.write("# Example: google/gemini-flash-1.5\n")
        print("  ✓ Created config/models.txt")

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install Python requirements"""
    base_dir = Path(__file__).parent
    requirements_file = base_dir / 'requirements.txt'
    
    if requirements_file.exists():
        print("Installing Python requirements...")
        os.system(f"pip install -r {requirements_file}")
        print("  ✓ Requirements installed")
    else:
        print("⚠️  requirements.txt not found")

def main():
    print("🐝 TangoBee Local Arena Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Setup directories
    setup_directories()
    
    # Setup configuration
    print("\nSetting up configuration...")
    setup_config()
    
    # Install requirements
    print("\nInstalling dependencies...")
    try:
        install_requirements()
    except Exception as e:
        print(f"⚠️  Error installing requirements: {e}")
        print("   You can install manually with: pip install -r requirements.txt")
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Get your OpenRouter API key from: https://openrouter.ai/keys")
    print("2. Either:")
    print("   - Set environment variable: export OPENROUTER_API_KEY=your_key")
    print("   - Or edit config/api_key.txt.template and rename to api_key.txt")
    print("3. Run evaluation: python src/local_arena.py")
    print("4. Or use the runner: python run_evaluation.py")

if __name__ == "__main__":
    main()