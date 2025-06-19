#!/usr/bin/env python3
"""
TangoBee Local Arena Runner
Simple script to run evaluations with common configurations
"""

import argparse
import os
import sys
from pathlib import Path

def get_models_from_config():
    """Load models from config file"""
    config_file = Path(__file__).parent / 'config' / 'models.txt'
    if not config_file.exists():
        return ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku", "deepseek/deepseek-r1"]
    
    models = []
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                models.append(line)
    
    return models if models else ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku", "deepseek/deepseek-r1"]

def check_api_key():
    """Check if API key is available"""
    # Check environment variable
    if os.getenv('OPENROUTER_API_KEY'):
        return True
    
    # Check config file
    config_file = Path(__file__).parent / 'config' / 'api_key.txt'
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read().strip()
            if content and not content.startswith('#') and 'your_openrouter_api_key_here' not in content:
                return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='TangoBee Local Arena - Easy Runner')
    parser.add_argument('--quick', action='store_true', help='Run with minimal models for quick testing')
    parser.add_argument('--models', nargs='+', help='Specify models to evaluate')
    parser.add_argument('--api-key', help='OpenRouter API key')
    
    args = parser.parse_args()
    
    print("🐝 TangoBee Local Arena")
    print("=" * 40)
    
    # Check API key
    if not args.api_key and not check_api_key():
        print("❌ OpenRouter API key not found!")
        print("\nPlease either:")
        print("1. Set environment variable: export OPENROUTER_API_KEY=your_key")
        print("2. Create config/api_key.txt with your API key")
        print("3. Use --api-key argument")
        print("\nGet your API key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    # Determine models to use
    if args.models:
        models = args.models
    elif args.quick:
        models = ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku"]  # Just 2 models for quick test
    else:
        models = get_models_from_config()
    
    print(f"📊 Models to evaluate: {len(models)}")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    print(f"\n⏱️  Estimated time: {len(models) * len(models) * 5 * 2} minutes")
    print("   (This includes question generation, answering, and evaluation)")
    
    # Confirm before running
    if not args.quick:
        response = input("\n🚀 Ready to start evaluation? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Evaluation cancelled.")
            sys.exit(0)
    
    # Import and run the arena
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from local_arena import TangoBeeLocalArena
        
        print("\n🎯 Starting evaluation...")
        arena = TangoBeeLocalArena(api_key=args.api_key)
        report_path = arena.run_full_evaluation(models)
        
        print(f"\n🏆 Evaluation complete!")
        print(f"📊 Report saved to: {report_path}")
        print(f"🌐 Open the HTML report in your browser to see results!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Try with fewer models first (--quick flag)")
        print("4. Check the log file: data/tangobee_local.log")
        sys.exit(1)

if __name__ == "__main__":
    main()