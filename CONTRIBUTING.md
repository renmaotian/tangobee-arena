# 🐝 Contributing to TangoBee Local Arena

We love your input! We want to make contributing to TangoBee Local Arena as easy and transparent as possible, whether it's:

- 🐛 Reporting a bug
- 💬 Discussing the current state of the code
- 🚀 Submitting a fix
- 🎯 Proposing new features
- 🤖 Adding support for new models
- 📚 Improving documentation

## 🔄 Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### 📋 Pull Request Process

1. **Fork** the repo and create your branch from `main`
2. **Add tests** if you've added code that should be tested
3. **Update documentation** if you've changed APIs or functionality
4. **Ensure the test suite passes**
5. **Make sure your code lints**
6. **Issue that pull request!**

## 🏗️ Development Setup

```bash
# Clone your fork
git clone https://github.com/renmaotian/tangobee-arena.git
cd tangobee-arena

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key for testing
export OPENROUTER_API_KEY=your_test_key
```

## 🧪 Testing

Before submitting a PR, please test your changes:

```bash
# Quick test with 2 models
python run_evaluation.py --quick

# Test setup
python setup.py

# Test with custom models
python run_evaluation.py --models openai/gpt-4o-mini anthropic/claude-3-5-haiku
```

## 📝 Code Style

- Use **clear, descriptive variable names**
- Add **docstrings** to all functions and classes
- Follow **PEP 8** style guidelines
- Keep **functions focused** and under 50 lines when possible
- Add **comments** for complex logic

### Example:
```python
def evaluate_answer(self, evaluator_model: str, question: str, answer: str, domain: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Have a model evaluate an answer to its own question.
    
    Args:
        evaluator_model: OpenRouter model name for evaluation
        question: The original question
        answer: The answer to evaluate
        domain: Domain category (reasoning, coding, etc.)
        
    Returns:
        Tuple of (score, reasoning) or (None, None) if failed
    """
```

## 🎯 Areas We Need Help With

### 🔥 High Priority
- 📊 **New Visualization Types** - Additional chart types, interactive plots
- 🎨 **UI Improvements** - Better HTML report styling, responsive design
- 🚀 **Performance Optimizations** - Faster evaluation, better concurrency
- 📱 **Mobile Support** - Mobile-friendly reports and interfaces

### 🌟 Medium Priority
- 🤖 **New Model Support** - Integration with additional providers
- 📈 **Advanced Analytics** - Statistical analysis, confidence intervals
- 🔧 **Configuration Options** - More customizable evaluation parameters
- 🌐 **Web Interface** - Optional web UI for local hosting

### 💡 Cool Ideas
- 📺 **Live Streaming** - Real-time evaluation progress
- 🎮 **Interactive Mode** - User-guided evaluation flows
- 📊 **Comparison Tools** - Side-by-side model comparisons
- 🔄 **Automated Scheduling** - Periodic evaluation runs

## 🐛 Bug Reports

Great bug reports tend to have:

- **Quick summary** and/or background
- **Steps to reproduce** - be specific!
- **What you expected** would happen
- **What actually happens**
- **Notes** - possibly including why you think this might be happening

## 🚀 Feature Requests

We actively welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** you're trying to solve
3. **Explain your proposed solution**
4. **Consider the impact** on existing users
5. **Think about implementation** complexity

## 📜 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Recognition

All contributors will be acknowledged in our README and releases. Significant contributions may be highlighted in our documentation.

## 📞 Questions?

- 💬 **GitHub Discussions** for general questions
- 🐛 **GitHub Issues** for bugs and feature requests
- 🌐 **TangoBee Arena** at [tangobee.sillymode.com](https://tangobee.sillymode.com)

---

<div align="center">

**🐝 Thank you for helping make TangoBee Local Arena better! 🐝**

</div>