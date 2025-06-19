<div align="center">

# 🐝 TangoBee Local Arena

**The Ultimate Peer-to-Peer AI Model Evaluation Tool**

*Run the same revolutionary evaluation system used by [TangoBee Arena](https://tangobee.sillymode.com) locally with your own OpenRouter API key!*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-green.svg)](https://openrouter.ai/)
[![TangoBee](https://img.shields.io/badge/TangoBee-Arena-gold.svg)](https://tangobee.sillymode.com)

![TangoBee Logo](assets/logo.png)

*Where AI models dance together in evaluation harmony* 🕺💃

---

</div>

## 🌟 What is TangoBee Local Arena?

**Finally, a fair way to evaluate AI models!** 🎯

TangoBee Local Arena brings the revolutionary "tango" evaluation methodology to your local machine. Unlike traditional benchmarks that can be gamed or biased, this tool implements a **true peer-to-peer AI evaluation system** where AI models evaluate each other across five critical domains:

| Domain | 🧠 Reasoning | 💻 Coding | 🗣️ Language | 🔢 Mathematics | 🎨 Creativity |
|--------|--------------|-----------|-------------|----------------|---------------|
| **Focus** | Logic & Critical Thinking | Programming & Development | Communication & Understanding | Problem Solving & Proofs | Innovation & Originality |
| **Example** | Complex reasoning puzzles | Algorithm implementation | Nuanced text analysis | Geometric calculations | Creative problem solving |

### 🎭 Why "Tango" Evaluation?

> *"It takes two to tango"* - and it takes multiple AI models to fairly evaluate each other!

❌ **Traditional benchmarks:** Static questions, human bias, training data contamination  
✅ **TangoBee method:** Dynamic questions, peer evaluation, eliminated bias

### The "Tango" Methodology

Our revolutionary three-phase evaluation process ensures fair, unbiased assessment:

1. **Phase 1: Question Generation** - Each model generates questions for all domains
2. **Phase 2: Answer Collection** - All models answer all questions from all other models  
3. **Phase 3: Peer Evaluation** - Each model evaluates answers to its own questions only

This eliminates single-point bias and creates a truly peer-to-peer evaluation ecosystem!

## 📊 Real TangoBee Arena Results

*Here are the latest rankings from the live TangoBee Arena (as of June 17, 2025):*

| 🏆 Rank | Model | Total | 🧠 Reasoning | 💻 Coding | 🗣️ Language | 🔢 Math | 🎨 Creativity |
|---------|--------|-------|-------------|----------|-------------|---------|---------------|
| 🥇 #1 | **OpenAI/O3 Pro** | **8.72** | 8.14 | **9.29** | 8.43 | **9.50** | 8.25 |
| 🥈 #2 | **Anthropic/Opus 4** | **8.32** | 7.71 | 8.13 | **9.38** | 7.75 | **8.63** |
| 🥉 #3 | **Anthropic/Sonnet 4** | **8.27** | **8.43** | 7.14 | 8.86 | 8.29 | **8.63** |

*🔥 Run the same evaluation system that generated these results!*

**🎯 Key Insights from Live Arena:**
- 🚀 **O3 Pro dominates** in coding (9.29) and mathematics (9.50)
- 🗣️ **Opus 4 excels** in language understanding (9.38) and creativity (8.63)  
- 🧠 **Sonnet 4 leads** in reasoning (8.43) with strong creative abilities
- ⚡ **Tight competition** - only 0.45 points separate top 3 models!
- 📊 **Each model has unique strengths** across different domains

*Now you can run this same peer-to-peer evaluation methodology locally!*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key ([Get yours here](https://openrouter.ai/keys))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tianrenmaogithub/tangobee-local-arena.git
   cd tangobee-local-arena
   ```

2. **Run the setup:**
   ```bash
   python setup.py
   ```

3. **Configure your API key:**
   
   **Option A: Environment variable (recommended)**
   ```bash
   export OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```
   
   **Option B: Config file**
   ```bash
   # Edit config/api_key.txt.template and rename to api_key.txt
   cp config/api_key.txt.template config/api_key.txt
   # Edit the file and add your API key
   ```

4. **Run your first evaluation:**
   ```bash
   python run_evaluation.py --quick
   ```

That's it! 🎉 Your HTML report will be generated in the `reports/` directory.

## 📊 Features

### ✨ Core Capabilities

- **🔄 Peer-to-Peer Evaluation** - True AI-evaluating-AI methodology
- **📈 Trend Analysis** - Track performance over time with beautiful charts
- **🎯 Same-Day Overwriting** - Run multiple evaluations per day
- **📱 Responsive HTML Reports** - Beautiful, mobile-friendly results
- **🌐 Multi-Model Support** - Works with any OpenRouter-supported model
- **💰 Cost-Effective** - Uses cheap models by default (GPT-4o-mini, Claude-3.5-haiku)

### 📊 Generated Visualizations

- **Performance Heatmaps** - See how each model rates others
- **Domain-Specific Analysis** - Detailed breakdown by capability area
- **Trend Plots** - 6-month historical performance tracking
- **Comprehensive Reports** - All data in beautiful HTML format

## 🛠️ Usage

### Basic Usage

```bash
# Quick evaluation with 2 cheap models
python run_evaluation.py --quick

# Full evaluation with default models
python run_evaluation.py

# Custom models
python run_evaluation.py --models openai/gpt-4o anthropic/claude-3-5-sonnet

# Direct API usage
python src/local_arena.py --api-key your_key_here
```

### Configuration

**🔥 Cutting-Edge Models Supported** (in `config/models.txt`):

| Model | Provider | Strengths | Cost Level |
|-------|----------|-----------|------------|
| `openai/o3-pro` | OpenAI | 🧠 Advanced reasoning | Premium |
| `openai/o3` | OpenAI | 🎯 Latest capabilities | High |
| `anthropic/claude-opus-4` | Anthropic | 🎨 Creative excellence | Premium |
| `anthropic/claude-sonnet-4` | Anthropic | ⚡ Fast & capable | Medium |
| `google/gemini-2.5-pro-preview` | Google | 🌟 Multimodal power | High |
| `deepseek/deepseek-r1-0528` | DeepSeek | 💻 Coding mastery | Budget |
| `openai/gpt-4o-mini` | OpenAI | 💰 Cost-effective | Budget |
| `anthropic/claude-3.5-haiku` | Anthropic | 🚀 Fast reasoning | Budget |

*Mix and match any models for your perfect evaluation setup!*

**Estimated Costs:**
- 2 models (quick): ~$0.50-1.00
- 3 models (default): ~$2.00-4.00  
- 5 models: ~$10.00-20.00

*Costs depend on model pricing and response lengths*

## 📁 Project Structure

```
tangobee-local-arena/
├── src/
│   └── local_arena.py          # Main evaluation engine
├── config/
│   ├── models.txt              # Default models configuration
│   └── api_key.txt.template    # API key template
├── assets/
│   └── logo.png               # TangoBee logo
├── data/                      # Local database and logs
├── reports/                   # Generated HTML reports
│   └── visualizations/        # Charts and heatmaps
├── setup.py                   # Easy setup script
├── run_evaluation.py          # User-friendly runner
└── requirements.txt           # Python dependencies
```

## 🔧 Advanced Usage

### Adding Custom Models

Edit `config/models.txt`:
```
openai/gpt-4o-mini
anthropic/claude-3-5-haiku
google/gemini-flash-1.5
meta/llama-3.1-8b-instruct
```

### API Key Management

**Environment Variable:**
```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

**Config File:**
```bash
echo "sk-or-v1-xxxxx" > config/api_key.txt
```

### Custom Evaluation

```python
from src.local_arena import TangoBeeLocalArena

arena = TangoBeeLocalArena(api_key="your-key")
report_path = arena.run_full_evaluation([
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-haiku"
])
print(f"Report generated: {report_path}")
```

## 📈 Understanding Results

### Scoring System
- **Scale:** 0-10 (higher is better)
- **Methodology:** Each model scores answers to its own questions
- **Aggregation:** Domain scores averaged for total score

### Report Sections
1. **🏆 Leaderboard** - Final rankings with medals
2. **📊 Performance Heatmaps** - Model-vs-model evaluation matrix
3. **📈 Trend Analysis** - Historical performance tracking

### Interpreting Heatmaps
- **Rows:** Evaluating models (judges)
- **Columns:** Evaluated models (participants)
- **Colors:** Green = high scores, Red = low scores
- **Numbers:** Average scores given by each judge

## 🔍 Troubleshooting

### Common Issues

**Error: API key not found**
```bash
# Set environment variable
export OPENROUTER_API_KEY=your_key

# Or create config file
echo "your_key" > config/api_key.txt
```

**Error: Rate limiting**
- The tool includes automatic retry logic
- Consider using fewer models for testing
- Check your OpenRouter account limits

**Error: Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Slow performance**
- Use `--quick` flag for faster testing
- Choose faster models (gpt-4o-mini, claude-3.5-haiku)
- Reduce number of models being evaluated

### Debug Mode

```bash
# Check logs
tail -f data/tangobee_local.log

# Verbose output
python src/local_arena.py --api-key your_key --models openai/gpt-4o-mini
```

## 🏆 Why Choose TangoBee Local Arena?

### 🎯 **Unbiased & Fair**
- ✅ No single entity controls evaluation criteria
- ✅ Models evaluate each other, eliminating human bias
- ✅ Dynamic questions prevent training data contamination
- ✅ Peer-to-peer methodology ensures fairness

### 🚀 **Easy & Accessible**
- ✅ One-command setup with `python setup.py`
- ✅ Works with any OpenRouter-supported model
- ✅ Beautiful HTML reports with visualizations
- ✅ Budget-friendly default model selection

### 📊 **Comprehensive & Insightful**
- ✅ Five distinct evaluation domains
- ✅ Historical trend analysis
- ✅ Performance heatmaps
- ✅ Detailed statistical breakdowns

### 🌟 **Proven Methodology**
- ✅ Based on live [TangoBee Arena](https://tangobee.sillymode.com)
- ✅ Trusted evaluation approach
- ✅ Continuously refined system
- ✅ Community-driven improvements

## 🤝 Contributing

We welcome contributions! This project is inspired by the live TangoBee Arena at [tangobee.sillymode.com](https://tangobee.sillymode.com).

### Development Setup

```bash
git clone https://github.com/yourusername/tangobee-local-arena.git
cd tangobee-local-arena
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Areas for Contribution

- 📊 Additional visualization types
- 🔧 New evaluation domains
- 🎯 Performance optimizations
- 📱 Mobile app version
- 🌐 Web interface
- 📈 Advanced analytics

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **TangoBee Arena** - Original inspiration and methodology
- **OpenRouter** - API platform enabling model access
- **The AI Community** - For advancing model evaluation techniques

## 📞 Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/tianrenmaogithub/tangobee-local-arena/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/tianrenmaogithub/tangobee-local-arena/discussions)
- 🌐 **Live Demo:** [TangoBee Arena](https://tangobee.sillymode.com)

---

<div align="center">

**🐝 Made with the TangoBee evaluation methodology 🐝**

*Where AI models dance together in evaluation harmony*

</div>