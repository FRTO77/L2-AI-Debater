# ⚔️ Haresha - Advanced AI Debate Platform

A sophisticated multi-agent debate system with real-time analytics, argument analysis, and comprehensive session management.

## 🌟 Features

### 🏠 Debate Arena
- **Multi-Agent System**: PRO, CON, and AI Moderator agents
- **Flexible Debate Formats**: Classic, Multi-Perspective, Role-Play, Expert Panel
- **Real-time Interaction**: Dynamic argument generation and response
- **Round Management**: Configurable round limits with progress tracking
- **User Participation**: Active participation in debates with your own arguments

### 📊 Advanced Analytics
- **Argument Strength Analysis**: Real-time evaluation of argument quality
- **Sentiment Analysis**: Track emotional tone throughout debates
- **Logical Fallacy Detection**: Identify common reasoning errors
- **Performance Metrics**: Comprehensive statistics and visualizations
- **Speaker Comparison**: Compare performance across different participants

### 💾 Session Management
- **Auto-save**: Automatic session preservation
- **Session Loading**: Resume debates from where you left off
- **Bulk Export**: Export multiple sessions at once
- **Session History**: Complete audit trail of all debates

### 📤 Export Options
- **Multiple Formats**: JSON, Markdown, CSV exports
- **Custom Exports**: Select specific components to export
- **Analytics Integration**: Include performance metrics in exports
- **Batch Processing**: Export all sessions simultaneously

### ⚙️ Configuration
- **LLM Support**: OpenAI and Ollama integration
- **Preset Configurations**: Performance, Quality, Local, Debug presets
- **Real-time Settings**: Adjust parameters during debates
- **Theme Support**: Light, Dark, and Auto themes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (or Ollama for local models)

### Installation

1. **Clone or download the project**
```bash
cd Haresha
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project directory:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct

# Application Settings
SESSION_DIRECTORY=debate_sessions
LOG_LEVEL=INFO
DEBUG_MODE=false
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📁 Project Structure

```
Haresha/
├── streamlit_app.py          # Main Streamlit application
├── debate_engine.py          # Core debate logic and AI interactions
├── argument_analyzer.py      # Argument analysis and metrics
├── components.py             # UI components and utilities
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .env                      # Environment variables (create this)
```

## 🎯 Usage Guide

### Starting a Debate

1. **Navigate to Debate Arena**
   - Select "🏠 Debate Arena" from the sidebar

2. **Configure Debate**
   - Enter your debate topic
   - Choose debate format
   - Set number of rounds
   - Define positions for PRO and CON sides

3. **Advanced Settings**
   - Adjust AI creativity level
   - Set response length preferences
   - Enable/disable moderator
   - Configure auto-save

4. **Begin Debate**
   - Click "🚀 Start Debate"
   - Use the control buttons to generate responses
   - Add your own arguments in the text area

### Using Analytics

1. **View Real-time Metrics**
   - Navigate to "📊 Analytics"
   - Monitor argument strength over time
   - Track speaker performance
   - Analyze sentiment trends

2. **Review Detailed Analysis**
   - Check fallacy detection results
   - View improvement recommendations
   - Compare argument quality

### Managing Sessions

1. **Save Sessions**
   - Sessions auto-save by default
   - Manual save available in session manager

2. **Load Previous Debates**
   - Go to "💾 Session Manager"
   - Browse available sessions
   - Click "📥 Load Session"

3. **Export Data**
   - Navigate to "📤 Export"
   - Choose export format
   - Select components to include
   - Download your data

## 🔧 Configuration

### LLM Providers

#### OpenAI
```python
# In config.py or .env
llm_provider = "OpenAI"
openai_model = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
```

#### Ollama (Local)
```python
# In config.py or .env
llm_provider = "Ollama"
ollama_model = "llama3:8b-instruct"
```

### Configuration Presets

Apply predefined configurations:

```python
from config import apply_preset

# Performance optimized
apply_preset("performance")

# High quality debates
apply_preset("quality")

# Local model usage
apply_preset("local")

# Debug mode
apply_preset("debug")
```

## 📊 Analytics Features

### Argument Metrics
- **Strength**: Evaluates logical coherence and evidence
- **Clarity**: Measures readability and comprehension
- **Persuasiveness**: Assesses rhetorical effectiveness
- **Evidence Quality**: Rates supporting information
- **Logical Consistency**: Checks for contradictions

### Fallacy Detection
- **Ad Hominem**: Personal attacks
- **Straw Man**: Misrepresenting opponent's position
- **Appeal to Authority**: Unwarranted expert claims
- **False Dichotomy**: Oversimplified choices
- **Slippery Slope**: Unwarranted causal chains
- **Appeal to Emotion**: Manipulative emotional appeals

### Visualizations
- **Strength Timeline**: Track argument quality over time
- **Speaker Performance**: Compare participant effectiveness
- **Argument Type Distribution**: Analyze debate structure
- **Sentiment Analysis**: Monitor emotional tone

## 🛠️ Advanced Features

### Custom Export Formats
```python
# Export specific components
export_options = ["Arguments", "Analytics", "Timeline"]
custom_export = generate_custom_export(session, export_options)
```

### Real-time Analysis
```python
# Enable real-time argument analysis
config.real_time_analysis = True
config.show_metrics = True
config.show_fallacies = True
```

### Session Management
```python
# Auto-save configuration
config.auto_save = True
config.session_directory = "my_debates"
config.max_sessions_per_user = 50
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your OpenAI API key is set in `.env`
   - Check API key format and validity
   - Verify account has sufficient credits

2. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)
   - Verify virtual environment activation

3. **Ollama Connection Issues**
   - Ensure Ollama is running: `ollama serve`
   - Check model availability: `ollama list`
   - Verify base URL in configuration

4. **Session Loading Problems**
   - Check file permissions for session directory
   - Verify JSON file integrity
   - Ensure proper file encoding (UTF-8)

### Performance Optimization

1. **For Large Debates**
   - Reduce max tokens in settings
   - Disable real-time analytics
   - Use performance preset

2. **For Local Models**
   - Use Ollama with appropriate model size
   - Adjust creativity settings
   - Monitor system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Powered by LangChain for AI interactions
- Enhanced with Plotly for visualizations
- Inspired by formal debate methodologies

## 📞 Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration documentation

---

**Happy Debating! ⚔️**

