# News SLM Distillation

> ⚠️ **Project Status**: This project is **still in development** and not yet finished. Many features are incomplete or in progress.

Distillation pipeline for news-structured language modeling: a **teacher model** generates structured JSON labels from Arabic news articles, and a **student Qwen LoRA model** is fine-tuned on this synthetic data, then served via vLLM.

## 🎯 Project Goal

Extract structured information from Arabic news articles using a Pydantic schema, generating JSON outputs with:
- Story title, keywords, summary
- Story category (politics, sports, art, technology, economy, health, entertainment, science)
- Named entities (persons, locations, organizations, events, etc.)

## 📊 Current Progress

### ✅ Completed
- **Controller Architecture**: Implemented BaseController, DataController, and ModelController for modular code organization
- **Pydantic Schema**: Defined `NewsDetails` schema with validation for structured data extraction
- **Model Management**: Created ModelEnum for centralized model ID management
- **Prompt Templates**: Built prompt generation system for teacher/student/base models
- **Base Model Evaluation**: Implemented evaluation pipeline for testing base model extraction capabilities
- **Data Loading**: DataController handles loading example stories from `data/raw/`

### 🚧 In Progress
- Fine-tuning student model with LoRA adapters
- Teacher model integration (OpenAI API)
- Dataset preparation and processing

### 📝 Known Issues
- Base model sometimes returns English responses instead of Arabic (as requested)
- Output completeness and accuracy need improvement through fine-tuning

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│  Arabic News    │
│  Article (Raw)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Teacher Model (OpenAI GPT-4)    │
│  Generates structured JSON labels    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Synthetic Training Data (JSON)     │
│   Validated with NewsDetails schema  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Student Model (Qwen + LoRA)       │
│   Fine-tuned on synthetic data       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Structured JSON Output             │
│   (Title, Keywords, Summary,        │
│    Category, Entities)               │
└─────────────────────────────────────┘
```

### Component Architecture

#### 1. **Controllers** (`src/controllers/`)
- **BaseController**: Base class providing common functionality (settings, base directory paths)
- **DataController**: Handles data file operations (loading example stories, managing data directories)
- **ModelController**: Manages model and tokenizer loading, chat template application, and model inference

#### 2. **Models** (`src/models/`)
- **schemes/instruction.py**: Pydantic schema (`NewsDetails`, `Entity`) for structured data validation
- **enums/ModelEnum.py**: Enum for model ID management (BASE_MODEL_QWEN, etc.)

#### 3. **Utils** (`src/utils/`)
- **prompt_template.py**: Generates extraction prompts with system/user messages for teacher/student/base models

#### 4. **Evaluation** (`src/evaluation/`)
- **eval_base_local.py**: Evaluates base model performance on example stories

#### 5. **Helper** (`src/helper/`)
- **config.py**: Pydantic settings for environment variables (API keys, tokens)

### Data Flow

1. **Input**: Arabic news article (text file in `data/raw/`)
2. **Prompt Generation**: `create_details_extraction_prompt()` builds messages with:
   - System message: Instructions for NLP data parsing
   - User message: Story text + Pydantic schema JSON
3. **Model Processing**: 
   - Base/Student model processes prompt
   - Generates structured JSON response
4. **Validation**: Response validated against `NewsDetails` Pydantic schema
5. **Output**: Structured JSON with story details

## 📁 Project Structure

```
lora-finetuning/
│
├── .env                         # Environment variables (API keys)
├── .gitignore
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # Raw news articles
│   │   ├── example.txt          # Example Arabic story for testing
│   │   └── sample_news.jsonl
│   ├── synthetic/               # Teacher-generated labeled data
│   │   └── distillation_data.jsonl
│   └── processed/               # Final train/val splits
│       ├── train.json
│       └── val.json
│
├── src/
│   ├── controllers/             # Controller classes
│   │   ├── BaseController.py    # Base controller with common functionality
│   │   ├── DataController.py    # Data file operations
│   │   └── ModelController.py   # Model/tokenizer management
│   │
│   ├── models/                  # Data models and enums
│   │   ├── shcemes/
│   │   │   └── instruction.py  # Pydantic schemas (NewsDetails, Entity)
│   │   └── enums/
│   │       └── ModelEnum.py     # Model ID enums
│   │
│   ├── utils/                   # Utility functions
│   │   └── prompt_template.py   # Prompt generation for extraction
│   │
│   ├── evaluation/              # Evaluation scripts
│   │   └── eval_base_local.py   # Base model evaluation
│   │
│   ├── helper/                  # Helper modules
│   │   └── config.py            # Settings and configuration
│   │
│   ├── inference/               # Inference scripts
│   │   └── inference.py         # Model inference utilities
│   │
│   ├── teacher/                 # Teacher model (OpenAI)
│   │   └── (to be implemented)
│   │
│   ├── student/                 # Student model fine-tuning
│   │   └── (to be implemented)
│   │
│   └── test.py                  # Test script for evaluation
│
└── models/                       # Saved models and adapters
    └── (to be created during training)
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- Conda or virtual environment
- Hugging Face account (for model access)
- OpenAI API key (for teacher model)
- WandB account (optional, for experiment tracking)

### Step 1: Clone and Navigate

```bash
cd "/Users/shark/Desktop/lora finetuning"
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n news-slm python=3.10
conda activate news-slm

# Or using venv
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### Step 3: Install Dependencies

```bash
cd src
pip install -r requirements.txt

# Install accelerate for model loading (required)
pip install "accelerate>=0.26.0"
```

### Step 4: Configure Environment Variables

Create/update `.env` file in `src/` directory:

```bash
# Required
HUGGINGFACE_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here

# Optional (for teacher model)
OPENAI_API_KEY=your_openai_key_here
```

**Note**: Remove any spaces after `=` in the `.env` file.

### Step 5: Verify Installation

Test the base model evaluation:

```bash
cd src
PYTHONPATH=. python test.py
```

This will:
1. Load the base model (Qwen2.5-1.5B-Instruct)
2. Load example story from `data/raw/example.txt`
3. Generate extraction prompt
4. Run model inference
5. Return structured JSON output

## 📖 Usage Examples

### Evaluate Base Model

```bash
cd src
PYTHONPATH=. python test.py
```

Or run the evaluation function directly:

```python
from evaluation import eval_base_model

response = eval_base_model()
print(response)
```

### Load Example Story

```python
from controllers import DataController

dc = DataController()
story = dc.load_example_story()
print(story)
```

### Generate Extraction Prompt

```python
from utils.prompt_template import create_details_extraction_prompt
from models.shcemes import NewsDetails

messages = create_details_extraction_prompt(NewsDetails)
print(messages)
```

### Load Model and Generate Response

```python
from controllers import ModelController
from models.enums import ModelEnum

mc = ModelController()
model, tokenizer = mc.load_model_and_tokenizer(ModelEnum.BASE_MODEL_QWEN.value)

# Apply chat template
messages = [{"role": "user", "content": "Your Arabic story here..."}]
prompt = mc.apply_chat_templete(messages, tokenizer)

# Generate response
response = mc.model_output(prompt, tokenizer, model)
print(response)
```

## 🔧 Configuration

### Model Selection

Edit `src/models/enums/ModelEnum.py` to change the base model:

```python
class ModelEnum(Enum):
    BASE_MODEL_QWEN = "Qwen/Qwen2.5-1.5B-Instruct"  # Change model ID here
```

### Pydantic Schema

Modify `src/models/shcemes/instruction.py` to adjust the extraction schema:

- `StoryCategory`: Add/remove categories
- `EntityType`: Add/remove entity types
- `NewsDetails`: Modify fields and validation rules

## 🐛 Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure you're running from the correct directory:

```bash
cd src
PYTHONPATH=. python your_script.py
```

### Model Loading Errors

- **Accelerate required**: Install with `pip install "accelerate>=0.26.0"`
- **Memory issues**: Use smaller models or enable quantization
- **Hugging Face token**: Ensure `HUGGINGFACE_TOKEN` is set in `.env`

### Environment Variables

- Check `.env` file exists in `src/` directory
- Ensure no spaces after `=` in `.env` file
- Verify all required keys are present

## 📝 Next Steps (Remaining Work)

**Note**: The following features are **not yet implemented** and represent the remaining work to complete this project:

1. **Teacher Model Integration**: Connect OpenAI API for synthetic data generation
2. **Dataset Preparation**: Process and format training data
3. **LoRA Fine-tuning**: Train student model with LoRA adapters
4. **Evaluation Pipeline**: Compare teacher vs student vs base model performance
5. **Deployment**: Set up vLLM serving infrastructure

The project foundation (controllers, schemas, prompt templates, base evaluation) is in place, but the full pipeline is not yet operational.

## 📚 Dependencies

Key packages:
- `transformers`: Model loading and inference
- `pydantic-settings`: Configuration management
- `openai`: Teacher model API
- `accelerate`: Efficient model loading
- `datasets`: Data processing
- `wandb`: Experiment tracking (optional)

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a research project for Arabic news extraction. Contributions welcome!

## 📄 License

[Add your license here]
