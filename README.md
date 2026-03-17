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

# Result Comparison

## Extraction Task: Base Model(Qwen2.5-1.5B) vs GPT-4.1-mini


| Base Model (Qwen2.5-1.5B)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | GPT-4.1-mini                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Title:** How Family Influences Financial Behavior **Keywords:** family influence, financial behavior, moneymaking, money management, inheritance **Summary:** - Financial behavior is shaped by family dynamics. - Family plays a crucial role in determining one's financial personality. - Three main aspects of financial relationships are discussed: acquisition, use, and management. **Category:** economy **Entities:** Forbes Magazine (organization), Shain Enit (person-female), Financial Therapy Association (organization) | **Title:** دور العائلة في تشكيل العلاقة بالمال وأنماط الشخصية المالية **Keywords:** العلاقة بالمال, العائلة, الشخصية المالية, مخطط الجينوم المالي, أنماط السلوك المالي **Summary:** - تلعب العائلة دورا محوريا في تشكيل علاقة الأفراد بالمال من خلال أنماط السلوك المالي المتوارثة. - تحدد 'الشخصية المالية' لكل فرد بناء على تفاعله مع المال وتأثير تربية الأسرة وتجارب الطفولة. - هناك ثلاثة أبعاد رئيسية للعلاقة بالمال: الاكتساب، الاستخدام، والإدارة، لكل منها خصائص وسلوكيات مميزة. - تؤثر التجارب الأسرية بشكل مباشر على تبني الفرد لأنماط مالية صحية أو غير صحية. - طورت رابطة العلاج المالي أداة مخطط الجينوم المالي لتحليل الأنماط المالية داخل العائلة عبر رسم شجرة عائلية وتصنيف الأفراد. **Category:** economy **Entities:** مجلة فوربس (organization), شاين إنيت (person-male), رابطة العلاج المالي (organization), مخطط الجينوم المالي (artifact) |


## Translate Task: Base Model (Qwen2.5-1.5B) vs GPT-4.1-mini

| GPT-4.1-mini  | Base Model(Qwen2.5-1.5B) |
|-----------------------|-------------------------|
| **Title:** The Family's Role in Shaping Our Relationship with Money<br>**Content:** Forbes magazine mentioned that the family plays a pivotal role in shaping individuals' relationship with money, as this relationship is influenced by financial behavior patterns inherited across generations.<br><br>The report, based on research by university professor Shane Ennett on financial well-being, explains that each person has a "financial personality" determined by how they interact with money, which is directly influenced by family upbringing and childhood experiences.<br><br>Three Dimensions of the Relationship with Money:<br>- **Acquisition (A):** Viewing money as a collectible, risk of obsession or rejection.<br>- **Usage (U):** Seeing money as a tool for enjoyment, may lead to overspending or extreme frugality.<br>- **Management (M):** Treating money as a responsibility, could lead to excessive planning affecting relationships.<br><br>Family experiences shape "financial personality," e.g., parental behavior patterns.<br><br>The Financial Therapy Association developed the Money Genogram to analyze family financial patterns.<br>- Draw family tree<br>- Classify members by A, U, M<br>- Determine healthy (+) or unhealthy (-) financial behaviors | **Title:** Forbes Magazine: Family Plays a Central Role in Forming Individuals' Financial Relationships<br>**Content:** According to Forbes magazine, family plays a crucial role in shaping individuals' financial relationships, as these relationships are influenced by inherited behavioral patterns across generations. |

- Base model sometimes returns English responses instead of Arabic (as requested)
- Output completeness and accuracy need improvement through fine-tuning

## 🏗️ Architecture

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
│   │
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
conda create -n news-slm python=3.12
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

Create/update `example.env` file in `src/` directory:

```bash
# Required
HUGGINGFACE_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_key_here

# Optional (for teacher model)
OPENAI_API_KEY=your_openai_key_here
```
after that chnage the name of env file
```bash
mv example.env env
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

python -m src.run test-base-model --task extraction --runner local

python -m src.run test-base-model --task translation --runner local
```

### Evaluate Teacher Model(OpenAI)

```bash
cd src

python -m src.run test-base-model --task extraction --runner openai

python -m src.run test-base-model --task translation --runner openai
```
### Load Example Story

```python
from controllers import DataController

dc = DataController()
story = dc.load_example_story()
print(story)
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
