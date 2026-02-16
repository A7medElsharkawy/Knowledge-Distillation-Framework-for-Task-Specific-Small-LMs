# news-slm-distillation

Distillation pipeline for news-structured language modeling: a **teacher model** generates structured JSON labels from raw news, and a **student Qwen LoRA model** is fine-tuned on this synthetic data, then served via vLLM.

## Project Structure

```
news-slm-distillation/
│
├── .env                         # API keys (OpenAI, WandB)
├── .gitignore
├── requirements.txt
├── README.md
├── setup.py
│
├── config/                      # Training & model configs
│   ├── qwen_lora_sft.yaml       # Llama Factory LoRA config
│   └── generation_config.yaml   # Teacher prompt & schema settings
│
├── data/
│   ├── raw/                     # Raw news articles
│   │   └── sample_news.jsonl
│   ├── synthetic/               # Teacher-generated labeled data
│   │   └── distillation_data.jsonl
│   ├── processed/               # Final train/val splits
│   │   ├── train.json
│   │   └── val.json
│   └── dataset_info.json        # Llama Factory dataset registry
│
├── src/
│   ├── teacher/                 # Knowledge distillation (Teacher model)
│   │   ├── prompt_template.py
│   │   ├── schema.py            # Pydantic JSON schema
│   │   └── generate_synthetic.py
│   │
│   ├── student/                 # Student model fine-tuning
│   │   ├── prepare_dataset.py
│   │   ├── train_lora.sh
│   │   └── merge_adapter.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_json.py     # JSON schema validation accuracy
│   │   ├── evaluate_metrics.py  # BLEU / ROUGE / F1
│   │   └── compare_teacher.py   # Compare teacher vs student outputs
│   │
│   ├── inference/
│   │   └── inference.py         # Local testing of fine-tuned model
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py
│       └── io.py
│
├── models/                      # Saved LoRA adapters & merged models
│   ├── base/                    # Original Qwen model (optional ref)
│   └── lora_adapters/
│       ├── extraction_adapter/
│       └── translation_adapter/
│
├── serving/                     # Deployment layer
│   ├── serve_vllm.sh            # Launch vLLM server
│   ├── api_router.py            # Route requests to correct LoRA adapter
│   └── request_example.json
│
├── testing/                     # Stress & load testing
│   ├── locustfile.py
│   └── performance_report.md
│
├── experiments/                 # Reproducibility tracking
│   ├── exp_01_qwen_lora/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   └── notes.md
│   └── wandb_logs/
│
├── notebooks/                   # Optional analysis
│   ├── data_exploration.ipynb
│   └── evaluation_analysis.ipynb
│
└── docs/                        # Architecture & diagrams
    ├── system_design.md
    └── architecture.png

```
