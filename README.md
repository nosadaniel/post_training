# Post-Training Techniques for Language Models

This project demonstrates advanced post-training techniques for language models including **Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Online Reinforcement Learning (RL)**. Built with the Hugging Face Transformers library and TRL (Transformer Reinforcement Learning), the project works with Qwen models and provides comprehensive before/after comparisons of model performance.

## 🚀 Features

- **Multiple Training Approaches**: SFT, DPO, and Online RL training methods
- **Model Comparison**: Compare base models vs. fine-tuned models across different techniques
- **GPU Support**: Full CUDA support for accelerated training
- **Memory Monitoring**: Real-time GPU/CPU memory usage tracking
- **Interactive Notebooks**: Jupyter notebooks with step-by-step implementation
- **Custom Dataset Support**: Works with Hugging Face datasets
- **Performance Evaluation**: Comprehensive metrics and comparison tools

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- At least 6GB GPU memory for training (or sufficient RAM for CPU training)
- Jupyter notebook environment

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd post_training
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training Approaches

#### 1. Supervised Fine-Tuning (SFT)
**File**: `basic_sft.ipynb`

Supervised fine-tuning for instruction-following capabilities:
```bash
jupyter notebook basic_sft.ipynb
```

#### 2. Direct Preference Optimization (DPO)
**File**: `basic_dpo.ipynb`

DPO training for preference alignment:
```bash
jupyter notebook basic_dpo.ipynb
```

#### 3. Online Reinforcement Learning
**File**: `online_rl.ipynb`

Online RL training for continuous improvement:
```bash
jupyter notebook online_rl.ipynb
```

### Quick Start

1. **Choose your training approach** (SFT, DPO, or Online RL)
2. **Open the corresponding notebook**
3. **Run the cells sequentially** to:
   - Load and test base models
   - Compare with pre-fine-tuned models
   - Perform training with your chosen method
   - Evaluate training results and model performance

### Key Components

#### Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Load base model
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", use_gpu=True)
```

#### Training Configuration Examples

**SFT Training:**
```python
sft_config = SFTConfig(
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=2
)
```

**DPO Training:**
```python
dpo_config = DPOConfig(
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=2
)
```

## 📊 Supported Models

- **Base Models**: Qwen/Qwen3-0.6B-Base
- **Pre-fine-tuned**: banghua/Qwen3-0.6B
- **Custom Models**: Any compatible Hugging Face model
- **Training Outputs**: Checkpoints saved in `trainer_output/` directory

## 🗃️ Datasets

- **SFT Dataset**: `banghua/DL-SFT-Dataset` for supervised fine-tuning
- **DPO Dataset**: Preference datasets for direct preference optimization
- **RL Dataset**: Dynamic datasets for online reinforcement learning

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 8e-5 (SFT), 5e-5 (DPO)
- **Epochs**: 1 (configurable)
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps
- **Logging**: Every 2 steps

### Memory Management
- **GPU Memory**: Automatic monitoring and reporting
- **CPU Fallback**: Automatic fallback when GPU unavailable
- **Gradient Checkpointing**: Configurable for memory optimization

## 📈 Performance Monitoring

Each notebook includes comprehensive monitoring:
- GPU memory usage
- Training time tracking
- Memory efficiency metrics
- Performance comparisons between methods
- Model output quality assessment

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Model Loading Errors**: Ensure sufficient disk space and stable internet connection
3. **Dependency Conflicts**: Use the provided virtual environment
4. **Training Convergence**: Adjust learning rates and batch sizes for better results

### Memory Optimization
- Use smaller models for limited GPU memory
- Enable gradient checkpointing
- Reduce batch size and increase gradient accumulation steps
- Monitor memory usage during training

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [TRL Library Documentation](https://huggingface.co/docs/trl)
- [Qwen Model Documentation](https://huggingface.co/Qwen)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [RLHF Documentation](https://huggingface.co/docs/trl/conceptual_guides/rlhf)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is open source. Please check individual dependency licenses for specific terms.

## 🔍 Project Structure

```
post_training/
├── README.md                 # This file
├── basic_sft.ipynb          # Supervised Fine-Tuning notebook
├── basic_dpo.ipynb          # Direct Preference Optimization notebook
├── online_rl.ipynb          # Online Reinforcement Learning notebook
├── requirements.txt          # Python dependencies
├── trainer_output/           # Training outputs and checkpoints (gitignored)
│   ├── checkpoint-4/        # Training checkpoints
│   ├── checkpoint-13/       # Training checkpoints
│   └── checkpoint-371/      # Training checkpoints
└── .venv/                   # Virtual environment
```

## 🎓 Learning Path

1. **Start with SFT** (`basic_sft.ipynb`) to understand basic fine-tuning
2. **Move to DPO** (`basic_dpo.ipynb`) for preference alignment
3. **Explore Online RL** (`online_rl.ipynb`) for advanced training techniques

---

**Note**: This project is designed for educational and research purposes. Always ensure you have the necessary permissions and licenses when fine-tuning models with your own data. The `trainer_output/` directory contains large model files and is automatically ignored by git.
