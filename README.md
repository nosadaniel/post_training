# Supervised Fine-Tuning (SFT) for Language Models

This project demonstrates how to perform supervised fine-tuning on language models using the Hugging Face Transformers library and TRL (Transformer Reinforcement Learning). The project specifically works with Qwen models and shows the before/after comparison of model performance.

## 🚀 Features

- **Model Comparison**: Compare base models vs. fine-tuned models
- **GPU Support**: Full CUDA support for accelerated training
- **Memory Monitoring**: Real-time GPU/CPU memory usage tracking
- **Interactive Notebook**: Jupyter notebook with step-by-step implementation
- **Custom Dataset Support**: Works with Hugging Face datasets

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- At least 6GB GPU memory for training (or sufficient RAM for CPU training)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd supervised_fine_tuning
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

### Quick Start

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook basic_sft.ipynb
   ```

2. **Run the cells sequentially** to:
   - Load and test base models
   - Compare with pre-fine-tuned models
   - Perform supervised fine-tuning
   - Evaluate training results

### Key Components

#### Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Load base model
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B-Base", use_gpu=True)
```

#### Training Configuration
```python
sft_config = SFTConfig(
    learning_rate=8e-5,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=2
)
```

#### Training Process
```python
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer_stats = sft_trainer.train()
```

## 📊 Supported Models

- **Base Models**: Qwen/Qwen3-0.6B-Base
- **Pre-fine-tuned**: banghua/Qwen3-0.6B
- **Custom Models**: Any compatible Hugging Face model

## 🗃️ Dataset

The project uses the `banghua/DL-SFT-Dataset` for training, which contains conversation pairs suitable for supervised fine-tuning.

## 🔧 Configuration

### Training Parameters
- **Learning Rate**: 8e-5 (default)
- **Epochs**: 1 (configurable)
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps
- **Logging**: Every 2 steps

### Memory Management
- **GPU Memory**: Automatic monitoring and reporting
- **CPU Fallback**: Automatic fallback when GPU unavailable
- **Gradient Checkpointing**: Configurable for memory optimization

## 📈 Performance Monitoring

The notebook includes comprehensive monitoring:
- GPU memory usage
- Training time tracking
- Memory efficiency metrics
- Performance comparisons

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Model Loading Errors**: Ensure sufficient disk space and stable internet connection
3. **Dependency Conflicts**: Use the provided virtual environment

### Memory Optimization
- Use smaller models for limited GPU memory
- Enable gradient checkpointing
- Reduce batch size and increase gradient accumulation steps

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [TRL Library Documentation](https://huggingface.co/docs/trl)
- [Qwen Model Documentation](https://huggingface.co/Qwen)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📄 License

This project is open source. Please check individual dependency licenses for specific terms.

## 🔍 Project Structure

```
supervised_fine_tuning/
├── README.md                 # This file
├── basic_sft.ipynb          # Main training notebook
├── requirements.txt          # Python dependencies
├── trainer_output/           # Training outputs and checkpoints
└── .venv/                   # Virtual environment
```

---

**Note**: This project is designed for educational and research purposes. Always ensure you have the necessary permissions and licenses when fine-tuning models with your own data.
