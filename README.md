# Titans: Revolutionizing Memory in Deep Learning

## Overview
This repository contains the implementation of the Titans architecture, a next-generation framework for scalable sequence modeling introduced in the paper **"Titans: Learning to Memorize at Test Time"**. Titans redefine memory management in deep learning, seamlessly integrating short-term and long-term memory modules to handle large context windows efficiently and effectively.

### Key Features:
- **Memory as Context (MAC):** Combines input sequences with long-term and persistent memory, using attention mechanisms to dynamically decide the relevance of historical data.
- **Memory as Gate (MAG):** Employs sliding-window attention for short-term memory and a gating mechanism to blend long-term context effectively.
- **Memory as Layer (MAL):** Treats the memory module as an independent layer, compressing past and current information before attention mechanisms.

### Visualization:
![Titan Model Visualization](titan_model_torchviz1.png)


## Code Structure
### Architecture Modules
- **`PersistentMemory`**: Provides static task-specific knowledge.
- **`LongTermMemory`**: Encodes historical patterns for effective retrieval.
- **`SlidingWindowAttention`**: Processes short-term memory with a focus on recent context.
- **MAC/MAG/MAL Implementations**: Three architectural variants tailored for different sequence modeling tasks.

### Main Files
- `titans_memory_architectures.py`: Core implementation of the Titans architecture, including MAC, MAG, and MAL variants.
- `train.py`: Script for training the Titans model.
- `evaluate.py`: Script for evaluating the model on specific datasets.
- `datasets.py`: Preprocessing and loading scripts for various datasets.

### Example Usage
```python
# Import the MAC, MAG, and MAL architectures
from titans_memory_architectures import MemoryAsContext, MemoryAsGate, MemoryAsLayer

# Initialize models
mac = MemoryAsContext(feature_dim=128, memory_size=10)
mag = MemoryAsGate(feature_dim=128)
mal = MemoryAsLayer(feature_dim=128)

# Input data
inputs = torch.randn(8, 32, 128)  # Batch size: 8, Sequence length: 32, Feature dimension: 128

# Forward pass
output_mac = mac(inputs)
output_mag = mag(inputs)
output_mal = mal(inputs)
```

## Installation
Clone this repository:
```bash
git clone https://github.com/yourusername/titans-memory.git
cd titans-memory
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Datasets
### Supported Datasets
- **WikiText-103**: For language modeling.
- **PIQA, HellaSwag**: For commonsense reasoning.
- **ETTh/ETTm**: For time-series forecasting.

### Preprocessing
Use the `datasets.py` script to preprocess your dataset. Example:
```bash
python datasets.py --dataset wikitext --output_dir ./processed_data
```

## Training
Train the Titans model using `train.py`:
```bash
python train.py --model mac --dataset ./processed_data --epochs 10 --batch_size 16
```

## Evaluation
Evaluate the model using `evaluate.py`:
```bash
python evaluate.py --model_path ./checkpoints/best_model.pt --dataset ./processed_data
```

## Experimental Results
- **Language Modeling:** Achieved state-of-the-art perplexity on WikiText-103.
- **Commonsense Reasoning:** Outperformed GPT-4 and Llama 3.1 on PIQA and HellaSwag.
- **Time-Series Forecasting:** Showcased exceptional ability to model long-term dependencies.


## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This repository is licensed under the MIT License.

