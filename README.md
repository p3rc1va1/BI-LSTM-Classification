# BI-LSTM Classification
## Main goal:
Develop an ML model with BI-LSTM algorithm that classifies each coordinate group into "line" or "arc"

## Topic:
AI-based classification of geometric trajectory segments (arc vs line) from sequential 2D coordinates

## Tasks:
- Visualization and explanation of Artificial Intelligence decision-making process​
- Initial data collection and pre-processing​
- Prototyping and rapid development of Artificial Intelligence systems​
- Artificial intelligence algorithm implementation and model training​
- Hyperparameter tuning and optimization​
- Development and execution of a comprehensive testing plan for the Artificial Intelligence prototype​
- Security and privacy assessment of Artificial Intelligence systems​
Deployment and scalability of Artificial Intelligence systems​

## Contribution:
1. Create a feature branch from main: git checkout -b feature/feature-name.
2. Crate a PR and will be merged after review.

---

## Setup & Installation

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone Repository
```bash
git clone https://github.com/p3rc1va1/BI-LSTM-Classification.git
cd BI-LSTM-Classification
```

### 2. Install Dependencies
Using `uv` (recommended):
```bash
uv sync
```

This will:
- Create a virtual environment (`.venv`)
- Install all dependencies from `pyproject.toml`
- Set up the project for development

### 3. Verify Installation
```bash
uv run python --version
uv run python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
```

---

## Running the Model

### Training from Scratch

Navigate to the training directory and run:
```bash
cd src/classification
uv run python train.py
```

**What happens:**
1. **Preprocessing** (~30s): 
   - Loads `data/combined_data.csv`
   - Extracts continuous segments (filters transitions)
   - Normalizes coordinates (scale/translation invariance)
   - Engineers 13 features per timestep
   - Splits into train/val/test (70/15/15)
   - Saves processed data to `data/processed/*.npy`

2. **Model Building** (~1s):
   - Creates BI-LSTM architecture (320k parameters)
   - Compiles with Adam optimizer

3. **Training** (~2.5 min/epoch):
   - Trains for up to 20 epochs
   - Early stopping (patience=5) on validation loss
   - Saves best model to `models/bilstm_classifier_YYYYMMDD_HHMMSS.keras`

4. **Evaluation**:
   - Tests on held-out test set
   - Saves results to `models/test_results_*.json`

### Configuration

Edit `src/classification/train.py` to adjust hyperparameters:
```python
config = {
    'max_sequence_length': 150,  # Sequence length
    'lstm_units': 128,            # LSTM hidden units
    'dropout_rate': 0.4,          # Dropout for regularization
    'learning_rate': 0.001,       # Adam learning rate
    'batch_size': 64,             # Training batch size
    'epochs': 20,                 # Maximum epochs
    'patience': 5,                # Early stopping patience
}
```

### Using Existing Preprocessed Data

If you've already preprocessed the data:
```python
config = {
    'use_existing_data': True,  # Skip preprocessing
    # ... other settings
}
```

---

## Project Structure

```
BI-LSTM-Classification/
├── data/
│   └── combined_data.csv          # Source dataset (1.2M coordinates)
├── models/
│   ├── preprocessor.pkl           # Fitted preprocessor
│   └── bilstm_classifier_*.keras  # Trained models (generated)
├── src/
│   ├── EDA/
│   │   └── feature_engineering.ipynb  # Feature analysis
│   └── classification/
│       ├── preprocessing.py       # Data preprocessing
│       ├── model.py              # BI-LSTM architecture
│       ├── train.py              # Training pipeline
│       ├── evaluate.py           # Evaluation metrics
│       ├── inference.py          # Inference on new data
│       └── visualization.py      # Plotting utilities
├── model_architecture.md         # Detailed model documentation
└── README.md                     # This file
```

---

## Model Architecture

For detailed architecture information, see [model_architecture.md](model_architecture.md)

**Quick Summary:**
- **Input**: 150 timesteps × 13 features
- **Model**: 2-layer Bidirectional LSTM + Dense layers
- **Output**: Binary classification (0=arc, 1=line)
- **Parameters**: 320,129
- **Performance**: ~98% accuracy, 0.998 AUC

---

## Expected Results

After training completes, you'll see:
- **Training accuracy**: ~98%
- **Validation AUC**: ~0.998
- **Test metrics**: Saved to `models/test_results_*.json`

**Output files:**
```
models/
├── bilstm_classifier_20251123_130140.keras  # Best model weights
├── training_history_20251123_130140.json    # Training curves
├── test_results_20251123_130140.json        # Final metrics
└── model_config.json                        # Hyperparameters
```

---

## Troubleshooting

### Out of Memory
Reduce batch size or sequence length:
```python
'batch_size': 32,           # Reduce from 64
'max_sequence_length': 100  # Reduce from 150
```

### Slow Training
Increase batch size or reduce sequence length:
```python
'batch_size': 128,          # Increase from 64
'max_sequence_length': 100  # Reduce from 150
```

### Poor Accuracy
Try adjusting:
- L2 regularization (increase/decrease)
- Dropout rate (tune for overfitting/underfitting)
- Learning rate (reduce if loss is unstable)

---

## Dependencies

Main packages (see `pyproject.toml` for full list):
- TensorFlow >= 2.13.0
- NumPy >= 2.3.4
- Pandas >= 2.3.3
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.10.7
- Seaborn >= 0.13.2