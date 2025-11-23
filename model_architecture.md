# BI-LSTM Model Architecture

## Overview
Binary classification model to distinguish between **arc** (curved) and **line** (straight) trajectory segments using Bidirectional LSTM networks.

---

## Input/Output

**Input**: Sequence of 150 timesteps × 13 features  
**Output**: Binary prediction (0 = arc, 1 = line)

**Features (per timestep)**:
- Normalized x, y coordinates
- Velocity (dx, dy)
- Speed, acceleration
- Distance from centroid
- Angle, curvature
- Cumulative angle change
- Local curvature variability
- Local radius consistency

---

## Architecture

```
Input (150, 13)
    ↓
Masking Layer (handles variable-length sequences)
    ↓
Bidirectional LSTM (128 units) → 256 features
    ↓
Dropout (40%)
    ↓
Bidirectional LSTM (64 units) → 128 features
    ↓
Dropout (40%)
    ↓
Dense (64, ReLU)
    ↓
Dropout (20%)
    ↓
Dense (32, ReLU)
    ↓
Output (1, Sigmoid) → [0, 1]
```

---

## Layer Details

| Layer | Output Shape | Parameters | Purpose |
|-------|-------------|------------|---------|
| Input | (None, 150, 13) | 0 | Entry point |
| Masking | (None, 150, 13) | 0 | Ignore padding |
| Bi-LSTM 1 | (None, 150, 256) | 145,408 | Learn temporal patterns (forward + backward) |
| Dropout | (None, 150, 256) | 0 | Regularization |
| Bi-LSTM 2 | (None, 128) | 164,352 | Aggregate sequence summary |
| Dropout | (None, 128) | 0 | Regularization |
| Dense 1 | (None, 64) | 8,256 | Feature compression |
| Dropout | (None, 64) | 0 | Regularization |
| Dense 2 | (None, 32) | 2,080 | Final feature refinement |
| Output | (None, 1) | 33 | Binary classification |

**Total Parameters**: 320,129 (1.22 MB)

---

## Hyperparameters

```python
max_sequence_length = 150
lstm_units = 128
dropout_rate = 0.4
dense_units = 64
learning_rate = 0.001
l2_regularization = 0.0001
batch_size = 64
patience = 5  # Early stopping
```

---

## Training Configuration

- **Loss**: Binary Crossentropy
- **Optimizer**: Adam (lr=0.001)
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Callbacks**:
  - Early Stopping (patience=15, monitor=val_loss)
  - Model Checkpoint (save best weights)
  - ReduceLROnPlateau (factor=0.5, patience=7)
  - TensorBoard logging

---

## Data Preprocessing

1. **Coordinate Normalization**: Center at origin, scale to unit size (translation/scale invariance)
2. **Feature Engineering**: 13 features per timestep computed from normalized coordinates
3. **Sequence Extraction**: Continuous segments split by label transitions
4. **Filtering**: Remove Label 2 (transition points), keep only Label 0 (arc) and 1 (line)
5. **Padding/Truncation**: All sequences normalized to 150 timesteps
6. **StandardScaler**: Applied to engineered features (fit on train only)

---

## Key Design Choices

- **Bidirectional LSTM**: Captures context from both past and future timesteps
- **Two-layer LSTM**: First layer learns low-level patterns, second aggregates high-level summary
- **Masking**: Handles variable-length sequences efficiently
- **Dropout**: Prevents overfitting (40% in LSTM, 20% in dense)
- **L2 Regularization**: Minimal (0.0001) to avoid over-penalizing
- **Sequence Length**: 150 timesteps balances information retention and training speed

