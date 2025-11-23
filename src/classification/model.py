"""
BI-LSTM model architecture for trajectory classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


class BiLSTMClassifier:
    """Bidirectional LSTM model for binary classification of trajectories."""
    
    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_units: int = 128,
        dropout_rate: float = 0.4,
        dense_units: int = 64,
        learning_rate: float = 0.001,
        l2_regularization: float = 0.01
    ):
        """
        Initialize BI-LSTM classifier.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layer
            learning_rate: Learning rate for Adam optimizer
            l2_regularization: L2 regularization factor
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the BI-LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='trajectory_input'
        )
        
        # Masking layer (to handle padded sequences)
        x = layers.Masking(mask_value=0.0)(inputs)
        
        # First Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                recurrent_regularizer=keras.regularizers.l2(self.l2_regularization)
            ),
            name='bi_lstm_1'
        )(x)
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                recurrent_regularizer=keras.regularizers.l2(self.l2_regularization)
            ),
            name='bi_lstm_2'
        )(x)
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
            name='dense_1'
        )(x)
        
        x = layers.Dropout(self.dropout_rate / 2)(x)
        
        x = layers.Dense(
            self.dense_units // 2,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
            name='dense_2'
        )(x)
        
        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Classifier')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ]
        )
        
        self.model = model
        return model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def get_callbacks(
        self,
        checkpoint_path: str,
        patience: int = 15,
        min_delta: float = 0.001
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            checkpoint_path: Path to save best model
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=min_delta
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(
        self,
        X_train: tf.Tensor,
        y_train: tf.Tensor,
        X_val: tf.Tensor,
        y_val: tf.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        checkpoint_path: str = 'models/best_model.keras',
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Get callbacks
        callbacks = self.get_callbacks(checkpoint_path)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: tf.Tensor, threshold: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        probabilities = self.model.predict(X)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def evaluate(self, X: tf.Tensor, y: tf.Tensor) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Test sequences
            y: Test labels
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        results = self.model.evaluate(X, y, return_dict=True)
        return results
    
    def save(self, filepath: str):
        """Save model."""
        if self.model is None:
            raise ValueError("Model not built.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'learning_rate': self.learning_rate,
            'l2_regularization': self.l2_regularization
        }


def create_simple_baseline(sequence_length: int, n_features: int) -> Model:
    """
    Create a simple baseline model for comparison.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features
        
    Returns:
        Compiled baseline model
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name='Baseline_LSTM')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    return model


if __name__ == '__main__':
    # Example: Create and display model architecture
    print("Creating BI-LSTM Classifier...")
    classifier = BiLSTMClassifier(
        sequence_length=300,
        n_features=10,  # Will depend on feature engineering
        lstm_units=128,
        dropout_rate=0.4,
        dense_units=64
    )
    
    classifier.build_model()
    classifier.summary()
    
    print("\n" + "="*60)
    print("Model architecture created successfully!")
    print("="*60)

