"""
Training pipeline for BI-LSTM trajectory classifier.
"""

import numpy as np
import json
import os
from datetime import datetime
from model import BiLSTMClassifier
from preprocessing import TrajectoryPreprocessor


class TrainingPipeline:
    """Complete training pipeline for trajectory classification."""
    
    def __init__(
        self,
        data_path: str,
        model_dir: str = '../../models',
        processed_data_dir: str = '../../data/processed'
    ):
        """
        Initialize training pipeline.
        
        Args:
            data_path: Path to combined_data.csv
            model_dir: Directory to save models
            processed_data_dir: Directory to save processed data
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)
        
        self.preprocessor = None
        self.classifier = None
        self.data = None
        
    def preprocess_data(self, max_sequence_length: int = 300):
        """
        Preprocess data and create sequences.
        
        Args:
            max_sequence_length: Maximum sequence length
        """
        print("="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)
        
        # Initialize preprocessor
        self.preprocessor = TrajectoryPreprocessor(max_sequence_length=max_sequence_length)
        
        # Load data
        df = self.preprocessor.load_data(self.data_path)
        print(f"Loaded {len(df)} data points from {df['shape_id'].nunique()} shapes")
        
        # Prepare data
        self.data = self.preprocessor.prepare_data(df)
        
        # Save processed data
        for key, value in self.data.items():
            filepath = os.path.join(self.processed_data_dir, f'{key}.npy')
            np.save(filepath, value)
            print(f"Saved {key} to {filepath}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
        print("\nPreprocessing complete!")
        
    def load_preprocessed_data(self):
        """Load preprocessed data from disk."""
        print("Loading preprocessed data...")
        self.data = {}
        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            filepath = os.path.join(self.processed_data_dir, f'{key}.npy')
            self.data[key] = np.load(filepath)
            print(f"Loaded {key}: {self.data[key].shape}")
    
    def build_model(
        self,
        lstm_units: int = 128,
        dropout_rate: float = 0.4,
        dense_units: int = 64,
        learning_rate: float = 0.001,
        l2_regularization: float = 0.01
    ):
        """
        Build the BI-LSTM model.
        
        Args:
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            dense_units: Dense layer units
            learning_rate: Learning rate
            l2_regularization: L2 regularization factor
        """
        print("\n" + "="*60)
        print("STEP 2: MODEL BUILDING")
        print("="*60)
        
        # Get dimensions from data
        _, seq_len, n_features = self.data['X_train'].shape
        
        # Create classifier
        self.classifier = BiLSTMClassifier(
            sequence_length=seq_len,
            n_features=n_features,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization
        )
        
        # Build model
        self.classifier.build_model()
        self.classifier.summary()
        
        print("\nModel built successfully!")
        
    def train_model(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        patience: int = 5
    ):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        # Validate labels before training
        print("\nValidating data...")
        print(f"  y_train unique values: {np.unique(self.data['y_train'])}")
        print(f"  y_train shape: {self.data['y_train'].shape}")
        print(f"  y_train dtype: {self.data['y_train'].dtype}")
        print(f"  X_train shape: {self.data['X_train'].shape}")
        print(f"  X_train range: [{self.data['X_train'].min():.3f}, {self.data['X_train'].max():.3f}]")
        
        # Check for NaN or Inf
        if np.any(np.isnan(self.data['X_train'])) or np.any(np.isinf(self.data['X_train'])):
            raise ValueError("X_train contains NaN or Inf values!")
        if np.any(np.isnan(self.data['y_train'])) or np.any(np.isinf(self.data['y_train'])):
            raise ValueError("y_train contains NaN or Inf values!")
        
        # Ensure labels are 0 and 1
        if not np.all(np.isin(self.data['y_train'], [0, 1])):
            raise ValueError(f"Labels must be 0 or 1, got: {np.unique(self.data['y_train'])}")
        
        # Checkpoint path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.model_dir,
            f'bilstm_classifier_{timestamp}.keras'
        )
        
        # Train
        history = self.classifier.train(
            X_train=self.data['X_train'],
            y_train=self.data['y_train'],
            X_val=self.data['X_val'],
            y_val=self.data['y_val'],
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path
        )
        
        # Save training history
        history_path = os.path.join(
            self.model_dir,
            f'training_history_{timestamp}.json'
        )
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Model saved to: {checkpoint_path}")
        print(f"History saved to: {history_path}")
        
        return history
    
    def evaluate_model(self):
        """Evaluate model on test set."""
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        # Evaluate
        results = self.classifier.evaluate(
            self.data['X_test'],
            self.data['y_test']
        )
        
        print("\nTest Set Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.model_dir,
            f'test_results_{timestamp}.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        
        return results
    
    def save_model_config(self):
        """Save model configuration."""
        config = self.classifier.get_config()
        config_path = os.path.join(self.model_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model config saved to: {config_path}")
    
    def run_full_pipeline(
        self,
        max_sequence_length: int = 300,
        lstm_units: int = 128,
        dropout_rate: float = 0.4,
        dense_units: int = 64,
        learning_rate: float = 0.001,
        l2_regularization: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        use_existing_data: bool = False
    ):
        """
        Run complete training pipeline.
        
        Args:
            max_sequence_length: Maximum sequence length
            lstm_units: LSTM units
            dropout_rate: Dropout rate
            dense_units: Dense layer units
            learning_rate: Learning rate
            l2_regularization: L2 regularization
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            use_existing_data: Whether to load existing preprocessed data
        """
        print("\n" + "="*70)
        print(" "*15 + "BI-LSTM TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Preprocess or load data
        if use_existing_data and os.path.exists(
            os.path.join(self.processed_data_dir, 'X_train.npy')
        ):
            self.load_preprocessed_data()
            # Load preprocessor
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
            self.preprocessor = TrajectoryPreprocessor.load(preprocessor_path)
        else:
            self.preprocess_data(max_sequence_length)
        
        # Step 2: Build model
        self.build_model(
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            learning_rate=learning_rate,
            l2_regularization=l2_regularization
        )
        
        # Step 3: Train model
        history = self.train_model(
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # Step 4: Evaluate model
        results = self.evaluate_model()
        
        # Step 5: Save configuration
        self.save_model_config()
        
        print("\n" + "="*70)
        print(" "*20 + "PIPELINE COMPLETE!")
        print("="*70)
        
        return history, results


def main():
    """Run training pipeline."""
    # Configuration
    config = {
        'data_path': '../../data/combined_data.csv',
        'model_dir': '../../models',
        'processed_data_dir': '../../data/processed',
        'max_sequence_length': 150,  # Reduced from 300 for faster training
        'lstm_units': 128,
        'dropout_rate': 0.4,
        'dense_units': 64,
        'learning_rate': 0.001,
        'l2_regularization': 0.0001,  # Reduced from 0.01 - was causing huge loss!
        'epochs': 20,
        'batch_size': 64,  # Increased from 32 for faster training
        'patience': 5,
        'use_existing_data': False
    }
    
    # Create pipeline
    pipeline = TrainingPipeline(
        data_path=config['data_path'],
        model_dir=config['model_dir'],
        processed_data_dir=config['processed_data_dir']
    )
    
    # Run pipeline
    history, results = pipeline.run_full_pipeline(
        max_sequence_length=config['max_sequence_length'],
        lstm_units=config['lstm_units'],
        dropout_rate=config['dropout_rate'],
        dense_units=config['dense_units'],
        learning_rate=config['learning_rate'],
        l2_regularization=config['l2_regularization'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        patience=config['patience'],
        use_existing_data=config['use_existing_data']
    )
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

