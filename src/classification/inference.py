"""
Inference module for making predictions on new trajectory data.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple
import tensorflow as tf
from preprocessing import TrajectoryPreprocessor


class TrajectoryClassifier:
    """Production-ready classifier for trajectory sequences."""
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize classifier with trained model and preprocessor.
        
        Args:
            model_path: Path to trained model (.h5 file)
            preprocessor_path: Path to saved preprocessor (.pkl file)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
        # Load model and preprocessor
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = TrajectoryPreprocessor.load(preprocessor_path)
        
        self.class_names = ['Line', 'Arc']
        
        print(f"Model loaded from {model_path}")
        print(f"Preprocessor loaded from {preprocessor_path}")
    
    def predict_from_coordinates(
        self,
        coordinates: np.ndarray,
        threshold: float = 0.5,
        return_probability: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict class from raw coordinates.
        
        Args:
            coordinates: Array of shape (n_points, 2) with x, y coordinates
            threshold: Classification threshold
            return_probability: Whether to return probability
            
        Returns:
            Class name, or (class_name, probability) if return_probability=True
        """
        # Engineer features
        features = self.preprocessor.engineer_features(coordinates)
        
        # Pad or truncate
        padded = self.preprocessor.pad_or_truncate(features)
        
        # Add batch dimension
        X = np.expand_dims(padded, axis=0)
        
        # Normalize
        X = self.preprocessor.normalize_sequences(X, fit=False)
        
        # Predict
        prob = self.model.predict(X, verbose=0)[0, 0]
        pred = 1 if prob >= threshold else 0
        
        class_name = self.class_names[pred]
        
        if return_probability:
            return class_name, float(prob)
        return class_name
    
    def predict_batch(
        self,
        coordinates_list: List[np.ndarray],
        threshold: float = 0.5
    ) -> Tuple[List[str], np.ndarray]:
        """
        Predict classes for multiple trajectories.
        
        Args:
            coordinates_list: List of coordinate arrays
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Process all trajectories
        sequences = []
        for coords in coordinates_list:
            features = self.preprocessor.engineer_features(coords)
            padded = self.preprocessor.pad_or_truncate(features)
            sequences.append(padded)
        
        X = np.array(sequences)
        X = self.preprocessor.normalize_sequences(X, fit=False)
        
        # Predict
        probs = self.model.predict(X, verbose=0).flatten()
        preds = (probs >= threshold).astype(int)
        
        class_names = [self.class_names[p] for p in preds]
        
        return class_names, probs
    
    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Predict classes for trajectories in a dataframe.
        
        Args:
            df: DataFrame with columns [x, y, shape_id]
            threshold: Classification threshold
            
        Returns:
            DataFrame with predictions and probabilities
        """
        results = []
        
        for shape_id in df['shape_id'].unique():
            shape_data = df[df['shape_id'] == shape_id].sort_index()
            coords = shape_data[['x', 'y']].values
            
            class_name, prob = self.predict_from_coordinates(
                coords, threshold, return_probability=True
            )
            
            results.append({
                'shape_id': shape_id,
                'predicted_class': class_name,
                'probability': prob,
                'n_points': len(coords)
            })
        
        return pd.DataFrame(results)
    
    def predict_with_confidence(
        self,
        coordinates: np.ndarray,
        threshold: float = 0.5,
        confidence_threshold: float = 0.7
    ) -> dict:
        """
        Predict with confidence assessment.
        
        Args:
            coordinates: Coordinate array
            threshold: Classification threshold
            confidence_threshold: Minimum confidence for high-confidence prediction
            
        Returns:
            Dictionary with prediction, probability, and confidence level
        """
        class_name, prob = self.predict_from_coordinates(
            coordinates, threshold, return_probability=True
        )
        
        # Compute confidence (distance from 0.5)
        confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]
        
        confidence_level = 'high' if confidence >= confidence_threshold else 'low'
        
        return {
            'predicted_class': class_name,
            'probability': prob,
            'confidence': confidence,
            'confidence_level': confidence_level
        }
    
    def explain_prediction(self, coordinates: np.ndarray) -> dict:
        """
        Provide detailed explanation for a prediction.
        
        Args:
            coordinates: Coordinate array
            
        Returns:
            Dictionary with prediction details and feature statistics
        """
        # Get prediction
        result = self.predict_with_confidence(coordinates)
        
        # Compute features
        features = self.preprocessor.engineer_features(coordinates)
        
        # Feature statistics
        feature_names = [
            'rel_x', 'rel_y', 'dx', 'dy', 'speed',
            'dist_from_center', 'angle', 'curvature',
            'acceleration', 'consecutive_dist'
        ]
        
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feat_values = features[:, i]
            feature_stats[name] = {
                'mean': float(np.mean(feat_values)),
                'std': float(np.std(feat_values)),
                'min': float(np.min(feat_values)),
                'max': float(np.max(feat_values))
            }
        
        return {
            'prediction': result,
            'trajectory_info': {
                'n_points': len(coordinates),
                'x_range': [float(np.min(coordinates[:, 0])), 
                           float(np.max(coordinates[:, 0]))],
                'y_range': [float(np.min(coordinates[:, 1])), 
                           float(np.max(coordinates[:, 1]))]
            },
            'feature_statistics': feature_stats
        }


def load_and_predict(
    model_path: str,
    preprocessor_path: str,
    data_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Load model and make predictions on a dataset.
    
    Args:
        model_path: Path to model
        preprocessor_path: Path to preprocessor
        data_path: Path to data CSV
        output_path: Optional path to save predictions
        
    Returns:
        DataFrame with predictions
    """
    # Load classifier
    classifier = TrajectoryClassifier(model_path, preprocessor_path)
    
    # Load data
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print(f"Loaded {len(df)} data points from {df['shape_id'].nunique()} shapes")
    
    # Make predictions
    predictions = classifier.predict_from_dataframe(df)
    
    # Save if output path provided
    if output_path:
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return predictions


if __name__ == '__main__':
    # Example usage
    print("Inference module for trajectory classification.")
    print("\nExample usage:")
    print("""
    from inference import TrajectoryClassifier
    
    classifier = TrajectoryClassifier(
        model_path='models/best_model.h5',
        preprocessor_path='models/preprocessor.pkl'
    )
    
    # Predict from coordinates
    coords = np.array([[0, 0], [1, 1], [2, 2], ...])
    prediction, probability = classifier.predict_from_coordinates(coords)
    
    print(f"Prediction: {prediction}, Probability: {probability:.3f}")
    """)

