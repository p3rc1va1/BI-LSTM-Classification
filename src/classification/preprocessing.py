"""
Data preprocessing and feature engineering for trajectory classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Dict, List


class TrajectoryPreprocessor:
    """Handles data preprocessing and feature engineering for trajectory sequences."""
    
    def __init__(self, max_sequence_length: int = 300):
        """
        Initialize preprocessor.
        
        Args:
            max_sequence_length: Maximum length for padding/truncating sequences
        """
        self.max_sequence_length = max_sequence_length
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load combined dataset."""
        df = pd.read_csv(filepath)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    
    def normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates for scale and translation invariance.
        
        Centers at origin and scales to unit size while preserving aspect ratio.
        This ensures the model learns geometry, not absolute position/size.
        
        Args:
            coords: Array of shape (n_points, 2) with x, y coordinates
            
        Returns:
            Normalized coordinates centered at origin and scaled to [-1, 1] range
        """
        # Center at origin (translation invariance)
        centroid = coords.mean(axis=0)
        coords_centered = coords - centroid
        
        # Scale to unit size (scale invariance)
        # Use max range to preserve aspect ratio
        x_range = coords_centered[:, 0].max() - coords_centered[:, 0].min()
        y_range = coords_centered[:, 1].max() - coords_centered[:, 1].min()
        max_range = max(x_range, y_range)
        
        if max_range > 1e-8:  # Avoid division by zero
            coords_normalized = coords_centered / max_range
        else:
            coords_normalized = coords_centered
        
        return coords_normalized
    
    def engineer_features(self, coords: np.ndarray) -> np.ndarray:
        """
        Engineer features from raw coordinates.
        
        IMPORTANT: Normalizes coordinates first for scale/translation invariance.
        This ensures a line is a line regardless of position or size.
        
        Args:
            coords: Array of shape (n_points, 2) with x, y coordinates
            
        Returns:
            Array of shape (n_points, n_features) with engineered features
        """
        # STEP 1: Normalize coordinates for scale/translation invariance
        coords = self.normalize_coordinates(coords)
        
        features = []
        
        # Extract normalized coordinates (already centered at origin and scaled)
        x, y = coords[:, 0], coords[:, 1]
        
        # Use normalized coordinates as features (already relative to centroid)
        features.extend([x, y])
        
        # Velocity/Direction (first differences)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        features.extend([dx, dy])
        
        # Speed (magnitude of velocity)
        speed = np.sqrt(dx**2 + dy**2)
        features.append(speed)
        
        # Distance from centroid (origin after normalization)
        dist_from_center = np.sqrt(x**2 + y**2)
        features.append(dist_from_center)
        
        # Angle of motion
        angles = np.arctan2(dy, dx)
        features.append(angles)
        
        # Curvature (change in angle)
        curvature = np.diff(angles, prepend=angles[0])
        # Normalize curvature to [-pi, pi]
        curvature = np.arctan2(np.sin(curvature), np.cos(curvature))
        features.append(curvature)
        
        # Acceleration (change in speed)
        acceleration = np.diff(speed, prepend=speed[0])
        features.append(acceleration)
        
        # Distance between consecutive points
        dist_consecutive = np.sqrt(dx**2 + dy**2)
        features.append(dist_consecutive)
        
        # ===== TOP DISCRIMINATIVE FEATURES (from Random Forest analysis) =====
        
        # 1. Cumulative angle change (temporal version of total_angle_change)
        # Running sum of absolute curvature - arcs accumulate steadily, lines stay near 0
        cumulative_angle = np.cumsum(np.abs(curvature))
        features.append(cumulative_angle)
        
        # 2. Local curvature variability (temporal version of std_curvature)
        # Rolling std of curvature - arcs have consistent curvature, lines near 0
        window_size = 5
        curvature_rolling_std = np.zeros(len(curvature))
        for i in range(len(curvature)):
            start_idx = max(0, i - window_size + 1)
            window = curvature[start_idx:i+1]
            curvature_rolling_std[i] = np.std(window) if len(window) > 1 else 0
        features.append(curvature_rolling_std)
        
        # 3. Local radius consistency (temporal version of cv_radius)
        # Rolling CV of distance from centroid - arcs have low CV, lines/other have high CV
        radius_rolling_cv = np.zeros(len(dist_from_center))
        for i in range(len(dist_from_center)):
            start_idx = max(0, i - window_size + 1)
            window = dist_from_center[start_idx:i+1]
            if len(window) > 1:
                mean_val = np.mean(window)
                std_val = np.std(window)
                radius_rolling_cv[i] = std_val / (mean_val + 1e-8)
            else:
                radius_rolling_cv[i] = 0
        features.append(radius_rolling_cv)
        
        # Stack all features
        feature_array = np.column_stack(features)
        
        return feature_array
    
    def pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad or truncate sequence to fixed length.
        
        Args:
            sequence: Array of shape (n_points, n_features)
            
        Returns:
            Array of shape (max_sequence_length, n_features)
        """
        n_points, n_features = sequence.shape
        
        if n_points > self.max_sequence_length:
            # Truncate by sampling evenly
            indices = np.linspace(0, n_points - 1, self.max_sequence_length, dtype=int)
            return sequence[indices]
        elif n_points < self.max_sequence_length:
            # Pad with zeros
            padding = np.zeros((self.max_sequence_length - n_points, n_features))
            return np.vstack([sequence, padding])
        else:
            return sequence
    
    def create_sequences(self, df: pd.DataFrame, min_segment_length: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create feature sequences from dataframe by extracting contiguous segments.
        
        Each shape_id may contain multiple segments (line/arc transitions).
        This method extracts each contiguous segment with the same label.
        
        Args:
            df: DataFrame with columns [x, y, label, shape, shape_id]
            min_segment_length: Minimum points required for a valid segment
            
        Returns:
            Tuple of (sequences, labels, segment_ids)
            - sequences: (n_segments, max_seq_len, n_features)
            - labels: (n_segments,) - 0 (arc) and 1 (line), label 2 (transitions) filtered out
            - segment_ids: List of segment identifiers (shape_id + segment_num)
        """
        sequences = []
        labels = []
        segment_ids = []
        
        for shape_id in df['shape_id'].unique():
            shape_data = df[df['shape_id'] == shape_id].sort_index()
            
            # Find label transitions to split into segments
            label_array = shape_data['label'].values
            
            # Identify segment boundaries (where label changes)
            segment_starts = [0]
            for i in range(1, len(label_array)):
                if label_array[i] != label_array[i-1]:
                    segment_starts.append(i)
            segment_starts.append(len(label_array))  # End boundary
            
            # Extract each segment
            for seg_num, (start, end) in enumerate(zip(segment_starts[:-1], segment_starts[1:])):
                segment_length = end - start
                
                # Skip segments that are too short
                if segment_length < min_segment_length:
                    continue
                
                # Get segment data
                segment = shape_data.iloc[start:end]
                coords = segment[['x', 'y']].values
                segment_label = segment['label'].iloc[0]
                
                # Skip label 2 (transition points - too short for meaningful sequences)
                if segment_label == 2:
                    continue
                
                # Only accept labels 0 (arc) and 1 (line)
                if segment_label not in [0, 1]:
                    continue
                
                # Engineer features for this segment
                features = self.engineer_features(coords)
                
                # Pad or truncate
                padded_features = self.pad_or_truncate(features)
                
                sequences.append(padded_features)
                
                # Keep labels as-is: 0 = arc, 1 = line
                labels.append(segment_label)
                
                # Create unique segment identifier
                segment_ids.append(f"{shape_id}_seg{seg_num}")
        
        return np.array(sequences), np.array(labels), segment_ids
    
    def normalize_sequences(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features across all sequences.
        
        Args:
            X: Array of shape (n_samples, seq_len, n_features)
            fit: Whether to fit the scaler
            
        Returns:
            Normalized array of same shape
        """
        n_samples, seq_len, n_features = X.shape
        
        # Reshape to 2D for scaling
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to 3D
        return X_scaled.reshape(n_samples, seq_len, n_features)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Prepare complete dataset with train/val/test splits.
        
        Args:
            df: Input dataframe
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test)
            random_state: Random seed
            
        Returns:
            Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Create sequences (extracts segments from shapes)
        X, y, segment_ids = self.create_sequences(df)
        
        print(f"\nExtracted {len(X)} segments from {df['shape_id'].nunique()} shapes")
        print(f"Sequence shape: {X.shape}")
        print(f"Feature dimensions: {X.shape[2]}")
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        # Normalize (fit on train, transform all)
        X_train = self.normalize_sequences(X_train, fit=True)
        X_val = self.normalize_sequences(X_val, fit=False)
        X_test = self.normalize_sequences(X_test, fit=False)
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        print(f"\nLabel distribution:")
        print(f"  Train - Line: {np.sum(y_train == 0)}, Arc: {np.sum(y_train == 1)}")
        print(f"  Val   - Line: {np.sum(y_val == 0)}, Arc: {np.sum(y_val == 1)}")
        print(f"  Test  - Line: {np.sum(y_test == 0)}, Arc: {np.sum(y_test == 1)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'max_sequence_length': self.max_sequence_length,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrajectoryPreprocessor':
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(max_sequence_length=state['max_sequence_length'])
        preprocessor.scaler = state['scaler']
        preprocessor.is_fitted = state['is_fitted']
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def main():
    """Example usage of preprocessor."""
    import os
    from pathlib import Path
    
    # Get project root (assuming script is in src/classification/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Create directories
    processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    data_path = project_root / 'data' / 'combined_data.csv'
    
    # Load and process data
    preprocessor = TrajectoryPreprocessor(max_sequence_length=300)
    df = preprocessor.load_data(str(data_path))
    
    print("Dataset loaded:")
    print(f"  Total points: {len(df):,}")
    print(f"  Unique shapes: {df['shape_id'].nunique()}")
    print(f"  Labels: {sorted(df['label'].unique())}")
    
    # Prepare data
    data = preprocessor.prepare_data(df)
    
    # Save processed data
    np.save(str(processed_dir / 'X_train.npy'), data['X_train'])
    np.save(str(processed_dir / 'X_val.npy'), data['X_val'])
    np.save(str(processed_dir / 'X_test.npy'), data['X_test'])
    np.save(str(processed_dir / 'y_train.npy'), data['y_train'])
    np.save(str(processed_dir / 'y_val.npy'), data['y_val'])
    np.save(str(processed_dir / 'y_test.npy'), data['y_test'])
    
    # Save preprocessor
    preprocessor.save(str(models_dir / 'preprocessor.pkl'))
    
    print("\nData preprocessing complete!")
    print(f"Files saved in {processed_dir} and {models_dir}")


if __name__ == '__main__':
    main()

