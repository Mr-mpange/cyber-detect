"""
Data Loading and Preprocessing Module for Cyber Attack Detection System
Supports Kaggle datasets and real cybersecurity data
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CyberDataLoader:
    """
    Handles loading and preprocessing of cybersecurity datasets
    Supports CIC-IDS2017, UNSW-NB15, NSL-KDD datasets, and Kaggle datasets
    """
    
    def __init__(self, dataset_type='CIC-IDS2017'):
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_kaggle_dataset(self, file_path, label_column='Label', sample_size=None):
        """
        Load cybersecurity dataset from Kaggle with enhanced preprocessing
        
        Popular Kaggle datasets:
        - Network Intrusion Detection Dataset
        - CIC-IDS2017 Dataset
        - UNSW-NB15 Dataset
        - NSL-KDD Dataset
        
        Args:
            file_path: Path to CSV file
            label_column: Name of label column
            sample_size: Number of samples to load (None for all)
            
        Returns:
            df: Cleaned dataframe
            actual_label_col: Actual label column name used
        """
        print(f"Loading Kaggle dataset from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Try to read the CSV file
        try:
            # Read with sample size if specified
            if sample_size:
                print(f"Loading sample of {sample_size} rows...")
                df = pd.read_csv(file_path, nrows=sample_size)
            else:
                df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
        
        # Display basic info
        print(f"Columns: {len(df.columns)} features")
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for label column
        possible_labels = [label_column, 'label', 'Label', 'attack', 'Attack', 'class', 'Class', 'target', 'Target']
        actual_label_col = None
        
        for col in possible_labels:
            if col in df.columns:
                actual_label_col = col
                break
        
        if actual_label_col is None:
            print("Available columns:", df.columns.tolist())
            raise ValueError(f"Label column not found. Please specify the correct column name from: {df.columns.tolist()}")
        
        print(f"Using label column: {actual_label_col}")
        print(f"Label distribution:\n{df[actual_label_col].value_counts().head(10)}")
        
        # Enhanced data cleaning
        df_cleaned = self.enhanced_data_cleaning(df, actual_label_col)
        
        return df_cleaned, actual_label_col
    
    def enhanced_data_cleaning(self, df, label_column):
        """
        Enhanced data cleaning for real cybersecurity datasets
        
        Args:
            df: Input dataframe
            label_column: Name of label column
            
        Returns:
            Cleaned dataframe
        """
        print("Performing enhanced data cleaning...")
        
        # Handle infinite and NaN values
        print("Handling infinite and missing values...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count missing values before cleaning
        missing_before = df.isnull().sum().sum()
        
        # Fill NaN values for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != label_column]
        
        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical NaN values
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != label_column]
        
        for col in categorical_columns:
            mode_values = df[col].mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else 'unknown'
            df[col] = df[col].fillna(fill_value)
        
        # Handle label column missing values
        if df[label_column].isnull().any():
            print(f"Warning: Found missing values in label column '{label_column}'")
            df = df.dropna(subset=[label_column])
            print(f"Dropped rows with missing labels. New shape: {df.shape}")
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        # Remove duplicate rows
        duplicates_before = len(df)
        df = df.drop_duplicates()
        duplicates_after = len(df)
        if duplicates_before != duplicates_after:
            print(f"Removed {duplicates_before - duplicates_after} duplicate rows")
        
        # Handle outliers in numeric columns (optional - can be aggressive)
        print("Checking for extreme outliers...")
        for col in numeric_columns[:5]:  # Check first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Very conservative outlier removal (only extreme outliers)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"  {col}: {outliers} extreme outliers detected")
        
        print(f"Data cleaning completed. Final shape: {df.shape}")
        return df
        
    def load_cic_ids2017(self, file_path):
        """
        Load CIC-IDS2017 dataset
        
        Dataset Features:
        - 78 features including flow duration, packet statistics, flags
        - Labels: BENIGN, DoS, DDoS, PortScan, Bot, Infiltration, Web Attack, Brute Force
        """
        print("Loading CIC-IDS2017 dataset...")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names (remove spaces and special characters)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        
        return df
    
    def preprocess_kaggle_data(self, df, target_column='Label'):
        """
        Preprocess Kaggle dataset for machine learning
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            
        Returns:
            X_train, X_test, y_train, y_test: Preprocessed data splits
        """
        print("Preprocessing Kaggle data...")
        
        # Create binary classification (Normal vs Attack)
        y = df[target_column].copy()
        
        # Convert to binary classification
        # Common normal/benign labels
        normal_labels = ['BENIGN', 'benign', 'normal', 'Normal', 'NORMAL', '0', 0]
        
        # Create binary labels
        y_binary = np.where(y.isin(normal_labels), 0, 1)
        
        print(f"Binary classification - Normal: {np.sum(y_binary == 0)}, Attack: {np.sum(y_binary == 1)}")
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"Encoding categorical features: {categorical_columns.tolist()}")
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Remove constant features (if any)
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"Removing constant features: {constant_features}")
            X = X.drop(columns=constant_features)
            self.feature_names = [col for col in self.feature_names if col not in constant_features]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Features used: {len(self.feature_names)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        """
        Preprocess the dataset for machine learning
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            
        Returns:
            X_train, X_test, y_train, y_test: Preprocessed data splits
        """
        print("Preprocessing data...")
        
        # Separate features and target
        if target_column not in df.columns:
            # Try common label column names
            possible_labels = ['Label', 'label', 'Attack', 'attack', 'class', 'Class']
            target_column = None
            for col in possible_labels:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                raise ValueError("Target column not found. Please specify the correct column name.")
        
        # Create binary classification (Normal vs Attack)
        y = df[target_column].copy()
        
        # Convert to binary: BENIGN/Normal = 0, Everything else = 1
        y_binary = np.where(y.str.upper().str.contains('BENIGN|NORMAL'), 0, 1)
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Handle categorical features if any
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            print(f"Encoding categorical features: {categorical_columns.tolist()}")
            for col in categorical_columns:
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Class distribution - Normal: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_sample_data(self):
        """
        Generate sample cybersecurity data for demonstration
        This simulates network traffic features
        """
        print("Generating sample cybersecurity data...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Simulate network traffic features
        data = {
            'flow_duration': np.random.exponential(1000, n_samples),
            'total_fwd_packets': np.random.poisson(50, n_samples),
            'total_bwd_packets': np.random.poisson(30, n_samples),
            'total_length_fwd_packets': np.random.exponential(2000, n_samples),
            'total_length_bwd_packets': np.random.exponential(1500, n_samples),
            'fwd_packet_length_max': np.random.exponential(500, n_samples),
            'fwd_packet_length_min': np.random.exponential(50, n_samples),
            'fwd_packet_length_mean': np.random.exponential(200, n_samples),
            'bwd_packet_length_max': np.random.exponential(400, n_samples),
            'bwd_packet_length_min': np.random.exponential(40, n_samples),
            'flow_bytes_per_sec': np.random.exponential(10000, n_samples),
            'flow_packets_per_sec': np.random.exponential(100, n_samples),
            'flow_iat_mean': np.random.exponential(1000, n_samples),
            'flow_iat_std': np.random.exponential(500, n_samples),
            'flow_iat_max': np.random.exponential(2000, n_samples),
            'flow_iat_min': np.random.exponential(10, n_samples),
            'fwd_iat_total': np.random.exponential(5000, n_samples),
            'fwd_iat_mean': np.random.exponential(800, n_samples),
            'bwd_iat_total': np.random.exponential(4000, n_samples),
            'bwd_iat_mean': np.random.exponential(600, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create labels (80% normal, 20% attack)
        labels = np.random.choice(['BENIGN', 'DoS', 'DDoS', 'PortScan', 'Bot'], 
                                 n_samples, p=[0.8, 0.05, 0.05, 0.05, 0.05])
        df['Label'] = labels
        
        # Add some attack patterns
        attack_mask = df['Label'] != 'BENIGN'
        
        # DoS attacks: high packet rate, low duration
        dos_mask = df['Label'] == 'DoS'
        df.loc[dos_mask, 'flow_packets_per_sec'] *= 10
        df.loc[dos_mask, 'flow_duration'] /= 5
        
        # DDoS attacks: very high packet rate
        ddos_mask = df['Label'] == 'DDoS'
        df.loc[ddos_mask, 'flow_packets_per_sec'] *= 20
        df.loc[ddos_mask, 'total_fwd_packets'] *= 5
        
        # Port scan: many small packets
        portscan_mask = df['Label'] == 'PortScan'
        df.loc[portscan_mask, 'fwd_packet_length_mean'] /= 10
        df.loc[portscan_mask, 'total_fwd_packets'] *= 3
        
        print(f"Sample data generated: {df.shape}")
        print(f"Label distribution:\n{df['Label'].value_counts()}")
        
        return df

def main():
    """Test the data loader"""
    loader = CyberDataLoader()
    
    # Generate sample data
    df = loader.get_sample_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = loader.preprocess_data(df)
    
    print("\nData loading and preprocessing completed successfully!")
    print(f"Feature names: {loader.feature_names[:10]}...")  # Show first 10 features

if __name__ == "__main__":
    main()