"""
Jupyter Notebook Script for Data Analysis and Exploration
Machine Learning-Based Cyber Attack Detection System

This script provides comprehensive data analysis and visualization
for cybersecurity datasets used in the ML-based attack detection system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CyberDataAnalyzer:
    """
    Comprehensive data analysis class for cybersecurity datasets
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("="*80)
        print("CYBERSECURITY DATASET - BASIC STATISTICS")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Numeric Columns: {len(self.numeric_columns)}")
        print(f"Categorical Columns: {len(self.categorical_columns)}")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing Values: {missing_values.sum()}")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found ✓")
        
        # Data types
        print(f"\nData Types Distribution:")
        print(self.df.dtypes.value_counts())
        
        return self.df.describe()
    
    def label_distribution_analysis(self, label_column='Label'):
        """Analyze the distribution of attack types"""
        if label_column not in self.df.columns:
            print(f"Column '{label_column}' not found in dataset")
            return
        
        print("\n" + "="*60)
        print("ATTACK TYPE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Count distribution
        label_counts = self.df[label_column].value_counts()
        print("Label Distribution:")
        print(label_counts)
        
        # Percentage distribution
        label_percentages = (label_counts / len(self.df)) * 100
        print("\nPercentage Distribution:")
        for label, percentage in label_percentages.items():
            print(f"{label}: {percentage:.2f}%")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        label_counts.plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Attack Type Distribution (Count)')
        axes[0].set_xlabel('Attack Type')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Attack Type Distribution (Percentage)')
        
        plt.tight_layout()
        plt.show()
        
        return label_counts
    
    def feature_correlation_analysis(self, top_n=20):
        """Analyze feature correlations"""
        print("\n" + "="*60)
        print("FEATURE CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns for correlation
        numeric_df = self.df[self.numeric_columns]
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        
        print(f"Highly correlated feature pairs (|correlation| > 0.8): {len(high_corr_pairs)}")
        for feat1, feat2, corr in high_corr_pairs[:10]:  # Show top 10
            print(f"{feat1} <-> {feat2}: {corr:.3f}")
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        
        # Select top features for visualization
        if len(self.numeric_columns) > top_n:
            # Select features with highest variance
            feature_variance = numeric_df.var().sort_values(ascending=False)
            top_features = feature_variance.head(top_n).index.tolist()
            plot_corr = correlation_matrix.loc[top_features, top_features]
        else:
            plot_corr = correlation_matrix
        
        sns.heatmap(plot_corr, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'Feature Correlation Matrix (Top {len(plot_corr)} Features)')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def attack_pattern_analysis(self, label_column='Label'):
        """Analyze patterns in different attack types"""
        if label_column not in self.df.columns:
            print(f"Column '{label_column}' not found in dataset")
            return
        
        print("\n" + "="*60)
        print("ATTACK PATTERN ANALYSIS")
        print("="*60)
        
        # Group by attack type and analyze key features
        attack_groups = self.df.groupby(label_column)
        
        # Select key network features for analysis
        key_features = [col for col in self.numeric_columns 
                       if any(keyword in col.lower() for keyword in 
                             ['packet', 'byte', 'flow', 'duration', 'rate'])][:8]
        
        if not key_features:
            key_features = self.numeric_columns[:8]  # Fallback to first 8 numeric columns
        
        print(f"Analyzing key features: {key_features}")
        
        # Statistical summary by attack type
        attack_stats = attack_groups[key_features].mean()
        print("\nMean values by attack type:")
        print(attack_stats)
        
        # Visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            if i < len(axes):
                attack_groups[feature].mean().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{feature} by Attack Type')
                axes[i].set_xlabel('Attack Type')
                axes[i].set_ylabel('Mean Value')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(key_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return attack_stats
    
    def dimensionality_reduction_analysis(self, label_column='Label', sample_size=5000):
        """Perform PCA and t-SNE analysis"""
        print("\n" + "="*60)
        print("DIMENSIONALITY REDUCTION ANALYSIS")
        print("="*60)
        
        # Prepare data
        numeric_df = self.df[self.numeric_columns].fillna(0)
        
        # Sample data if too large
        if len(numeric_df) > sample_size:
            sample_indices = np.random.choice(len(numeric_df), sample_size, replace=False)
            numeric_df_sample = numeric_df.iloc[sample_indices]
            labels_sample = self.df[label_column].iloc[sample_indices] if label_column in self.df.columns else None
        else:
            numeric_df_sample = numeric_df
            labels_sample = self.df[label_column] if label_column in self.df.columns else None
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df_sample)
        
        # PCA Analysis
        print("Performing PCA analysis...")
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"First 10 components explain {cumulative_variance[9]:.2%} of variance")
        
        # t-SNE Analysis
        print("Performing t-SNE analysis...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(scaled_data[:1000])  # Limit for t-SNE performance
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA Explained Variance
        axes[0, 0].plot(range(1, min(21, len(explained_variance_ratio)+1)), 
                       explained_variance_ratio[:20], 'bo-')
        axes[0, 0].set_title('PCA Explained Variance Ratio')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].grid(True)
        
        # Cumulative Explained Variance
        axes[0, 1].plot(range(1, min(21, len(cumulative_variance)+1)), 
                       cumulative_variance[:20], 'ro-')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].grid(True)
        
        # PCA 2D Visualization
        if labels_sample is not None:
            unique_labels = labels_sample.iloc[:len(pca_result)].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels_sample.iloc[:len(pca_result)] == label
                axes[1, 0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                                 c=[colors[i]], label=label, alpha=0.6)
            axes[1, 0].set_title('PCA 2D Visualization')
            axes[1, 0].set_xlabel('First Principal Component')
            axes[1, 0].set_ylabel('Second Principal Component')
            axes[1, 0].legend()
            
            # t-SNE 2D Visualization
            for i, label in enumerate(unique_labels):
                mask = labels_sample.iloc[:len(tsne_result)] == label
                axes[1, 1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                                 c=[colors[i]], label=label, alpha=0.6)
            axes[1, 1].set_title('t-SNE 2D Visualization')
            axes[1, 1].set_xlabel('t-SNE Component 1')
            axes[1, 1].set_ylabel('t-SNE Component 2')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return pca, tsne, explained_variance_ratio
    
    def temporal_analysis(self, timestamp_column=None):
        """Analyze temporal patterns in the data"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        if timestamp_column and timestamp_column in self.df.columns:
            # Real timestamp analysis
            self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column])
            
            # Attack frequency over time
            attack_timeline = self.df.groupby([timestamp_column, 'Label']).size().unstack(fill_value=0)
            
            plt.figure(figsize=(15, 8))
            attack_timeline.plot(kind='area', stacked=True, alpha=0.7)
            plt.title('Attack Frequency Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Attacks')
            plt.legend(title='Attack Type')
            plt.show()
        else:
            # Simulated temporal analysis based on data order
            print("No timestamp column found. Performing analysis based on data order...")
            
            # Create time windows based on data index
            window_size = len(self.df) // 20  # 20 time windows
            time_windows = []
            
            for i in range(0, len(self.df), window_size):
                window_data = self.df.iloc[i:i+window_size]
                if 'Label' in window_data.columns:
                    attack_counts = window_data['Label'].value_counts()
                    time_windows.append({
                        'window': i // window_size,
                        'total_samples': len(window_data),
                        **attack_counts.to_dict()
                    })
            
            # Convert to DataFrame and visualize
            temporal_df = pd.DataFrame(time_windows).fillna(0)
            
            plt.figure(figsize=(12, 6))
            attack_columns = [col for col in temporal_df.columns 
                            if col not in ['window', 'total_samples']]
            
            for col in attack_columns:
                plt.plot(temporal_df['window'], temporal_df[col], marker='o', label=col)
            
            plt.title('Attack Distribution Across Data Windows')
            plt.xlabel('Data Window')
            plt.ylabel('Number of Samples')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def feature_distribution_analysis(self, top_features=8):
        """Analyze distribution of key features"""
        print("\n" + "="*60)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Select top features by variance
        feature_variance = self.df[self.numeric_columns].var().sort_values(ascending=False)
        top_feature_names = feature_variance.head(top_features).index.tolist()
        
        print(f"Analyzing top {top_features} features by variance:")
        for i, feature in enumerate(top_feature_names, 1):
            print(f"{i}. {feature}: variance = {feature_variance[feature]:.2e}")
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_feature_names):
            if i < len(axes):
                # Histogram
                self.df[feature].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue')
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Box plots by attack type
        if 'Label' in self.df.columns:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(top_feature_names):
                if i < len(axes):
                    self.df.boxplot(column=feature, by='Label', ax=axes[i])
                    axes[i].set_title(f'{feature} by Attack Type')
                    axes[i].set_xlabel('Attack Type')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Feature Distributions by Attack Type')
            plt.tight_layout()
            plt.show()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive data analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CYBERSECURITY DATA ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        basic_stats = self.basic_statistics()
        
        # Label distribution
        if 'Label' in self.df.columns:
            label_dist = self.label_distribution_analysis()
        
        # Correlation analysis
        correlation_matrix = self.feature_correlation_analysis()
        
        # Attack pattern analysis
        if 'Label' in self.df.columns:
            attack_patterns = self.attack_pattern_analysis()
        
        # Dimensionality reduction
        pca, tsne, variance_ratio = self.dimensionality_reduction_analysis()
        
        # Feature distributions
        self.feature_distribution_analysis()
        
        # Temporal analysis
        self.temporal_analysis()
        
        print("\n" + "="*80)
        print("DATA ANALYSIS COMPLETED")
        print("="*80)
        print("Key Insights:")
        print("• Dataset characteristics and quality assessed")
        print("• Attack type distributions analyzed")
        print("• Feature correlations and importance identified")
        print("• Dimensionality reduction reveals data structure")
        print("• Temporal patterns in attack occurrences examined")
        print("="*80)

def main():
    """Main function to run the data analysis"""
    # Import data loader
    import sys
    sys.path.append('../src')
    from data_loader import CyberDataLoader
    
    # Load sample data
    loader = CyberDataLoader()
    df = loader.get_sample_data()
    
    # Create analyzer and run comprehensive analysis
    analyzer = CyberDataAnalyzer(df)
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()