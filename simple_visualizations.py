"""
Simple Data Visualization for Cyber Attack Detection System
Creates basic visualizations without Unicode characters to avoid encoding issues
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_output_directory():
    """Create directory for visualization outputs"""
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_sample_data():
    """Generate sample data for visualization"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample network data
    data = {
        'duration': np.random.exponential(2, n_samples),
        'src_bytes': np.random.lognormal(8, 2, n_samples),
        'dst_bytes': np.random.lognormal(7, 2, n_samples),
        'packets': np.random.poisson(10, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'other'], n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples, p=[0.6, 0.2, 0.1, 0.1])
    }
    
    # Create attack labels
    attack_types = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    labels = np.random.choice(attack_types, n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02])
    data['Label'] = labels
    
    return pd.DataFrame(data)

def create_basic_visualizations(df, output_dir):
    """Create basic performance and data visualizations"""
    
    # 1. Attack Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cyber Attack Detection System - Data Analysis', fontsize=16, fontweight='bold')
    
    # Attack type distribution
    attack_counts = df['Label'].value_counts()
    axes[0, 0].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Attack Type Distribution')
    
    # Protocol distribution
    protocol_counts = df['protocol'].value_counts()
    axes[0, 1].bar(protocol_counts.index, protocol_counts.values)
    axes[0, 1].set_title('Protocol Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Duration by attack type
    df.boxplot(column='duration', by='Label', ax=axes[1, 0])
    axes[1, 0].set_title('Duration by Attack Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Bytes distribution
    normal_data = df[df['Label'] == 'normal']['src_bytes']
    attack_data = df[df['Label'] != 'normal']['src_bytes']
    
    axes[1, 1].hist(normal_data, alpha=0.7, label='Normal', bins=30)
    axes[1, 1].hist(attack_data, alpha=0.7, label='Attack', bins=30)
    axes[1, 1].set_title('Source Bytes Distribution')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Source Bytes (log scale)')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Sample model results
    models = ['Random Forest', 'XGBoost', 'Neural Network', 'SVM', 'Ensemble']
    accuracy = [0.926, 0.926, 0.898, 0.882, 0.926]
    f1_score = [0.791, 0.790, 0.737, 0.689, 0.789]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
    axes[0].bar(x + width/2, f1_score, width, label='F1-Score', alpha=0.8)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Accuracy and F1-Score Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Performance radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed']
    rf_scores = [0.926, 0.897, 0.707, 0.791, 0.85]
    xgb_scores = [0.926, 0.901, 0.703, 0.790, 0.80]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    rf_scores += rf_scores[:1]
    xgb_scores += xgb_scores[:1]
    angles += angles[:1]
    
    ax_radar = plt.subplot(1, 2, 2, projection='polar')
    ax_radar.plot(angles, rf_scores, 'o-', linewidth=2, label='Random Forest', color='blue')
    ax_radar.fill(angles, rf_scores, alpha=0.25, color='blue')
    ax_radar.plot(angles, xgb_scores, 'o-', linewidth=2, label='XGBoost', color='red')
    ax_radar.fill(angles, xgb_scores, alpha=0.25, color='red')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Model Performance Radar Chart')
    ax_radar.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Real-time Detection Simulation
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Real-time Attack Detection Simulation', fontsize=16, fontweight='bold')
    
    # Simulate network traffic over time
    time_points = np.arange(0, 100, 0.1)
    normal_traffic = 50 + 10 * np.sin(0.1 * time_points) + np.random.normal(0, 5, len(time_points))
    
    # Add attack events
    dos_start, dos_end = 20, 30
    ddos_start, ddos_end = 60, 70
    
    traffic_with_attacks = normal_traffic.copy()
    traffic_with_attacks[dos_start*10:dos_end*10] += 200 + np.random.normal(0, 20, dos_end*10-dos_start*10)
    traffic_with_attacks[ddos_start*10:ddos_end*10] += 400 + np.random.normal(0, 30, ddos_end*10-ddos_start*10)
    
    axes[0].plot(time_points, traffic_with_attacks, color='blue', alpha=0.7, linewidth=1)
    axes[0].axvspan(dos_start, dos_end, alpha=0.3, color='red', label='DoS Attack')
    axes[0].axvspan(ddos_start, ddos_end, alpha=0.3, color='darkred', label='DDoS Attack')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Network Traffic (packets/sec)')
    axes[0].set_title('Network Traffic with Attack Events')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Detection confidence over time
    detection_threshold = 0.8
    confidence_scores = []
    
    for i, traffic in enumerate(traffic_with_attacks):
        if traffic > 150:  # High traffic indicates potential attack
            confidence = min(0.95, 0.5 + (traffic - 150) / 500)
        else:
            confidence = max(0.1, 0.3 - np.random.random() * 0.2)
        confidence_scores.append(confidence)
    
    axes[1].plot(time_points, confidence_scores, color='purple', linewidth=2)
    axes[1].axhline(detection_threshold, color='red', linestyle='--', 
                   label=f'Alert Threshold ({detection_threshold})')
    axes[1].fill_between(time_points, confidence_scores, detection_threshold, 
                        where=np.array(confidence_scores) > detection_threshold, 
                        alpha=0.3, color='red', label='Attack Detected')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Detection Confidence')
    axes[1].set_title('Attack Detection Confidence Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_realtime_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function"""
    print("Starting Cyber Attack Detection Visualization")
    print("="*50)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data()
    
    # Create visualizations
    print("Creating visualizations...")
    create_basic_visualizations(df, output_dir)
    
    print("Visualizations completed successfully!")
    print(f"Files saved in: ./{output_dir}/")
    print("Generated files:")
    print("  - 01_data_analysis.png")
    print("  - 02_model_performance.png") 
    print("  - 03_realtime_detection.png")

if __name__ == "__main__":
    main()