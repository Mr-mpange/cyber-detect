# �️ Cyber Atttack Detection System

A comprehensive machine learning-based cybersecurity system that achieves 92%+ accuracy in detecting network intrusions and cyber attacks.

## 🚀 Features

- **High Accuracy**: Achieves 92.64% accuracy with ensemble ML models
- **Real Data Processing**: Works with real cybersecurity datasets
- **8 Optimized Models**: Random Forest, XGBoost, Neural Networks, SVM, and more
- **Automatic Visualizations**: Generates comprehensive analysis graphs
- **Production Ready**: Saves trained models for deployment

## 📁 Project Structure

```
cyber-detect/
├── main.py                           # Main execution script
├── data_visualization_analysis.py    # Visualization generation
├── download_datasets.py              # Dataset download utility
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── data/                             # Dataset storage
│   └── sample_network_intrusion.csv  # Sample cybersecurity data
├── src/                              # Source code modules
│   ├── data_loader.py                # Data loading utilities
│   ├── enhanced_models.py            # Enhanced ML models
│   └── models.py                     # Base model definitions
├── models/                           # Trained model storage
│   └── kaggle/                       # Real data models
├── results/                          # Analysis results
│   └── kaggle/                       # Real data results
└── visualizations/                   # Generated graphs
    ├── 01_data_distribution_overview.png
    ├── 02_attack_pattern_analysis.png
    ├── 03_feature_engineering_impact.png
    ├── 04_model_performance_comparison.png
    ├── 05_preprocessing_pipeline.png
    ├── 06_realtime_detection_simulation.png
    └── 07_comprehensive_dashboard.png
```

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cyber-detect
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start
```bash
python main.py
```

This will:
- Auto-download real cybersecurity data
- Train 8 optimized ML models
- Generate comprehensive performance reports
- Create detailed visualizations
- Save all models for production use

### Advanced Usage
```bash
# Use specific dataset
python main.py --dataset data/your_dataset.csv

# Specify label column
python main.py --dataset data/your_dataset.csv --label attack_type

# Limit samples for testing
python main.py --sample 1000

# Show dataset options
python main.py --info
```

## 📊 Performance Results

- **Best Accuracy**: 92.64% (Ultra Random Forest & XGBoost)
- **Best F1-Score**: 79.09% (Ultra Random Forest)
- **Best Recall**: 72.76% (Ultra Neural Network)
- **Dataset**: 5,000 real cybersecurity samples
- **Models**: 8 ultra-optimized algorithms with ensemble methods

## 🎨 Visualizations

The system automatically generates 7 comprehensive visualizations:

1. **Data Distribution Overview** - Network traffic patterns
2. **Attack Pattern Analysis** - Attack characteristics
3. **Feature Engineering Impact** - Preprocessing effects
4. **Model Performance Comparison** - Accuracy comparisons
5. **Preprocessing Pipeline** - Data transformation steps
6. **Real-time Detection Simulation** - Live monitoring simulation
7. **Comprehensive Dashboard** - Complete system overview

## 🛠️ Technical Features

- **Advanced Feature Engineering** with polynomial features
- **SMOTE Balancing** for optimal class distribution
- **Hyperparameter Optimization** for all models
- **Ensemble Methods** for enhanced performance
- **Real Dataset Processing** with automatic type detection
- **Production-Ready Models** saved as pickle files

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Ultra Random Forest | 92.64% | 89.69% | 70.73% | 79.09% |
| Ultra XGBoost | 92.64% | 90.10% | 70.33% | 79.00% |
| Ultra Gradient Boosting | 92.40% | 89.95% | 69.11% | 78.16% |
| Ultra Neural Network | 89.76% | 74.58% | 72.76% | 73.66% |

## 🔒 Security Applications

- **Network Intrusion Detection**
- **DoS/DDoS Attack Detection**
- **Port Scan Detection**
- **Malware Traffic Analysis**
- **Real-time Threat Monitoring**

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.