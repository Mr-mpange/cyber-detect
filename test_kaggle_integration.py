"""
Test Script for Kaggle Dataset Integration
🧪 Verify that the real dataset functionality works correctly
"""

import os
import sys
import pandas as pd
import numpy as np

def test_sample_dataset():
    """Test with sample dataset creation and processing"""
    print("🧪 TESTING KAGGLE DATASET INTEGRATION")
    print("="*50)
    
    # Test 1: Create sample dataset
    print("\n1️⃣ Testing sample dataset creation...")
    
    try:
        # Run dataset downloader
        os.system("python download_datasets.py --sample")
        
        # Check if file was created
        sample_path = "data/sample_network_intrusion.csv"
        if os.path.exists(sample_path):
            print("✅ Sample dataset created successfully")
            
            # Load and check dataset
            df = pd.read_csv(sample_path)
            print(f"   Shape: {df.shape}")
            print(f"   Classes: {df['class'].value_counts().to_dict()}")
        else:
            print("❌ Sample dataset not created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating sample dataset: {e}")
        return False
    
    # Test 2: Test data loading
    print("\n2️⃣ Testing data loader...")
    
    try:
        sys.path.append('src')
        from data_loader import CyberDataLoader
        
        loader = CyberDataLoader()
        df, label_col = loader.load_kaggle_dataset(sample_path)
        
        print(f"✅ Data loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Label column: {label_col}")
        print(f"   Features: {len([col for col in df.columns if col != label_col])}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Test 3: Test main kaggle script (dry run)
    print("\n3️⃣ Testing main kaggle script...")
    
    try:
        # Test argument parsing
        import argparse
        from main_kaggle import KaggleDatasetProcessor
        
        processor = KaggleDatasetProcessor()
        print("✅ Kaggle processor initialized")
        
        # Test dataset detection
        from main_kaggle import detect_dataset_type
        dataset_type, label_col, df_sample = detect_dataset_type(sample_path)
        
        print(f"✅ Dataset detection works")
        print(f"   Detected type: {dataset_type}")
        print(f"   Label column: {label_col}")
        
    except Exception as e:
        print(f"❌ Error testing main script: {e}")
        return False
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n🚀 Ready to use real datasets!")
    print("\nNext steps:")
    print("1. python download_datasets.py --nsl-kdd")
    print("2. python main_kaggle.py --dataset data/KDDTrain+.csv")
    
    return True

def test_requirements():
    """Test if all required packages are installed"""
    print("\n🔍 CHECKING REQUIREMENTS")
    print("-" * 30)
    
    required_packages = [
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'), 
        ('sklearn', 'scikit-learn'), 
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'), 
        ('xgboost', 'xgboost'), 
        ('imblearn', 'imbalanced-learn'), 
        ('requests', 'requests')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 USAGE EXAMPLES")
    print("="*40)
    
    examples = [
        {
            "title": "Download NSL-KDD Dataset",
            "command": "python download_datasets.py --nsl-kdd",
            "description": "Downloads the NSL-KDD dataset (~5MB)"
        },
        {
            "title": "Create Sample Dataset", 
            "command": "python download_datasets.py --sample",
            "description": "Creates a sample dataset for testing"
        },
        {
            "title": "Run with NSL-KDD",
            "command": "python main_kaggle.py --dataset data/KDDTrain+.csv",
            "description": "Train models on NSL-KDD dataset"
        },
        {
            "title": "Run with Sample Dataset",
            "command": "python main_kaggle.py --dataset data/sample_network_intrusion.csv",
            "description": "Train models on sample dataset"
        },
        {
            "title": "Use Custom Label Column",
            "command": "python main_kaggle.py --dataset data/custom.csv --label attack_type",
            "description": "Specify custom label column name"
        },
        {
            "title": "Use Subset of Large Dataset",
            "command": "python main_kaggle.py --dataset data/large.csv --sample 10000",
            "description": "Use only first 10,000 rows"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")

def main():
    """Main test function"""
    print("🚀 KAGGLE DATASET INTEGRATION TEST SUITE")
    print("="*60)
    
    # Test requirements
    if not test_requirements():
        print("\n❌ Requirements test failed. Please install missing packages.")
        return
    
    # Test sample dataset functionality
    if not test_sample_dataset():
        print("\n❌ Sample dataset test failed.")
        return
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "="*60)
    print("🎉 KAGGLE INTEGRATION READY!")
    print("="*60)
    print("\n📋 Quick Start:")
    print("1. python download_datasets.py --sample")
    print("2. python main_kaggle.py --dataset data/sample_network_intrusion.csv")
    print("\n🔗 For more info: See KAGGLE_USAGE.md")

if __name__ == "__main__":
    main()