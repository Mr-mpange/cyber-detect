"""
Dataset Downloader for Cyber Attack Detection System
🎯 Download and prepare real cybersecurity datasets

This script helps download popular cybersecurity datasets from various sources
and prepares them for use with the cyber attack detection system.
"""

import os
import requests
import zipfile
import pandas as pd
from urllib.parse import urlparse
import argparse

def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✅ Created data/ directory")

def download_file(url, filename):
    """
    Download a file from URL with progress indication
    
    Args:
        url: URL to download from
        filename: Local filename to save to
    """
    print(f"📥 Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded {filename} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    print(f"📦 Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Extracted to {extract_to}")
        
        # Remove zip file after extraction
        os.remove(zip_path)
        print(f"🗑️  Removed {zip_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    print("\n🔹 NSL-KDD Dataset")
    print("-" * 40)
    
    create_data_directory()
    
    # NSL-KDD URLs (from official sources)
    urls = {
        'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
        'KDDTest+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
    }
    
    # Column names for NSL-KDD
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]
    
    for filename, url in urls.items():
        local_path = f"data/{filename}"
        
        if download_file(url, local_path):
            # Convert to CSV with proper headers
            try:
                df = pd.read_csv(local_path, header=None, names=column_names)
                csv_path = local_path.replace('.txt', '.csv')
                df.to_csv(csv_path, index=False)
                
                print(f"✅ Converted to CSV: {csv_path}")
                print(f"   Shape: {df.shape}")
                print(f"   Classes: {df['class'].unique()[:5]}...")
                
                # Remove original txt file
                os.remove(local_path)
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

def download_sample_datasets():
    """Download sample datasets for testing"""
    print("\n🔹 Sample Datasets")
    print("-" * 40)
    
    create_data_directory()
    
    # Create a sample dataset for immediate testing
    print("📊 Creating sample cybersecurity dataset...")
    
    import numpy as np
    np.random.seed(42)
    
    n_samples = 5000
    
    # Generate realistic network features
    data = {
        'duration': np.random.exponential(100, n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(800, n_samples),
        'count': np.random.poisson(10, n_samples),
        'srv_count': np.random.poisson(8, n_samples),
        'serror_rate': np.random.beta(1, 10, n_samples),
        'rerror_rate': np.random.beta(1, 15, n_samples),
        'same_srv_rate': np.random.beta(5, 2, n_samples),
        'diff_srv_rate': np.random.beta(1, 5, n_samples),
        'srv_diff_host_rate': np.random.beta(1, 8, n_samples),
        'dst_host_count': np.random.poisson(50, n_samples),
        'dst_host_srv_count': np.random.poisson(30, n_samples),
        'dst_host_same_srv_rate': np.random.beta(8, 2, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(1, 4, n_samples),
        'dst_host_same_src_port_rate': np.random.beta(3, 2, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet', 'other'], n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'SH'], n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Create labels (80% normal, 20% attacks)
    labels = np.random.choice(['normal', 'dos', 'probe', 'r2l', 'u2r'], 
                             n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02])
    df['class'] = labels
    
    # Add attack patterns
    attack_mask = df['class'] != 'normal'
    
    # DoS attacks: high connection counts
    dos_mask = df['class'] == 'dos'
    df.loc[dos_mask, 'count'] *= 10
    df.loc[dos_mask, 'srv_count'] *= 8
    
    # Probe attacks: high error rates
    probe_mask = df['class'] == 'probe'
    df.loc[probe_mask, 'serror_rate'] *= 5
    df.loc[probe_mask, 'rerror_rate'] *= 4
    
    # Save sample dataset
    sample_path = 'data/sample_network_intrusion.csv'
    df.to_csv(sample_path, index=False)
    
    print(f"✅ Created sample dataset: {sample_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Classes: {df['class'].value_counts().to_dict()}")

def show_dataset_info():
    """Show information about available datasets"""
    print("\n📊 AVAILABLE CYBERSECURITY DATASETS")
    print("="*60)
    
    datasets = {
        "NSL-KDD": {
            "description": "Network Security Laboratory - Knowledge Discovery Dataset",
            "size": "~5MB",
            "features": "41 network features",
            "classes": "normal, dos, probe, r2l, u2r",
            "samples": "~148,000 training + test",
            "download": "python download_datasets.py --nsl-kdd"
        },
        "Sample Dataset": {
            "description": "Generated sample for immediate testing",
            "size": "~1MB", 
            "features": "18 network features",
            "classes": "normal, dos, probe, r2l, u2r",
            "samples": "5,000",
            "download": "python download_datasets.py --sample"
        }
    }
    
    for name, info in datasets.items():
        print(f"\n🔹 {name}")
        for key, value in info.items():
            print(f"   {key.title()}: {value}")
    
    print(f"\n💡 USAGE EXAMPLES:")
    print("# Download NSL-KDD dataset")
    print("python download_datasets.py --nsl-kdd")
    print()
    print("# Create sample dataset for testing")
    print("python download_datasets.py --sample")
    print()
    print("# Use with main system")
    print("python main_kaggle.py --dataset data/KDDTrain+.csv")
    print("python main_kaggle.py --dataset data/sample_network_intrusion.csv")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download cybersecurity datasets')
    parser.add_argument('--nsl-kdd', action='store_true', help='Download NSL-KDD dataset')
    parser.add_argument('--sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--info', action='store_true', help='Show dataset information')
    
    args = parser.parse_args()
    
    print("🚀 CYBERSECURITY DATASET DOWNLOADER")
    print("="*50)
    
    if args.info or not any(vars(args).values()):
        show_dataset_info()
        return
    
    if args.nsl_kdd:
        download_nsl_kdd()
    
    if args.sample:
        download_sample_datasets()
    
    print(f"\n✅ Dataset download completed!")
    print(f"📁 Check the 'data/' directory for downloaded files")
    print(f"🚀 Use with: python main_kaggle.py --dataset data/your_dataset.csv")

if __name__ == "__main__":
    main()