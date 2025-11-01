"""
Data Preparation Script for IDS Project
Prepares CICIDS 2017, NSL-KDD, and UNSW-NB15 datasets
Works with folder structure: data/cicids2017/, data/nslkdd/, data/unsw-nb15/
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DatasetPreparator:
    """Prepare and standardize multiple IDS datasets"""
    
    def __init__(self):
        self.universal_features = [
            'flow_duration',
            'total_fwd_packets',
            'total_bwd_packets',
            'total_fwd_bytes',
            'total_bwd_bytes',
            'packet_length_mean',
            'packet_length_std',
            'packet_length_min',
            'packet_length_max',
            'flow_bytes_per_sec',
            'flow_packets_per_sec',
            'fwd_bwd_packet_ratio',
            'fwd_bwd_byte_ratio',
            'iat_mean',
            'iat_std',
            'iat_max',
            'iat_min',
            'protocol',
            'avg_fwd_packet_size',
            'avg_bwd_packet_size',
            'total_packets',
            'total_bytes'
        ]
    
    def prepare_cicids2017(self, file_path):
        """
        Prepare CICIDS 2017 dataset
        Expected folder: data/cicids2017/
        """
        print(f"Loading CICIDS 2017 from {file_path}...")
        
        try:
            # Read CSV with various encodings
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            print(f"  Original shape: {df.shape}")
            
            # Remove spaces from column names
            df.columns = df.columns.str.strip()
            
            # Feature mapping for CICIDS 2017
            feature_map = {
                'Flow Duration': 'flow_duration',
                'Total Fwd Packets': 'total_fwd_packets',
                'Total Backward Packets': 'total_bwd_packets',
                'Total Length of Fwd Packets': 'total_fwd_bytes',
                'Total Length of Bwd Packets': 'total_bwd_bytes',
                'Fwd Packet Length Mean': 'packet_length_mean',
                'Fwd Packet Length Std': 'packet_length_std',
                'Fwd Packet Length Min': 'packet_length_min',
                'Fwd Packet Length Max': 'packet_length_max',
                'Flow Bytes/s': 'flow_bytes_per_sec',
                'Flow Packets/s': 'flow_packets_per_sec',
                'Flow IAT Mean': 'iat_mean',
                'Flow IAT Std': 'iat_std',
                'Flow IAT Max': 'iat_max',
                'Flow IAT Min': 'iat_min',
                'Protocol': 'protocol',
                'Label': 'label',
                ' Label': 'label'  # Sometimes there's a space
            }
            
            # Rename columns that exist
            rename_dict = {k: v for k, v in feature_map.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Calculate missing features
            if 'total_packets' not in df.columns:
                df['total_packets'] = df['total_fwd_packets'] + df['total_bwd_packets']
            
            if 'total_bytes' not in df.columns:
                df['total_bytes'] = df['total_fwd_bytes'] + df['total_bwd_bytes']
            
            if 'fwd_bwd_packet_ratio' not in df.columns:
                df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1)
            
            if 'fwd_bwd_byte_ratio' not in df.columns:
                df['fwd_bwd_byte_ratio'] = df['total_fwd_bytes'] / (df['total_bwd_bytes'] + 1)
            
            if 'avg_fwd_packet_size' not in df.columns:
                df['avg_fwd_packet_size'] = df['total_fwd_bytes'] / (df['total_fwd_packets'] + 1)
            
            if 'avg_bwd_packet_size' not in df.columns:
                df['avg_bwd_packet_size'] = df['total_bwd_bytes'] / (df['total_bwd_packets'] + 1)
            
            # Standardize labels
            df = self.standardize_labels(df, 'cicids')
            
            # Keep only universal features + label
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            
            # Clean data
            df = self.clean_data(df)
            
            print(f"  Processed: {df.shape}")
            
            return df
        
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            return None
    
    def prepare_nslkdd(self, file_path):
        """
        Prepare NSL-KDD dataset
        Expected folder: data/nslkdd/
        """
        print(f"Loading NSL-KDD from {file_path}...")
        
        try:
            # NSL-KDD column names
            columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'label', 'difficulty'
            ]
            
            df = pd.read_csv(file_path, names=columns)
            
            print(f"  Original shape: {df.shape}")
            
            # Map to universal features
            df['flow_duration'] = df['duration']
            df['total_fwd_bytes'] = df['src_bytes']
            df['total_bwd_bytes'] = df['dst_bytes']
            df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
            df['total_packets'] = df['count']
            df['total_fwd_packets'] = df['count'] * 0.5
            df['total_bwd_packets'] = df['count'] * 0.5
            
            # Calculate rates
            df['flow_bytes_per_sec'] = df['total_bytes'] / (df['duration'] + 1)
            df['flow_packets_per_sec'] = df['count'] / (df['duration'] + 1)
            
            # Packet statistics (estimated)
            df['packet_length_mean'] = df['total_bytes'] / (df['count'] + 1)
            df['packet_length_std'] = df['packet_length_mean'] * 0.3
            df['packet_length_min'] = df['packet_length_mean'] * 0.5
            df['packet_length_max'] = df['packet_length_mean'] * 2
            
            # Ratios
            df['fwd_bwd_packet_ratio'] = 1.0
            df['fwd_bwd_byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            df['avg_fwd_packet_size'] = df['src_bytes'] / (df['total_fwd_packets'] + 1)
            df['avg_bwd_packet_size'] = df['dst_bytes'] / (df['total_bwd_packets'] + 1)
            
            # IAT
            df['iat_mean'] = df['duration'] / (df['count'] + 1)
            df['iat_std'] = df['iat_mean'] * 0.5
            df['iat_max'] = df['iat_mean'] * 2
            df['iat_min'] = df['iat_mean'] * 0.1
            
            # Protocol encoding
            protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
            df['protocol'] = df['protocol_type'].map(protocol_map).fillna(0)
            
            # Standardize labels
            df = self.standardize_labels(df, 'nslkdd')
            
            # Keep only universal features + label
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            
            # Clean data
            df = self.clean_data(df)
            
            print(f"  Processed: {df.shape}")
            
            return df
        
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            return None
    
    def prepare_unsw_nb15(self, file_path):
        """
        Prepare UNSW-NB15 dataset
        Expected folder: data/unsw-nb15/
        """
        print(f"Loading UNSW-NB15 from {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
            
            print(f"  Original shape: {df.shape}")
            
            # Feature mapping for UNSW-NB15
            feature_map = {
                'dur': 'flow_duration',
                'spkts': 'total_fwd_packets',
                'dpkts': 'total_bwd_packets',
                'sbytes': 'total_fwd_bytes',
                'dbytes': 'total_bwd_bytes',
                'rate': 'flow_packets_per_sec',
                'smean': 'packet_length_mean',
                'proto': 'protocol'
            }
            
            # Rename columns
            rename_dict = {k: v for k, v in feature_map.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Handle label column - use attack_cat if available, otherwise use label
            if 'attack_cat' in df.columns:
                df['label'] = df['attack_cat'].astype(str)
            elif 'label' in df.columns:
                # If numeric label, keep as is for now
                df['label'] = df['label'].astype(str)
            
            # Calculate missing features
            if 'total_packets' not in df.columns:
                df['total_packets'] = df['total_fwd_packets'] + df['total_bwd_packets']
            
            if 'total_bytes' not in df.columns:
                df['total_bytes'] = df['total_fwd_bytes'] + df['total_bwd_bytes']
            
            if 'flow_bytes_per_sec' not in df.columns:
                df['flow_bytes_per_sec'] = df['total_bytes'] / (df['flow_duration'] + 0.001)
            
            if 'fwd_bwd_packet_ratio' not in df.columns:
                df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1)
            
            if 'fwd_bwd_byte_ratio' not in df.columns:
                df['fwd_bwd_byte_ratio'] = df['total_fwd_bytes'] / (df['total_bwd_bytes'] + 1)
            
            if 'avg_fwd_packet_size' not in df.columns:
                df['avg_fwd_packet_size'] = df['total_fwd_bytes'] / (df['total_fwd_packets'] + 1)
            
            if 'avg_bwd_packet_size' not in df.columns:
                df['avg_bwd_packet_size'] = df['total_bwd_bytes'] / (df['total_bwd_packets'] + 1)
            
            # Estimate packet statistics
            if 'packet_length_std' not in df.columns:
                df['packet_length_std'] = df['packet_length_mean'] * 0.3
            
            if 'packet_length_min' not in df.columns:
                df['packet_length_min'] = df['packet_length_mean'] * 0.3
            
            if 'packet_length_max' not in df.columns:
                df['packet_length_max'] = df['packet_length_mean'] * 2
            
            # Estimate IAT
            df['iat_mean'] = df['flow_duration'] / (df['total_packets'] + 1)
            df['iat_std'] = df['iat_mean'] * 0.5
            df['iat_max'] = df['iat_mean'] * 3
            df['iat_min'] = df['iat_mean'] * 0.1
            
            # Standardize labels
            df = self.standardize_labels(df, 'unsw')
            
            # Keep only universal features + label
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            
            # Clean data
            df = self.clean_data(df)
            
            print(f"  Processed: {df.shape}")
            
            return df
        
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            return None
    
    def standardize_labels(self, df, dataset_type):
        """Standardize attack labels across datasets"""
        
        label_mapping = {
            # Normal traffic
            'BENIGN': 'Normal',
            'normal': 'Normal',
            'Normal': 'Normal',
            '0': 'Normal',  # For numeric labels
            
            # DDoS/DoS attacks
            'DDoS': 'DDoS',
            'ddos': 'DDoS',
            'DoS': 'DDoS',
            'DoS Hulk': 'DDoS',
            'DoS GoldenEye': 'DDoS',
            'DoS slowloris': 'DDoS',
            'DoS Slowhttptest': 'DDoS',
            'neptune': 'DDoS',
            'smurf': 'DDoS',
            'back': 'DDoS',
            'teardrop': 'DDoS',
            'pod': 'DDoS',
            'land': 'DDoS',
            
            # Port Scan
            'PortScan': 'Port Scan',
            'portsweep': 'Port Scan',
            'nmap': 'Port Scan',
            'ipsweep': 'Port Scan',
            'satan': 'Port Scan',
            'mscan': 'Port Scan',
            'Analysis': 'Port Scan',
            'Reconnaissance': 'Port Scan',
            
            # Brute Force
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'guess_passwd': 'Brute Force',
            'Fuzzers': 'Brute Force',
            
            # Bot/Botnet
            'Bot': 'Botnet',
            'Botnet': 'Botnet',
            
            # Web Attacks
            'Web Attack – Brute Force': 'Web Attack',
            'Web Attack – XSS': 'Web Attack',
            'Web Attack – Sql Injection': 'Web Attack',
            'Web Attack Brute Force': 'Web Attack',
            'Web Attack XSS': 'Web Attack',
            'Web Attack Sql Injection': 'Web Attack',
            'XSS': 'Web Attack',
            'SQL Injection': 'Web Attack',
            
            # Exploits
            'Infiltration': 'Exploit',
            'Heartbleed': 'Exploit',
            'buffer_overflow': 'Exploit',
            'rootkit': 'Exploit',
            'loadmodule': 'Exploit',
            'perl': 'Exploit',
            'Exploits': 'Exploit',
            'Backdoor': 'Exploit',
            'Backdoors': 'Exploit',
            
            # Worms
            'Worms': 'Worm',
            
            # Shellcode
            'Shellcode': 'Shellcode',
            
            # Generic
            'Generic': 'Other',
            '1': 'Other'  # For numeric labels
        }
        
        if 'label' in df.columns:
            # Ensure we have a Series, not a DataFrame
            if isinstance(df['label'], pd.Series):
                df['label'] = df['label'].astype(str).str.strip()
                df['label'] = df['label'].map(label_mapping).fillna('Other')
            else:
                # If somehow it's not a Series, convert it properly
                df['label'] = pd.Series(df['label']).astype(str).str.strip()
                df['label'] = df['label'].map(label_mapping).fillna('Other')
        
        return df
    
    def clean_data(self, df):
        """Clean and prepare data"""
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(0)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Remove negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'label':
                df[col] = df[col].clip(lower=0)
        
        return df
    
    def balance_dataset(self, df, max_samples_per_class=50000):
        """Balance dataset by limiting samples per class"""
        
        balanced_dfs = []
        
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            if len(label_df) > max_samples_per_class:
                label_df = label_df.sample(n=max_samples_per_class, random_state=42)
            balanced_dfs.append(label_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)


def main():
    """Main data preparation pipeline"""
    
    print("=" * 70)
    print("IDS DATA PREPARATION PIPELINE")
    print("=" * 70)
    
    preparator = DatasetPreparator()
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    all_datasets = []
    
    # ========== PROCESS CICIDS 2017 ==========
    print("\n[1/3] Processing CICIDS 2017...")
    print("-" * 70)
    
    cicids_folder = 'data/cicids2017'
    if os.path.exists(cicids_folder):
        # Find all CSV files in the folder
        cicids_files = glob.glob(os.path.join(cicids_folder, '*.csv'))
        print(f"Found {len(cicids_files)} CSV files in {cicids_folder}")
        
        cicids_dfs = []
        for file in cicids_files:
            print(f"\nProcessing: {os.path.basename(file)}")
            df = preparator.prepare_cicids2017(file)
            if df is not None:
                cicids_dfs.append(df)
        
        if cicids_dfs:
            cicids_combined = pd.concat(cicids_dfs, ignore_index=True)
            cicids_combined = preparator.balance_dataset(cicids_combined)
            cicids_combined.to_csv('data/processed/cicids_processed.csv', index=False)
            all_datasets.append(cicids_combined)
            print(f"\n✅ CICIDS 2017 Combined: {cicids_combined.shape}")
            print(f"   Labels: {cicids_combined['label'].value_counts().to_dict()}")
    else:
        print(f"⚠️  Folder not found: {cicids_folder}")
        print(f"   Please create folder and add CICIDS CSV files")
    
    # ========== PROCESS NSL-KDD ==========
    print("\n[2/3] Processing NSL-KDD...")
    print("-" * 70)
    
    nslkdd_folder = 'data/nslkdd'
    if os.path.exists(nslkdd_folder):
        # Find train and test files
        nslkdd_files = glob.glob(os.path.join(nslkdd_folder, '*.txt'))
        nslkdd_files.extend(glob.glob(os.path.join(nslkdd_folder, '*.csv')))
        print(f"Found {len(nslkdd_files)} files in {nslkdd_folder}")
        
        nslkdd_dfs = []
        for file in nslkdd_files:
            print(f"\nProcessing: {os.path.basename(file)}")
            df = preparator.prepare_nslkdd(file)
            if df is not None:
                nslkdd_dfs.append(df)
        
        if nslkdd_dfs:
            nslkdd_combined = pd.concat(nslkdd_dfs, ignore_index=True)
            nslkdd_combined = preparator.balance_dataset(nslkdd_combined)
            nslkdd_combined.to_csv('data/processed/nslkdd_processed.csv', index=False)
            all_datasets.append(nslkdd_combined)
            print(f"\n✅ NSL-KDD Combined: {nslkdd_combined.shape}")
            print(f"   Labels: {nslkdd_combined['label'].value_counts().to_dict()}")
    else:
        print(f"⚠️  Folder not found: {nslkdd_folder}")
        print(f"   Please create folder and add NSL-KDD files")
    
    # ========== PROCESS UNSW-NB15 ==========
    print("\n[3/3] Processing UNSW-NB15...")
    print("-" * 70)
    
    unsw_folder = 'data/unsw-nb15'
    if os.path.exists(unsw_folder):
        # Find all CSV files (including training and testing sets)
        unsw_files = glob.glob(os.path.join(unsw_folder, '*.csv'))
        
        # Also look for specific naming patterns
        possible_names = [
            'UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv',
            'training set.csv', 'testing set.csv', 
            'UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv',
            'training_set.csv', 'testing_set.csv'
        ]
        
        for name in possible_names:
            file_path = os.path.join(unsw_folder, name)
            if os.path.exists(file_path) and file_path not in unsw_files:
                unsw_files.append(file_path)
        
        print(f"Found {len(unsw_files)} CSV files in {unsw_folder}")
        
        unsw_dfs = []
        for file in unsw_files:
            print(f"\nProcessing: {os.path.basename(file)}")
            df = preparator.prepare_unsw_nb15(file)
            if df is not None:
                unsw_dfs.append(df)
        
        if unsw_dfs:
            unsw_combined = pd.concat(unsw_dfs, ignore_index=True)
            unsw_combined = preparator.balance_dataset(unsw_combined)
            unsw_combined.to_csv('data/processed/unsw_processed.csv', index=False)
            all_datasets.append(unsw_combined)
            print(f"\n✅ UNSW-NB15 Combined: {unsw_combined.shape}")
            print(f"   Labels: {unsw_combined['label'].value_counts().to_dict()}")
    else:
        print(f"⚠️  Folder not found: {unsw_folder}")
        print(f"   Please create folder and add UNSW-NB15 CSV files")
    
    # ========== COMBINE ALL DATASETS ==========
    if all_datasets:
        print("\n" + "=" * 70)
        print("COMBINING ALL DATASETS")
        print("=" * 70)
        
        combined_dataset = pd.concat(all_datasets, ignore_index=True)
        combined_dataset = preparator.balance_dataset(combined_dataset, max_samples_per_class=30000)
        
        # Shuffle
        combined_dataset = combined_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_dataset.to_csv('data/processed/combined_dataset.csv', index=False)
        
        print(f"\n✅ Combined dataset created: {combined_dataset.shape}")
        print(f"\nLabel distribution:")
        print(combined_dataset['label'].value_counts())
        
        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE!")
        print("=" * 70)
        print("\nFiles created:")
        print("  ✅ data/processed/cicids_processed.csv")
        print("  ✅ data/processed/nslkdd_processed.csv")
        print("  ✅ data/processed/unsw_processed.csv")
        print("  ✅ data/processed/combined_dataset.csv")
        print("\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(combined_dataset):,}")
        print(f"   Total features: {len(combined_dataset.columns)-1}")
        print(f"   Attack classes: {combined_dataset['label'].nunique()}")
        print("\n🚀 Next step: Run 'python train_model.py' to train models")
    
    else:
        print("\n" + "=" * 70)
        print("⚠️  NO DATASETS WERE PROCESSED")
        print("=" * 70)
        print("\nExpected folder structure:")
        print("data/")
        print("  ├── cicids2017/")
        print("  │   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        print("  │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        print("  │   └── *.csv")
        print("  ├── nslkdd/")
        print("  │   ├── KDDTrain+.txt")
        print("  │   ├── KDDTest+.txt")
        print("  │   └── *.csv")
        print("  └── unsw-nb15/")
        print("      ├── UNSW-NB15_1.csv")
        print("      ├── UNSW-NB15_2.csv")
        print("      └── *.csv")
        print("\nPlease:")
        print("  1. Create the folders: data/cicids2017, data/nslkdd, data/unsw-nb15")
        print("  2. Download datasets and place files in respective folders")
        print("  3. Run this script again")


if __name__ == "__main__":
    main()