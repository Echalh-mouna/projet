"""
Data Preparation Script for IDS Project
Prepares CICIDS 2017, NSL-KDD, and UNSW-NB15 datasets
MODIFICATION: AJOUT DE 3 FEATURES DE DENSITÉ POUR AMÉLIORER LE ML/XGBOOST.
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 🌟 MODIFICATION CLÉ 🌟: La liste est enrichie (25 features de base)
UNIVERSAL_FEATURES = [
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
    'total_bytes',
    # NOUVELLES FEATURES DE DENSITÉ CRITIQUES
    'total_bytes_per_packet',
    'duration_per_packet',
    'iat_std_to_mean_ratio'
] # Total de 25 features de base

class DatasetPreparator:
    """Prepare and standardize multiple IDS datasets"""
    
    def __init__(self):
        self.universal_features = UNIVERSAL_FEATURES
    
    def calculate_new_density_features(self, df):
        """Calcule les 3 nouvelles features de densité"""
        
        # Évite la division par zéro
        pkt_divisor = df['total_packets'].replace(0, 1)
        iat_mean_divisor = df['iat_mean'].replace(0, 1e-6) 

        df['total_bytes_per_packet'] = df['total_bytes'] / pkt_divisor
        df['duration_per_packet'] = df['flow_duration'] / pkt_divisor
        df['iat_std_to_mean_ratio'] = df['iat_std'] / iat_mean_divisor
        
        return df

    def prepare_cicids2017(self, file_path):
        """Prepare CICIDS 2017 dataset"""
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            df.columns = df.columns.str.strip()
            feature_map = {
                'Flow Duration': 'flow_duration', 'Total Fwd Packets': 'total_fwd_packets', 'Total Backward Packets': 'total_bwd_packets', 
                'Total Length of Fwd Packets': 'total_fwd_bytes', 'Total Length of Bwd Packets': 'total_bwd_bytes', 
                'Fwd Packet Length Mean': 'packet_length_mean', 'Fwd Packet Length Std': 'packet_length_std', 
                'Fwd Packet Length Min': 'packet_length_min', 'Fwd Packet Length Max': 'packet_length_max', 
                'Flow Bytes/s': 'flow_bytes_per_sec', 'Flow Packets/s': 'flow_packets_per_sec', 'Flow IAT Mean': 'iat_mean', 
                'Flow IAT Std': 'iat_std', 'Flow IAT Max': 'iat_max', 'Flow IAT Min': 'iat_min', 'Protocol': 'protocol', 
                'Label': 'label', ' Label': 'label'
            }
            rename_dict = {k: v for k, v in feature_map.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            df['total_packets'] = df['total_fwd_packets'] + df['total_bwd_packets']
            df['total_bytes'] = df['total_fwd_bytes'] + df['total_bwd_bytes']
            df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1)
            df['fwd_bwd_byte_ratio'] = df['total_fwd_bytes'] / (df['total_bwd_bytes'] + 1)
            df['avg_fwd_packet_size'] = df['total_fwd_bytes'] / (df['total_fwd_packets'] + 1)
            df['avg_bwd_packet_size'] = df['total_bwd_bytes'] / (df['total_bwd_packets'] + 1)
            
            df = self.calculate_new_density_features(df) # 🌟 NOUVEAU CALCUL
            
            df = self.standardize_labels(df, 'cicids')
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            return self.clean_data(df)
        
        except Exception: return None

    def prepare_nslkdd(self, file_path):
        """Prepare NSL-KDD dataset"""
        try:
            columns = [
                'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
            ]
            df = pd.read_csv(file_path, names=columns)
            
            df['flow_duration'] = df['duration']; df['total_fwd_bytes'] = df['src_bytes']; df['total_bwd_bytes'] = df['dst_bytes']
            df['total_bytes'] = df['src_bytes'] + df['dst_bytes']; df['total_packets'] = df['count']
            df['total_fwd_packets'] = df['count'] * 0.5; df['total_bwd_packets'] = df['count'] * 0.5
            df['flow_bytes_per_sec'] = df['total_bytes'] / (df['duration'] + 1); df['flow_packets_per_sec'] = df['count'] / (df['duration'] + 1)
            df['packet_length_mean'] = df['total_bytes'] / (df['count'] + 1); df['packet_length_std'] = df['packet_length_mean'] * 0.3
            df['packet_length_min'] = df['packet_length_mean'] * 0.5; df['packet_length_max'] = df['packet_length_mean'] * 2
            df['fwd_bwd_packet_ratio'] = 1.0; df['fwd_bwd_byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            df['avg_fwd_packet_size'] = df['src_bytes'] / (df['total_fwd_packets'] + 1); df['avg_bwd_packet_size'] = df['dst_bytes'] / (df['total_bwd_packets'] + 1)
            df['iat_mean'] = df['duration'] / (df['count'] + 1); df['iat_std'] = df['iat_mean'] * 0.5; df['iat_max'] = df['iat_mean'] * 2; df['iat_min'] = df['iat_mean'] * 0.1
            protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
            df['protocol'] = df['protocol_type'].map(protocol_map).fillna(0)

            df = self.calculate_new_density_features(df) # 🌟 NOUVEAU CALCUL

            df = self.standardize_labels(df, 'nslkdd')
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            return self.clean_data(df)
        except Exception: return None

    def prepare_unsw_nb15(self, file_path):
        """Prepare UNSW-NB15 dataset"""
        try:
            df = pd.read_csv(file_path)
            feature_map = {
                'dur': 'flow_duration', 'spkts': 'total_fwd_packets', 'dpkts': 'total_bwd_packets', 
                'sbytes': 'total_fwd_bytes', 'dbytes': 'total_bwd_bytes', 'rate': 'flow_packets_per_sec', 
                'smean': 'packet_length_mean', 'proto': 'protocol'
            }
            rename_dict = {k: v for k, v in feature_map.items() if k in df.columns}
            df = df.rename(columns=rename_dict)
            if 'attack_cat' in df.columns: df['label'] = df['attack_cat'].astype(str)
            elif 'label' in df.columns: df['label'] = df['label'].astype(str)
            
            df['total_packets'] = df['total_fwd_packets'] + df['total_bwd_packets']
            df['total_bytes'] = df['total_fwd_bytes'] + df['total_bwd_bytes']
            df['flow_bytes_per_sec'] = df['total_bytes'] / (df['flow_duration'] + 0.001)
            df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (df['total_bwd_packets'] + 1)
            df['fwd_bwd_byte_ratio'] = df['total_fwd_bytes'] / (df['total_bwd_bytes'] + 1)
            df['avg_fwd_packet_size'] = df['total_fwd_bytes'] / (df['total_fwd_packets'] + 1)
            df['avg_bwd_packet_size'] = df['total_bwd_bytes'] / (df['total_bwd_packets'] + 1)
            df['packet_length_std'] = df['packet_length_mean'] * 0.3
            df['packet_length_min'] = df['packet_length_mean'] * 0.3; df['packet_length_max'] = df['packet_length_mean'] * 2
            df['iat_mean'] = df['flow_duration'] / (df['total_packets'] + 1); df['iat_std'] = df['iat_mean'] * 0.5; df['iat_max'] = df['iat_mean'] * 3; df['iat_min'] = df['iat_mean'] * 0.1
            
            df = self.calculate_new_density_features(df) # 🌟 NOUVEAU CALCUL

            df = self.standardize_labels(df, 'unsw')
            features_to_keep = [f for f in self.universal_features if f in df.columns] + ['label']
            df = df[features_to_keep]
            return self.clean_data(df)
        except Exception: return None

    # Reste des méthodes standardize_labels, clean_data, balance_dataset et main() sont inchangées
    def standardize_labels(self, df, dataset_type):
        label_mapping = {
            'BENIGN': 'Normal', 'normal': 'Normal', 'Normal': 'Normal', '0': 'Normal', 
            'DDoS': 'DDoS', 'ddos': 'DDoS', 'DoS': 'DDoS', 'DoS Hulk': 'DDoS', 'DoS GoldenEye': 'DDoS', 'DoS slowloris': 'DDoS', 'DoS Slowhttptest': 'DDoS', 'neptune': 'DDoS', 'smurf': 'DDoS', 'back': 'DDoS', 'teardrop': 'DDoS', 'pod': 'DDoS', 'land': 'DDoS', 
            'PortScan': 'Port Scan', 'portsweep': 'Port Scan', 'nmap': 'Port Scan', 'ipsweep': 'Port Scan', 'satan': 'Port Scan', 'mscan': 'Port Scan', 'Analysis': 'Port Scan', 'Reconnaissance': 'Port Scan', 
            'FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force', 'guess_passwd': 'Brute Force', 'Fuzzers': 'Brute Force', 
            'Bot': 'Botnet', 'Botnet': 'Botnet', 
            'Web Attack – Brute Force': 'Web Attack', 'Web Attack – XSS': 'Web Attack', 'Web Attack – Sql Injection': 'Web Attack', 'Web Attack Brute Force': 'Web Attack', 'Web Attack XSS': 'Web Attack', 'Web Attack Sql Injection': 'Web Attack', 
            'Infiltration': 'Exploit', 'Heartbleed': 'Exploit', 'buffer_overflow': 'Exploit', 'rootkit': 'Exploit', 'loadmodule': 'Exploit', 'perl': 'Exploit', 'Exploits': 'Exploit', 'Backdoor': 'Exploit', 'Backdoors': 'Exploit', 
            'Worms': 'Worm', 
            'Shellcode': 'Shellcode', 
            'Generic': 'Other', '1': 'Other'
        }
        if 'label' in df.columns:
            if isinstance(df['label'], pd.Series):
                df['label'] = df['label'].astype(str).str.strip()
                df['label'] = df['label'].map(label_mapping).fillna('Other')
            else:
                df['label'] = pd.Series(df['label']).astype(str).str.strip()
                df['label'] = df['label'].map(label_mapping).fillna('Other')
        return df
    
    def clean_data(self, df):
        df = df.drop_duplicates()
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'label':
                df[col] = df[col].clip(lower=0)
        return df
    
    def balance_dataset(self, df, max_samples_per_class=50000):
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
        cicids_files = glob.glob(os.path.join(cicids_folder, '*.csv'))
        print(f"Found {len(cicids_files)} CSV files in {cicids_folder}")
        cicids_dfs = []
        for file in cicids_files:
            df = preparator.prepare_cicids2017(file)
            if df is not None: cicids_dfs.append(df)
        if cicids_dfs: all_datasets.append(pd.concat(cicids_dfs, ignore_index=True))

    # ========== PROCESS NSL-KDD ==========
    print("\n[2/3] Processing NSL-KDD...")
    print("-" * 70)
    
    nslkdd_folder = 'data/nslkdd'
    if os.path.exists(nslkdd_folder):
        nslkdd_files = glob.glob(os.path.join(nslkdd_folder, '*.txt')) + glob.glob(os.path.join(nslkdd_folder, '*.csv'))
        print(f"Found {len(nslkdd_files)} files in {nslkdd_folder}")
        nslkdd_dfs = []
        for file in nslkdd_files:
            df = preparator.prepare_nslkdd(file)
            if df is not None: nslkdd_dfs.append(df)
        if nslkdd_dfs: all_datasets.append(pd.concat(nslkdd_dfs, ignore_index=True))

    # ========== PROCESS UNSW-NB15 ==========
    print("\n[3/3] Processing UNSW-NB15...")
    print("-" * 70)
    
    unsw_folder = 'data/unsw-nb15'
    if os.path.exists(unsw_folder):
        unsw_files = glob.glob(os.path.join(unsw_folder, '*.csv'))
        print(f"Found {len(unsw_files)} CSV files in {unsw_folder}")
        unsw_dfs = []
        for file in unsw_files:
            df = preparator.prepare_unsw_nb15(file)
            if df is not None: unsw_dfs.append(df)
        if unsw_dfs: all_datasets.append(pd.concat(unsw_dfs, ignore_index=True))
    
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
    
    else:
        print("\n" + "=" * 70)
        print("⚠️  NO DATASETS WERE PROCESSED")
        print("=" * 70)
        
if __name__ == "__main__":
    main()