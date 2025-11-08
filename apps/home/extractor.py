import os
import pandas as pd
import numpy as np
import joblib
import traceback
from scapy.all import rdpcap, IP, TCP, UDP
from tensorflow.keras.models import load_model
from apps.home.custom_models import PrefitVotingEnsemble
import importlib.util, sys, os
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

# =============================================
# CONFIGURATION DES CHEMINS
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../models"))

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

# ✅ Mapping correct entre menu et fichiers modèles
MODEL_MAP = {
    "ensemble": "ensemble_model.pkl",
    "random_forest": "rf_model.pkl",
    "rf": "rf_model.pkl",
    "xgboost": "xgb_model.pkl",
    "xgb": "xgb_model.pkl",
    "dnn": "dnn_model.keras",
    "cnn": "cnn_model.keras"
}



def ensure_custom_classes_available():
    """Charge dynamiquement les classes personnalisées pour joblib.load."""
    try:
        train_path = os.path.join(os.getcwd(), 'train_model_hybride_fixed.py')
        if os.path.exists(train_path):
            name = 'user_mod_trainer'
            spec = importlib.util.spec_from_file_location(name, train_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for cls_name in ('KerasSklearnClassifier', 'SoftAverager', 'PrefitVotingEnsemble'):
                    if hasattr(mod, cls_name):
                        setattr(sys.modules['__main__'], cls_name, getattr(mod, cls_name))
    except Exception as e:
        print(f"[⚠️] Impossible de charger les classes personnalisées : {e}")


# =============================================
# EXTRACTION DES FEATURES À PARTIR DU PCAP
# =============================================
def extract_features_from_pcap(pcap_path):
    try:
        packets = rdpcap(pcap_path)
        flows = {}

        for pkt in packets:
            if IP in pkt:
                proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
                src, dst = pkt[IP].src, pkt[IP].dst
                sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
                dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)
                key = (src, dst, sport, dport, proto)

                time = float(pkt.time)
                length = len(pkt)

                if key not in flows:
                    flows[key] = {"times": [], "lengths": []}
                flows[key]["times"].append(time)
                flows[key]["lengths"].append(length)

        features = []
        for key, data in flows.items():
            times = np.array(data["times"], dtype=float)
            lengths = np.array(data["lengths"], dtype=float)
            iats = np.diff(np.sort(times)) if len(times) > 1 else [0.0]

            feat = {
                "src_ip": key[0],
                "dst_ip": key[1],
                "src_port": key[2],
                "dst_port": key[3],
                "protocol": 6 if key[4] == "TCP" else (17 if key[4] == "UDP" else 0),
                "flow_duration": float(max(times) - min(times)) if len(times) > 1 else 0.0,
                "packet_count": len(lengths),
                "packet_rate": len(lengths) / (max(times) - min(times) + 1e-6),
                "total_length": float(np.sum(lengths)),
                "avg_packet_size": float(np.mean(lengths)),
                "min_packet_size": float(np.min(lengths)),
                "max_packet_size": float(np.max(lengths)),
                "std_packet_size": float(np.std(lengths)),
                "iat_mean": float(np.mean(iats)),
                "iat_std": float(np.std(iats)),
                "iat_max": float(np.max(iats)),
                "iat_min": float(np.min(iats)),
            }
            features.append(feat)

        df = pd.DataFrame(features)
        print(f"[✅] Extraction réussie : {len(df)} flux détectés dans {os.path.basename(pcap_path)}.")
        return df

    except Exception as e:
        print(f"[❌] Erreur d’extraction PCAP : {e}")
        traceback.print_exc()
        return None

# =============================================
# PREDICTION AVEC LE MODÈLE CHOISI (VERSION FINALE)
# =============================================
def predict_with_model(df, model_type="ensemble"):
    ensure_custom_classes_available()

    try:
        model_type = model_type.lower()
        model_file = MODEL_MAP.get(model_type)

        if not model_file:
            return {"error": f"Modèle inconnu : {model_type}"}

        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            return {"error": f"Modèle {model_file} introuvable dans {MODELS_DIR}"}

        # Chargement scaler, encodeur et noms de features
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        feature_names = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else df.columns.tolist()

        # Prétraitement : on garde uniquement les colonnes numériques
        df = df.select_dtypes(include=[np.number])
        df = df.reindex(columns=feature_names, fill_value=0)

        # Application du scaler
        if scaler:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values

        # Chargement du modèle
        if model_file.endswith(".pkl"):
            setattr(__import__('__main__'), 'PrefitVotingEnsemble', PrefitVotingEnsemble)
            model = joblib.load(model_path, mmap_mode=None)
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(df_scaled)
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                y_pred = model.predict(df_scaled)
                y_pred_prob = None

        elif model_file.endswith(".keras"):
            model = load_model(model_path)
            y_pred_prob = model.predict(df_scaled)
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:
            return {"error": f"Format de modèle non supporté : {model_file}"}


        # Si label encoder existe, conversion vers les vraies classes
        if encoder is not None:
            try:
                y_pred = encoder.inverse_transform(y_pred)
            except Exception:
                y_pred = [str(x) for x in y_pred]
        else:
            y_pred = [str(x) for x in y_pred]

        # Comptage des classes
        counts = Counter(y_pred)
        predicted_classes = {str(k): int(v) for k, v in counts.items()}

        # Moyenne des probabilités par classe (si dispo)
        prob_by_class = {}
        if y_pred_prob is not None:
            if encoder is not None:
                class_labels = encoder.classes_
            else:
                class_labels = [str(i) for i in range(y_pred_prob.shape[1])]
            mean_probs = np.mean(y_pred_prob, axis=0)
            prob_by_class = {str(class_labels[i]): float(mean_probs[i]) for i in range(len(class_labels))}

        # Exemple de 10 premières prédictions
        preview = []
        for i in range(min(10, len(y_pred))):
            preview.append({
                "flow_id": str(df.index[i]) if "flow_id" not in df.columns else str(df.iloc[i]["flow_id"]),
                "prediction": str(y_pred[i])
            })

        print(f"[✅] Prédiction effectuée avec le modèle {model_type} ({len(y_pred)} échantillons).")

        # Retour JSON-friendly pour Flask
        return {
            "predicted_classes": predicted_classes,
            "prob_by_class": prob_by_class,
            "samples_preview": preview,
            "total_samples": int(len(df)),
            "model_used": model_type
        }

    except Exception as e:
        print(f"[❌] Erreur dans predict_with_model({model_type}): {e}")
        traceback.print_exc()
        return {"error": f"Erreur pendant la prédiction : {e}"}


### visualisation 

def plot_attack_distribution(predictions):
    """Crée un diagramme circulaire (Plotly avec fallback matplotlib)."""
    try:
        # Essayer avec Plotly d'abord
        pred_counts = pd.Series(predictions).value_counts()
        fig = px.pie(
            values=pred_counts.values,
            names=[str(x) for x in pred_counts.index],
            title="Répartition des Types d'Attaques",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        buf = BytesIO()
        try:
            fig.write_image(buf, format="png", engine="kaleido")
        except Exception:
            # Fallback: utiliser matplotlib si Plotly échoue
            return _plot_attack_distribution_matplotlib(predictions)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"[⚠️] Erreur plot_attack_distribution (Plotly) : {e}")
        # Fallback sur matplotlib
        return _plot_attack_distribution_matplotlib(predictions)

def _plot_attack_distribution_matplotlib(predictions):
    """Fallback matplotlib pour le camembert."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from collections import Counter
        
        counts = Counter(predictions)
        labels = list(counts.keys())
        sizes = list(counts.values())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title("Répartition des Types d'Attaques", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"[❌] Erreur matplotlib fallback : {e}")
        return None

def plot_feature_importance(features_df, predictions):
    """Crée un diagramme en barres (Plotly avec fallback matplotlib)."""
    try:
        df = features_df.copy()
        if len(predictions) != len(df):
            predictions = list(predictions)[:len(df)]
        df['Prediction'] = predictions
        
        # Vérifier si la colonne existe
        if 'flow_packets_per_sec' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col_name = numeric_cols[0]
            else:
                return _plot_feature_importance_matplotlib(features_df, predictions)
        else:
            col_name = 'flow_packets_per_sec'
        
        fig = go.Figure()
        for attack in df['Prediction'].unique():
            attack_data = df[df['Prediction'] == attack]
            if len(attack_data) > 0:
                fig.add_trace(go.Box(
                    y=attack_data[col_name],
                    name=str(attack),
                    boxmean='sd'
                ))
        fig.update_layout(
            title="Taux de Paquets par Type d'Attaque",
            yaxis_title='Packets/sec',
            xaxis_title="Type d'Attaque"
        )
        buf = BytesIO()
        try:
            fig.write_image(buf, format="png", engine="kaleido")
        except Exception:
            # Fallback: utiliser matplotlib si Plotly échoue
            return _plot_feature_importance_matplotlib(features_df, predictions)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"[⚠️] Erreur plot_feature_importance (Plotly) : {e}")
        return _plot_feature_importance_matplotlib(features_df, predictions)

def _plot_feature_importance_matplotlib(features_df, predictions):
    """Fallback matplotlib pour le graphique en barres."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = features_df.copy()
        if len(predictions) != len(df):
            predictions = list(predictions)[:len(df)]
        df['Prediction'] = predictions
        
        # Trouver une colonne numérique
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        
        feature = "flow_packets_per_sec" if "flow_packets_per_sec" in df.columns else numeric_cols[0]
        
        grouped = df.groupby("Prediction")[feature].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(grouped["Prediction"], grouped[feature], 
                      color=plt.cm.Set3(range(len(grouped))))
        ax.set_xlabel("Type d'attaque", fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.set_title(f"Moyenne de '{feature}' par classe prédite", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"[❌] Erreur matplotlib fallback : {e}")
        return None



