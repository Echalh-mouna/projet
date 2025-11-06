"""
train_model_hybride_fixed.py (VERSION DÉFINITIVE AVEC ALIGNEMENT ET SAUVEGARDE)
MODIFICATION: Adaptation aux 25 features de base enrichies.
"""

import os, gc, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import joblib 
import json 
import sys 
import importlib.util

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ===============================================
# 🌟 CHARGEMENT DES FEATURES 🌟
# ===============================================

def load_universal_features():
    try:
        spec = importlib.util.spec_from_file_location("data_preparation", "data_preparation.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["data_preparation"] = mod
        spec.loader.exec_module(mod)
        return mod.UNIVERSAL_FEATURES
    except Exception as e:
        print(f"❌ ERREUR: data_preparation.py ou UNIVERSAL_FEATURES est introuvable. {e}")
        # Fallback doit inclure les 3 nouvelles features pour la cohérence
        return [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets', 'total_fwd_bytes', 
            'total_bwd_bytes', 'packet_length_mean', 'packet_length_std', 'packet_length_min', 
            'packet_length_max', 'flow_bytes_per_sec', 'flow_packets_per_sec', 
            'fwd_bwd_packet_ratio', 'fwd_bwd_byte_ratio', 'iat_mean', 'iat_std', 
            'iat_max', 'iat_min', 'protocol', 'avg_fwd_packet_size', 
            'avg_bwd_packet_size', 'total_packets', 'total_bytes', 
            'total_bytes_per_packet', 'duration_per_packet', 'iat_std_to_mean_ratio'
        ]

UNIVERSAL_FEATURES = load_universal_features()
print(f"Features chargées: {len(UNIVERSAL_FEATURES)}")


# =============================
# TensorFlow (optionnel) + cfg
# =============================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, backend as K
    TF_AVAILABLE = True

    tf.random.set_seed(RANDOM_SEED)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    print("✅ TensorFlow disponible.")
except Exception as e:
    print(f"⚠️ TensorFlow indisponible: {e}")
    TF_AVAILABLE = False


# =============================
# Wrapper Keras -> sklearn OK
# =============================
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasSklearnClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, model=None, name="keras"):
        self.model = model
        self.name = name
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)


# =============================
#   Entraîneur principal
# =============================
class IDSModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        
        self.log_features = [
            'flow_duration', 'total_fwd_bytes', 'total_bwd_bytes', 'flow_bytes_per_sec', 
            'flow_packets_per_sec', 'iat_mean', 'iat_max', 'iat_min', 'total_packets', 'total_bytes'
        ]
        
        self.final_feature_names = None 

    # ---------- Préparation (ADAPTÉE AUX 25 FEATURES) ----------
    def prepare_training_data(self, df):
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if "label" not in df.columns:
            raise ValueError("La colonne 'label' est requise dans le dataset.")
        
        X = df.drop(columns=["label"])
        y = df["label"]
        
        # CORRECTION ROBUSTE ET ENCODAGE DU PROTOCOLE
        if "protocol" in X.columns:
            X['protocol'] = X['protocol'].astype(str).str.lower().str.strip()
            proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
            X["protocol"] = (
                X["protocol"]
                    .map(proto_map)
                    .fillna(X["protocol"])
            )
            
            X["protocol"] = pd.to_numeric(X["protocol"], errors="coerce").fillna(0)
            X['protocol'] = X['protocol'].astype(int)
            X = pd.get_dummies(X, columns=['protocol'], prefix='proto')
            
            # 🌟 GARANTIE DES COLONNES OHE 🌟
            expected_protocols = [1, 6, 17] 
            for p in expected_protocols:
                col_name = f'proto_{p}'
                if col_name not in X.columns:
                    X[col_name] = 0
            
            X = X.loc[:, ~X.columns.str.startswith('proto_0')]


        # Log-Transformation
        for col in self.log_features:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0)) 
                
        # Forcer numérique
        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
                
        # 🌟 EXPORT CRITIQUE DE LA LISTE DES FEATURES (le contrat final, ~28 colonnes)
        self.final_feature_names = list(X.columns)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Poids de classes
        classes = np.unique(y_encoded)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_encoded)
        self.class_weights = {i: w for i, w in enumerate(weights)}

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_SEED
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        print("✅ Données préparées.")
        print(f"Dimension après OHE/Log-Transform: {X_train.shape[1]} features.")
        print("Classes:", list(self.label_encoder.classes_))
        return X_train, X_test, y_train, y_test

    # ---------- ML ----------
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=RANDOM_SEED
        )
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"🌲 RF Accuracy: {acc:.4f}")
        return rf, acc

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        model = xgb.XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", 
            random_state=RANDOM_SEED, n_jobs=-1
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"🚀 XGB Accuracy: {acc:.4f}")
        return model, acc

    # ---------- DNN (AMÉLIORÉ et optimisé pour > 0.8) ----------
    def build_dnn(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(2048, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.45),
            layers.Dense(1024, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.40),
            layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Dense(num_classes, activation="softmax")
        ])
        opt = keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-5) 
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_dnn(self, X_train, y_train, X_test, y_test, epochs=100):
        if not TF_AVAILABLE:
            print("❌ DNN ignoré (TensorFlow indisponible).")
            return None, 0.0

        model = self.build_dnn(X_train.shape[1], len(np.unique(y_train)))

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True), 
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7), 
            keras.callbacks.TerminateOnNaN()
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=256, 
            verbose=1,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"✅ DNN Accuracy: {acc:.4f}")
        return model, acc

    # ---------- CNN 1D (AMÉLIORÉ et optimisé pour > 0.8) ----------
    def build_cnn(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(512, 5, activation="relu", padding="same"), 
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(256, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(num_classes, activation="softmax")
        ])
        opt = keras.optimizers.Adam(learning_rate=3e-4) 
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_cnn(self, X_train, y_train, X_test, y_test, epochs=100):
        if not TF_AVAILABLE:
            print("❌ CNN ignoré (TensorFlow indisponible).")
            return None, 0.0

        model = self.build_cnn(X_train.shape[1], len(np.unique(y_train)))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6)
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=512, 
            verbose=1,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        print(f"✅ CNN Accuracy: {acc:.4f}")

        return model, acc

    # ---------- Ensemble 2DL + 2ML ----------
    def build_hybrid_2dl_2ml(self, trained, X_train, y_train, X_test, y_test):
        estimators, weights = [], []

        # DL
        dnn, dnn_acc = trained.get("DNN", (None, 0.0))
        cnn, cnn_acc = trained.get("CNN", (None, 0.0))
        if dnn is not None:
            estimators.append(("dnn", KerasSklearnClassifier(dnn, "dnn"))); weights.append(max(dnn_acc, 1e-6))
        if cnn is not None:
            estimators.append(("cnn", KerasSklearnClassifier(cnn, "cnn"))); weights.append(max(cnn_acc, 1e-6))

        # ML
        xgb_model, xgb_acc = trained.get("XGBoost", (None, 0.0))
        rf, rf_acc       = trained.get("RandomForest", (None, 0.0))
        if xgb_model is not None:
            estimators.append(("xgb", xgb_model)); weights.append(max(xgb_acc, 1e-6))
        if rf is not None and hasattr(rf, "predict_proba"):
            estimators.append(("rf", rf)); weights.append(max(rf_acc, 1e-6))

        if len(estimators) < 2:
            print("⚠️ Pas assez de modèles pour l'ensemble hybride.")
            return None, 0.0

        w = np.array(weights, dtype=float); w = w / w.sum()

        # 1) Essai VotingClassifier (n_jobs=1 pour éviter les blocages Windows)
        try:
            ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=w.tolist(), n_jobs=1)
            ensemble.fit(X_train, y_train)
            acc = accuracy_score(y_test, ensemble.predict(X_test))
            print(f"🎛️ Ensemble (VotingClassifier) Accuracy: {acc:.4f}")
            return ensemble, acc
        except Exception as e:
            print(f"⚠️ VotingClassifier a échoué ({e}) → fallback soft maison.")

        # 2) Fallback soft voting maison
        class SoftAverager:
            _estimator_type = "classifier"
            def __init__(self, ests, weights):
                self.ests = ests
                self.weights = np.array(weights) / np.sum(weights)
                self.classes_ = None
            def fit(self, X, y):
                self.classes_ = np.unique(y); return self
            def predict_proba(self, X):
                agg = None
                for (name, est), w in zip(self.ests, self.weights):
                    if hasattr(est, "predict_proba"):
                        p = est.predict_proba(X)
                    else:
                        pred = est.predict(X)
                        k = int(np.max(pred)) + 1
                        p = np.zeros((len(pred), k), dtype=float)
                        for i, c in enumerate(pred): 
                            if 0 <= c < k: p[i, c] = 1.0
                    if agg is None:
                        agg = w * p
                    else:
                        if agg.shape[1] < p.shape[1]:
                            agg = np.pad(agg, ((0,0),(0,p.shape[1]-agg.shape[1])))
                        if p.shape[1] < agg.shape[1]:
                            p = np.pad(p, ((0,0),(0,agg.shape[1]-p.shape[1])))
                        agg += w * p
                return agg
            def predict(self, X):
                p = self.predict_proba(X)
                return np.argmax(p, axis=1)

        soft = SoftAverager(estimators, w).fit(X_train, y_train)
        y_pred = soft.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎛️ Ensemble (fallback soft) Accuracy: {acc:.4f}")
        return soft, acc


# =============================
#             MAIN
# =============================
def main():
    DATA_FILE = "data/processed/combined_dataset.csv"
    if not os.path.exists(DATA_FILE):
        print(f"❌ Fichier introuvable: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    trainer = IDSModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(df) 

    results = {}

    # ------- ML -------
    rf, rf_acc = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    results["RandomForest"] = (rf, rf_acc)
    results["XGBoost"] = (xgb_model, xgb_acc)

    # ------- DL -------
    if TF_AVAILABLE:
        dnn, dnn_acc = trainer.train_dnn(X_train, y_train, X_test, y_test, epochs=100)
        
        # Libérer la mémoire
        gc.collect(); K.clear_session()
        
        cnn, cnn_acc = trainer.train_cnn(X_train, y_train, X_test, y_test, epochs=100)
        results["DNN"] = (dnn, dnn_acc)
        results["CNN"] = (cnn, cnn_acc)
    else:
        print("ℹ️ DL ignoré (TensorFlow non installé).")
        results["DNN"] = (None, 0.0)
        results["CNN"] = (None, 0.0)

    # ------- Ensemble 2DL+2ML -------
    ensemble, ens_acc = trainer.build_hybrid_2dl_2ml(results, X_train, y_train, X_test, y_test)
    results["Ensemble"] = (ensemble, ens_acc)

    # ------- Récap -------
    print("\n" + "="*72)
    print(f"{'Modèle':<20} {'Accuracy':>9}")
    print("-"*72)
    for name, (m, acc) in results.items():
        print(f"{name:<20} {acc:>9.4f}")
    print("="*72)

    # ==================================
    # 🌟 BLOC DE SAUVEGARDE DÉFINITIF 🌟
    # ==================================
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 🌟 1. Sauvegarde CRITIQUE de la liste des features 🌟
    try:
        if trainer.final_feature_names is not None:
             joblib.dump(trainer.final_feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
             print(f"✅ Liste des features ({len(trainer.final_feature_names)} colonnes) sauvegardée dans feature_names.pkl.")
    except Exception as e:
         print(f"⚠️ Erreur lors de la sauvegarde du fichier feature_names.pkl: {e}")
    
    # 2. Sauvegarde des objets de pré-traitement
    try:
        joblib.dump(trainer.scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        joblib.dump(trainer.label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
        print("✅ Scaler et LabelEncoder sauvegardés.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde du pré-traitement: {e}")

    # 3. Sauvegarde des Modèles (ML et Ensemble)
    try:
        joblib.dump(results["RandomForest"][0], os.path.join(MODELS_DIR, "ids_rf_model.pkl"))
        joblib.dump(results["XGBoost"][0], os.path.join(MODELS_DIR, "ids_xgb_model.pkl"))
        if results["Ensemble"][0] is not None:
            joblib.dump(results["Ensemble"][0], os.path.join(MODELS_DIR, "ensemble_fitted.pkl"))
        
        print("✅ Modèles ML et Ensemble sauvegardés.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde des modèles ML/Ensemble: {e}")

    # 4. Sauvegarde des Modèles DL (Keras)
    if TF_AVAILABLE:
        try:
            if results["DNN"][0] is not None:
                results["DNN"][0].save(os.path.join(MODELS_DIR, "ids_dnn_model.keras")) 
            if results["CNN"][0] is not None:
                results["CNN"][0].save(os.path.join(MODELS_DIR, "ids_cnn_model.keras"))
            print("✅ Modèles DL sauvegardés au format .keras.")
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde des modèles Keras: {e}")

    # 5. Sauvegarde des métadonnées (Accuracy)
    try:
        models_info = {name: acc for name, (_, acc) in results.items()}
        with open(os.path.join(MODELS_DIR, "session_metadata.json"), 'w') as f:
            json.dump({'models_info': models_info}, f)
        print("✅ Métadonnées (Accuracy) sauvegardées.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde des métadonnées: {e}")
    
    # Nettoyage TF final
    if TF_AVAILABLE:
        K.clear_session()
    gc.collect()
    print("✅ Terminé (sans blocage).")

if __name__ == "__main__":
    main()