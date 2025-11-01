
"""
train_model_hybride.py
Version: 1.0
- Dataset: data/processed/combined_dataset.csv
- Models: RF, XGB, SVM + (DNN, CNN, LSTM si TensorFlow est dispo)
- Ensemble hybride (ML + DL) avec soft voting pondéré + pré-ensemble pré-entraîné
- Sauvegarde: models/
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Optional legacy imports kept for compatibility (non utilisés directement)
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
#      TensorFlow (optionnel) + Wrappers picklables
# ======================================================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("✅ TensorFlow chargé.")
except Exception as e:
    print(f"⚠️ TensorFlow non disponible: {e}")
    TF_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin

class TFEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper Keras -> interface sklearn, pour Voting/Ensembles."""
    _estimator_type = "classifier"

    def __init__(self, tf_model):
        self.tf_model = tf_model

    def fit(self, X, y):
        return self  # modèle déjà entraîné

    def predict(self, X):
        p = self.tf_model.predict(X, verbose=0)
        return np.argmax(p, axis=1)

    def predict_proba(self, X):
        return self.tf_model.predict(X, verbose=0)

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """Assure la compatibilité sklearn pour VotingClassifier (classes_)."""
    _estimator_type = "classifier"
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.xgb_model.predict(X)

    def predict_proba(self, X):
        return self.xgb_model.predict_proba(X)

# ======================================================
#                    Entraîneur IDS
# ======================================================
class IDSModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_weights = None  # pour DL

    # --------------- Préparation des données ---------------
    def prepare_training_data(self, df):
        # Nettoyage de base
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Conversion protocole -> numérique si présent
        if 'protocol' in df.columns:
            proto_map = {
                'tcp': 6, 'udp': 17, 'icmp': 1, 'crtp': 16, 'ospf': 89,
                'mobile': 55, 'sun-nd': 77, 'mux': 18
            }
            df['protocol'] = df['protocol'].astype(str).str.lower().map(proto_map).fillna(pd.to_numeric(df['protocol'], errors='coerce')).fillna(0).astype(int)

        # Clip valeurs négatives
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            if c != 'label' and c in df.columns:
                df[c] = df[c].clip(lower=0)

        # Séparation X / y
        X = df.drop(columns=['label'])
        y = df['label']

        # Force numérique
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

        self.feature_names = X.columns.tolist()

        # Encodage labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Weights pour classes (utile si dataset déséquilibré)
        classes = np.unique(y_encoded)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
        self.class_weights = {i: w for i, w in enumerate(weights)}

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print("✅ Données préparées.")
        print("Classes:", list(self.label_encoder.classes_))
        print("Weights:", self.class_weights)

        return X_train, X_test, y_train, y_test

    # --------------- Modèles ML ---------------
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🌲 RF Acc: {acc:.4f}")
        return rf, acc

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🚀 XGB Acc: {acc:.4f}")
        return xgb_model, acc

    def train_svm(self, X_train, y_train, X_test, y_test):
        # SVM avec probas (RBF), sous-échantillon si X trop grand
        if len(X_train) > 20000:
            idx = np.random.choice(len(X_train), 20000, replace=False)
            X_tr, y_tr = X_train[idx], y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train

        svm = SVC(kernel='rbf', C=4, gamma='scale', probability=True, random_state=42)
        svm.fit(X_tr, y_tr)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 SVM Acc: {acc:.4f}")
        return svm, acc

    # --------------- Modèles DL (Keras) ---------------
    def _build_dnn(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _build_cnn(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _build_lstm(self, input_dim, num_classes):
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def _train_keras(self, model, X_train, y_train, X_test, y_test, epochs=70):
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=256,
            callbacks=callbacks,
            verbose=1,
            class_weight=self.class_weights
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        return model, acc

    def train_dnn(self, X_train, y_train, X_test, y_test, epochs=70):
        if not TF_AVAILABLE:
            print("❌ DNN ignoré (TF indisponible).")
       	    return None, 0.0
        model = self._build_dnn(X_train.shape[1], len(np.unique(y_train)))
        print("🧠 Entraînement DNN...")
        return self._train_keras(model, X_train, y_train, X_test, y_test, epochs)

    def train_cnn(self, X_train, y_train, X_test, y_test, epochs=70):
        if not TF_AVAILABLE:
            print("❌ CNN ignoré (TF indisponible).")
            return None, 0.0
        model = self._build_cnn(X_train.shape[1], len(np.unique(y_train)))
        print("🏗️ Entraînement CNN...")
        return self._train_keras(model, X_train, y_train, X_test, y_test, epochs)

    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=70):
        if not TF_AVAILABLE:
            print("❌ LSTM ignoré (TF indisponible).")
            return None, 0.0
        model = self._build_lstm(X_train.shape[1], len(np.unique(y_train)))
        print("🔄 Entraînement LSTM...")
        return self._train_keras(model, X_train, y_train, X_test, y_test, epochs)

    # --------------- Utilitaires ---------------
    def plot_confusion(self, y_true, y_pred, model_name, out_dir='models'):
        os.makedirs(out_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Prédit')
        plt.ylabel('Réel')
        plt.tight_layout()
        path = os.path.join(out_dir, f'confusion_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(path, dpi=250)
        plt.close()
        return path

    def save_all(self, models_dict, out_dir='models'):
        import joblib
        os.makedirs(out_dir, exist_ok=True)

        # Sauvegarde modèles
        name_map = {
            'RandomForest':'rf', 'XGBoost':'xgb', 'SVM':'svm',
            'DNN':'dnn', 'CNN':'cnn', 'LSTM':'lstm',
            'Ensemble':'ensemble', 'HybridPrefit':'ensemble_fitted'
        }

        for name, (model, acc) in models_dict.items():
            if model is None:
                continue
            tag = name_map.get(name, name.lower())
            # Keras (DL) -> .keras
            if name in ['DNN','CNN','LSTM'] and TF_AVAILABLE:
                path = os.path.join(out_dir, f'ids_{tag}_model.keras')
                try:
                    model.save(path)
                except Exception:
                    # fallback SavedModel dir
                    path = os.path.join(out_dir, f'ids_{tag}_model')
                    model.save(path)
                print(f"💾 {name} sauvegardé -> {path} (acc={acc:.4f})")
            else:
                path = os.path.join(out_dir, f'ids_{tag}_model.pkl')
                joblib.dump(model, path)
                print(f"💾 {name} sauvegardé -> {path} (acc={acc:.4f})")

        # Préprocesseurs
        joblib.dump(self.scaler, os.path.join(out_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(out_dir, 'label_encoder.pkl'))
        joblib.dump(self.feature_names, os.path.join(out_dir, 'feature_names.pkl'))
        print("✅ Préprocesseurs sauvegardés.")

    def build_hybrid_voting(self, trained, X_train, y_train, X_test, y_test):
        """Construit un ensemble hybride (ML + DL) avec soft voting pondéré.
           Inclut RF, XGB, SVM + DNN/CNN/LSTM s'ils sont entraînés.
        """
        estimators = []
        weights = []

        # ML
        for key in ['RandomForest', 'XGBoost', 'SVM']:
            model, acc = trained.get(key, (None, 0.0))
            if model is not None:
                if key == 'XGBoost':
                    wrap = XGBoostWrapper(model); wrap.fit(X_train, y_train)
                    estimators.append((key.lower(), wrap)); weights.append(max(acc, 1e-6))
                else:
                    estimators.append((key.lower(), model)); weights.append(max(acc, 1e-6))

        # DL (si dispo)
        if TF_AVAILABLE:
            for key in ['DNN','CNN','LSTM']:
                model, acc = trained.get(key, (None, 0.0))
                if model is not None:
                    estimators.append((key.lower(), TFEnsembleWrapper(model)))
                    weights.append(max(acc, 1e-6))

        if len(estimators) < 2:
            print("⚠️ Pas assez de modèles pour un ensemble hybride.")
            return None, 0.0

        # Normalisation des poids
        w = np.array(weights, dtype=float)
        w = w / w.sum()

        try:
            ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=w.tolist(), n_jobs=-1)
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"🎛️ Ensemble hybride (soft) Acc: {acc:.4f}")
            return ensemble, acc
        except Exception as e:
            print(f"⚠️ Soft voting hybride échoué: {e}")
            # fallback hard
            try:
                ensemble = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"🎛️ Ensemble hybride (hard) Acc: {acc:.4f}")
                return ensemble, acc
            except Exception as e2:
                print(f"❌ Hard voting échoué aussi: {e2}")
                return None, 0.0

    def create_prefit_hybrid_from_disk(self, models_dir='models', out_path='models/ensemble_fitted.pkl'):
        """Construit un ensemble hybride pré-entraîné (depuis les fichiers .pkl/.keras)."""
        import joblib, pickle
        ests, weights = [], []
        # Charge sklearn
        for f in os.listdir(models_dir):
            p = os.path.join(models_dir, f)
            if os.path.isfile(p) and f.endswith('.pkl') and not f.startswith(('scaler','label_encoder','feature_names','ensemble')):
                try:
                    obj = joblib.load(p)
                    if hasattr(obj, 'predict'):
                        ests.append((os.path.splitext(f)[0], obj)); weights.append(1.0)
                except Exception:
                    pass
        # Charge Keras
        if TF_AVAILABLE:
            for f in os.listdir(models_dir):
                p = os.path.join(models_dir, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in ('.keras', '.h5', '.hdf5'):
                    try:
                        m = keras.models.load_model(p)
                        ests.append((os.path.splitext(f)[0], TFEnsembleWrapper(m))); weights.append(1.0)
                    except Exception:
                        pass

        if len(ests) < 2:
            print("⚠️ Pas assez de modèles chargés pour un prefit ensemble.")
            return None

        weights = np.array(weights); weights = weights / weights.sum()

        class PrefitHybrid:
            def __init__(self, estimators, weights):
                self.estimators = estimators
                self.weights = weights

            def predict_proba(self, X):
                ps = None
                for (name, est), w in zip(self.estimators, self.weights):
                    try:
                        if hasattr(est, 'predict_proba'):
                            p = est.predict_proba(X)
                        else:
                            preds = est.predict(X)
                            # one-hot approx
                            n = int(np.max(preds)) + 1
                            p = np.zeros((len(preds), n))
                            for i, k in enumerate(preds):
                                if k < n:
                                    p[i, k] = 1.0
                        if ps is None:
                            ps = w * p
                        else:
                            # align columns if needed
                            if ps.shape[1] < p.shape[1]:
                                ps = np.pad(ps, ((0,0),(0,p.shape[1]-ps.shape[1])))
                            if p.shape[1] < ps.shape[1]:
                                p = np.pad(p, ((0,0),(0,ps.shape[1]-p.shape[1])))
                            ps += w * p
                    except Exception:
                        continue
                return ps

            def predict(self, X):
                p = self.predict_proba(X)
                if p is None:
                    # fallback votes durs
                    votes = []
                    for _, est in self.estimators:
                        try:
                            votes.append(est.predict(X))
                        except Exception:
                            pass
                    if not votes:
                        return np.zeros((X.shape[0],), dtype=int)
                    votes = np.vstack(votes)
                    out = []
                    for i in range(votes.shape[1]):
                        try:
                            out.append(np.bincount(votes[:, i]).argmax())
                        except Exception:
                            out.append(0)
                    return np.array(out)
                return np.argmax(p, axis=1)

        obj = PrefitHybrid(ests, weights)
        try:
            import joblib
            joblib.dump(obj, out_path)
        except Exception:
            with open(out_path, 'wb') as f:
                pickle.dump(obj, f)
        print(f"✅ Pré-ensemble hybride sauvegardé -> {out_path}")
        return out_path

# ======================================================
#                        MAIN
# ======================================================
def main():
    DATA_FILE = 'data/processed/combined_dataset.csv'
    if not os.path.exists(DATA_FILE):
        print(f"❌ Fichier introuvable: {DATA_FILE}")
        return

    print("📂 Chargement dataset...")
    df = pd.read_csv(DATA_FILE)
    if 'label' not in df.columns:
        print("❌ Colonne 'label' absente !")
        return

    trainer = IDSModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_training_data(df)

    models = {}

    # --- ML ---
    rf, rf_acc = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    models['RandomForest'] = (rf, rf_acc)

    xgb_model, xgb_acc = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    models['XGBoost'] = (xgb_model, xgb_acc)

    svm, svm_acc = trainer.train_svm(X_train, y_train, X_test, y_test)
    models['SVM'] = (svm, svm_acc)

    # --- DL (si TensorFlow) ---
    if TF_AVAILABLE:
        dnn, dnn_acc = trainer.train_dnn(X_train, y_train, X_test, y_test, epochs=80)
        models['DNN'] = (dnn, dnn_acc)

        cnn, cnn_acc = trainer.train_cnn(X_train, y_train, X_test, y_test, epochs=80)
        models['CNN'] = (cnn, cnn_acc)

        lstm, lstm_acc = trainer.train_lstm(X_train, y_train, X_test, y_test, epochs=80)
        models['LSTM'] = (lstm, lstm_acc)
    else:
        print("ℹ️ DL ignoré (TensorFlow non installé).")

    # --- Ensemble Hybride (ML + DL) ---
    ensemble, ens_acc = trainer.build_hybrid_voting(models, X_train, y_train, X_test, y_test)
    if ensemble is not None:
        models['Ensemble'] = (ensemble, ens_acc)

    # --- Sauvegarde ---
    trainer.save_all(models, out_dir='models')

    # --- Pré-ensemble à partir des fichiers ---
    trainer.create_prefit_hybrid_from_disk(models_dir='models', out_path='models/ensemble_fitted.pkl')

    # --- Récap ---
    print("\n" + "="*72)
    print(f"{'Modèle':<18} {'Accuracy':>9}")
    print("-"*72)
    best_name, best_acc = None, -1
    for k, (m, acc) in models.items():
        print(f"{k:<18} {acc:>9.4f}")
        if acc > best_acc:
            best_name, best_acc = k, acc
    print("-"*72)
    if best_name:
        print(f"🏆 Meilleur: {best_name} ({best_acc:.4f})")
    print("="*72)
    print("✅ Entraînement terminé. Modèles dans: models/")

if __name__ == "__main__":
    main()
