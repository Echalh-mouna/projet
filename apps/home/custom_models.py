import numpy as np

# ===============================================================
# 🧠 Classe personnalisée PrefitVotingEnsemble
# ===============================================================
class PrefitVotingEnsemble:
    """
    Implémentation d’un modèle d’ensemble pré-entraîné,
    combinant plusieurs estimateurs (par ex. CNN, DNN, XGBoost, etc.).
    Utilisé à la fois dans l’app Streamlit et Flask pour prédire les classes réseau.
    """

    def __init__(self, estimators, weights=None, voting='soft', label_encoder=None):
        self.estimators = estimators
        self.voting = voting
        self.label_encoder = label_encoder
        self.classes_ = (
            list(label_encoder.classes_) if label_encoder is not None else None
        )

        # Normalisation des poids
        self.weights = np.array(weights) if weights is not None else np.ones(len(estimators))
        if self.weights.sum() == 0:
            self.weights = np.ones(len(estimators))
        self.weights = self.weights / self.weights.sum()

    # ---------------------------------------------------------------
    # 🔹 Prédiction directe
    # ---------------------------------------------------------------
    def predict(self, X):
        """
        Combine les prédictions de chaque modèle base.
        Retourne la classe finale majoritaire ou pondérée.
        """
        all_preds = []

        for i, est in enumerate(self.estimators):
            model = self._unwrap_estimator(est)
            try:
                preds = model.predict(X)
                all_preds.append(preds)
            except Exception as e:
                print(f"[⚠️] Erreur lors de la prédiction avec le modèle {i}: {e}")
                continue

        if not all_preds:
            raise ValueError("Aucun modèle n’a retourné de prédiction valide.")

        # Stack et vote majoritaire / pondéré
        all_preds = np.array(all_preds)

        if self.voting == 'soft':
            final_preds = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(), axis=0, arr=all_preds
            )
        else:
            # Hard voting (pondéré)
            weights = self.weights[:len(all_preds)]
            weighted_votes = np.tensordot(weights, all_preds, axes=(0, 0))
            final_preds = np.round(weighted_votes).astype(int)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(final_preds)
        return final_preds

    # ---------------------------------------------------------------
    # 🔹 Prédiction probabiliste
    # ---------------------------------------------------------------
    def predict_proba(self, X):
        """
        Retourne les probabilités moyennes pondérées de chaque modèle.
        Nécessite que tous les estimateurs supportent predict_proba().
        """
        probas = []

        for i, est in enumerate(self.estimators):
            model = self._unwrap_estimator(est)
            if hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(X)
                    probas.append(p * self.weights[i])
                except Exception as e:
                    print(f"[⚠️] Erreur dans predict_proba modèle {i}: {e}")
            else:
                print(f"[ℹ️] Le modèle {i} ne supporte pas predict_proba.")

        if not probas:
            raise ValueError("Aucun modèle valide pour predict_proba.")

        avg_proba = np.sum(probas, axis=0)
        return avg_proba / np.sum(self.weights)

    # ---------------------------------------------------------------
    # 🔹 Méthodes utilitaires
    # ---------------------------------------------------------------
    def get_params(self, deep=True):
        return {
            "estimators": self.estimators,
            "weights": self.weights,
            "voting": self.voting,
            "label_encoder": self.label_encoder,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ---------------------------------------------------------------
    # 🔹 Helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _unwrap_estimator(est):
        """Supporte les tuples (nom, modèle) issus de VotingClassifier."""
        if isinstance(est, tuple) and len(est) >= 2:
            return est[1]
        return est
