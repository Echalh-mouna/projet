import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_migrate import Migrate
from flask_minify import Minify
from sys import exit

# from api_generator.commands import gen_api  # Désactivé pour éviter les erreurs
from apps.config import config_dict
from apps import create_app, db

# Configuration de l'application
DEBUG = (os.getenv('DEBUG', 'False') == 'True')
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    app_config = config_dict[get_config_mode.capitalize()]
except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production]')

# Création de l'application Flask
app = create_app(app_config)

# Charger le modèle sauvegardé
# Charger le modèle sauvegardé
try:
    with open('nnc.pkl', 'rb') as f:
        loaded_rf_model = pickle.load(f)
except FileNotFoundError:
    print("Le fichier nnc.pkl n'a pas été trouvé.")
  


# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = request.form.to_dict()
        data_to_predict = [float(data[feature]) for feature in data.keys()]

        # Liste des noms de caractéristiques utilisés lors de l'entraînement du modèle
        feature_names = [
            'Unnamed: 0', 'Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets',
            'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
            'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
            'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
            'Packet Length Variance', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
            'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]

        # Convertir les données en DataFrame pandas
        data_to_predict_df = pd.DataFrame([data_to_predict], columns=feature_names)

        # Faire une prédiction
        prediction = loaded_rf_model.predict(data_to_predict_df)

        # Renvoyer la prédiction
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        # Renvoyer une erreur en cas d'exception
        return jsonify({'error': str(e)}), 500

# Initialiser Flask-Migrate
Migrate(app, db)

# Minifier l'application si elle n'est pas en mode debug
if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

# Log des informations de configuration
if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG))
    app.logger.info('Page Compression = ' + ('FALSE' if DEBUG else 'TRUE'))
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT)

# Ajouter les commandes CLI (désactivé)
# for command in [gen_api, ]:
#     app.cli.add_command(command)

if __name__ == "__main__":
    app.run(debug=DEBUG)
