from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import os
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D # Commenté car le graphique 3D n'est plus utilisé

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

model = None
expected_features_from_model = []
try:
    model = load("xgb_model.joblib")
    expected_features_from_model = [
        'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Bwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
        'Subflow Bwd Bytes', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou de la définition des features attendues : {e}")

class_map = {
    0: "BENIGN",
    1: "DrDoS_DNS",
    2: "DrDoS_LDAP",
    3: "DrDoS_MSSQL",
    4: "DrDoS_NTP",
    5: "DrDoS_NetBIOS",
    6: "DrDoS_SNMP",
    7: "DrDoS_SSDP"
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "Aucun fichier sélectionné"
        
        try:
            df_original = pd.read_csv(file)
        except Exception as e:
            return f"Erreur lors de la lecture du fichier CSV : {e}. Assurez-vous que le fichier est un CSV valide."

        df_for_prediction = df_original.copy()
        
        cols_to_drop_for_model = ['Source Port', 'Destination Port', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
        df_for_prediction = df_for_prediction.drop(columns=cols_to_drop_for_model, errors='ignore')

        df_final_for_model = pd.DataFrame(columns=expected_features_from_model)

        for col in expected_features_from_model:
            if col in df_for_prediction.columns:
                df_final_for_model[col] = df_for_prediction[col]
            else:
                df_final_for_model[col] = 0 
        
        df_final_for_model = df_final_for_model.apply(pd.to_numeric, errors='coerce')
        df_final_for_model = df_final_for_model.replace([np.inf, -np.inf], np.nan).fillna(0)

        if model is None:
            return "Erreur : Le modèle de prédiction n'a pas pu être chargé au démarrage de l'application. Vérifiez le fichier xgb_model.joblib."
        
        try:
            preds = model.predict(df_final_for_model)
        except Exception as e:
            return f"Erreur lors de la prédiction : {e}. Assurez-vous que les données d'entrée correspondent aux attentes du modèle."
        
        df_original["Prediction_Num"] = preds
        df_original["Prediction_Label"] = [class_map[p] for p in preds]

        result_path = os.path.join('static', 'predicted_results.csv')
        df_original.to_csv(result_path, index=False)

        plt.style.use('dark_background')
        plt.rcParams.update({
            'axes.facecolor': '#551c7f',
            'figure.facecolor': '#551c7f',
            'text.color': '#e0e0e0',
            'axes.labelcolor': '#a0e6ff',
            'xtick.color': '#e0e0e0',
            'ytick.color': '#e0e0e0',
            'grid.color': '#7a3a9b',
            'legend.facecolor': '#3f175a',
            'legend.edgecolor': '#a0e6ff',
            'savefig.facecolor': '#551c7f',
            'savefig.transparent': True
        })

        # Fonction utilitaire pour générer une image de placeholder si le graphique ne peut pas être créé
        # Conservée au cas où le graphique circulaire ne pourrait pas être généré (peu probable)
        def generate_placeholder_image(filename, title_text="Données non disponibles"):
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, title_text,
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=16, color='gray', transform=plt.gca().transAxes)
            plt.axis('off')
            plt.savefig(os.path.join('static', filename), bbox_inches='tight', transparent=True)
            plt.close()

        # Graphique 1: Proportion des types de trafic (Pie Chart) - CONSERVÉ
        plt.figure(figsize=(8,8))
        df_original["Prediction_Label"].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=90, cmap='Pastel1', pctdistance=0.85,
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}
        )
        plt.title("Proportion des types de trafic prédits", fontsize=14, color='#a0e6ff')
        plt.ylabel('')
        centre_circle = plt.Circle((0,0),0.70,fc='#551c7f', edgecolor='black', linewidth=0.5)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join('static', 'prediction_piechart.png'), bbox_inches='tight')
        plt.close()

        # --- Les autres graphiques ont été supprimés comme demandé ---
        # (Distribution de la Durée des Flux, Distribution par Protocole, Distribution de Total Fwd Packets)
        
        # Table 1: Résumé des prédictions (affiche toutes les classes, même avec 0 occurrences) - CONSERVÉ
        all_classes_df = pd.DataFrame({'Label': list(class_map.values())})
        
        prediction_counts = df_original['Prediction_Label'].value_counts().reset_index()
        prediction_counts.columns = ['Label', 'Nombre de Label']
        
        prediction_summary = pd.merge(all_classes_df, prediction_counts, on='Label', how='left').fillna(0)
        prediction_summary['Nombre de Label'] = prediction_summary['Nombre de Label'].astype(int)
        
        total_row = pd.DataFrame([['Total', prediction_summary['Nombre de Label'].sum()]], columns=['Label', 'Nombre de Label'])
        summary_table_html = pd.concat([prediction_summary, total_row], ignore_index=True).to_html(index=False, classes='table table-striped')

        # Table 2: Aperçu des 5 premières lignes du fichier traité (colonnes importantes seulement) - CONSERVÉ
        important_preview_columns = [
            'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Prediction_Label'
        ]
        existing_important_cols = [col for col in important_preview_columns if col in df_original.columns]
        
        preview_table_html = df_original[existing_important_cols].head().to_html(index=False, classes='table table-striped')


        # --- Rendu du template ---
        return render_template('result.html',
                               summary_table=summary_table_html,
                               preview_table=preview_table_html)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
