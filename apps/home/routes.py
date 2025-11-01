# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
import pandas as pd
import json
from werkzeug.utils import secure_filename
from flask import render_template, request, jsonify, flash, redirect, url_for, session
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound

from apps.config import API_GENERATOR
from apps.home import blueprint

@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index', page_title='Tableau de Bord', API_GENERATOR=len(API_GENERATOR))

@blueprint.route('/upload')
@login_required
def upload():
    return render_template('home/upload.html', segment='upload', page_title='Analyse de Fichiers CSV', API_GENERATOR=len(API_GENERATOR))

@blueprint.route('/analysis')
@login_required
def analysis():
    return render_template('home/analysis.html', segment='analysis', page_title='Analyse de Fichier', API_GENERATOR=len(API_GENERATOR))

# Configuration pour l'upload
# Utiliser un chemin absolu basé sur le répertoire du projet pour éviter les problèmes de CWD
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@blueprint.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Route pour uploader un fichier CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
        # Lire le CSV et stocker temporairement
        try:
            df = pd.read_csv(filepath)
            # Limiter la taille des données pour éviter les problèmes de cookie
            if len(df) > 1000:
                df = df.head(1000)  # Limiter à 1000 lignes max
            
            # Sauvegarder le DataFrame dans un fichier temporaire
            temp_file = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
            df.to_pickle(temp_file)
            
            # Stocker seulement le nom du fichier dans la session
            session['temp_file'] = temp_file
            session['filename'] = filename
            
            return jsonify({
                'success': True,
                'message': 'Fichier uploadé avec succès',
                'filename': filename,
                'shape': df.shape,
                'columns': df.columns.tolist()[:10]  # Premières 10 colonnes
            })
        except Exception as e:
            return jsonify({'error': f'Erreur lors de la lecture du CSV: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Format de fichier non supporté. Seuls les fichiers CSV sont acceptés'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Erreur lors de l\'upload: {str(e)}'}), 500

@blueprint.route('/analyze-data', methods=['POST'])
@login_required
def analyze_data():
    """Route pour analyser les données CSV avec les modèles d'ensemble"""
    try:
        if 'temp_file' not in session:
            return jsonify({'error': 'Aucune donnée à analyser'}), 400
        
        # Récupérer les données du fichier temporaire
        df = pd.read_pickle(session['temp_file'])
        
        # Simulation des résultats d'analyse (mode démonstration)
        results = {
            'demo_mode': True,
            'total_samples': len(df),
            'anomaly_count': int(len(df) * 0.15),  # 15% d'anomalies simulées
            'anomaly_percentage': 15.0,
            'class_distribution': {
                'BENIGN': int(len(df) * 0.7),
                'DDoS': int(len(df) * 0.1),
                'DoS Hulk': int(len(df) * 0.05),
                'PortScan': int(len(df) * 0.05),
                'Bot': int(len(df) * 0.03),
                'SSH-Patator': int(len(df) * 0.02),
                'FTP-Patator': int(len(df) * 0.02),
                'Web Attack - Brute Force': int(len(df) * 0.01),
                'Web Attack - Sql Injection': int(len(df) * 0.01),
                'Web Attack - XSS': int(len(df) * 0.01)
            },
            'preprocessed_data': {
                'original': df.head(10).to_dict('records'),
                'normalized': (df.head(10) * 0.1).to_dict('records')  # Simulation de normalisation
            },
            'predictions': {
                'cnn': {'class': 'BENIGN', 'confidence': 85.2},
                'dnn': {'class': 'BENIGN', 'confidence': 87.1},
                'lightgbm': {'class': 'BENIGN', 'confidence': 89.3},
                'isolation_forest': {'anomaly': False, 'score': 0.15},
                'final': {
                    'class': 'BENIGN',
                    'confidence': 87.2,
                    'is_anomaly': False
                }
            },
            'model_accuracy': {
                'cnn': 95.8,
                'dnn': 96.2,
                'lightgbm': 97.1,
                'ensemble': 98.7
            }
        }
        
        # Stocker les résultats dans la session
        session['analysis_results'] = results
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors de l\'analyse: {str(e)}'}), 500

@blueprint.route('/get-analysis-results')
@login_required
def get_analysis_results():
    """Route pour récupérer les résultats d'analyse"""
    if 'analysis_results' in session:
        return jsonify(session['analysis_results'])
    else:
        return jsonify({'error': 'Aucun résultat d\'analyse disponible'}), 404

@blueprint.route('/clear-data')
@login_required
def clear_data():
    """Route pour effacer les données de la session"""
    # Supprimer le fichier temporaire s'il existe
    if 'temp_file' in session:
        try:
            os.remove(session['temp_file'])
        except:
            pass
    
    session.pop('temp_file', None)
    session.pop('analysis_results', None)
    session.pop('filename', None)
    return jsonify({'success': True, 'message': 'Données effacées'})

@blueprint.route('/get-model-info')
@login_required
def get_model_info():
    """Route pour récupérer les informations du modèle"""
    return jsonify({
        'accuracy': 98.71,
        'models_loaded': False,  # Mode démonstration
        'class_names': ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
                       'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan',
                       'SSH-Patator', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS']
    })

@blueprint.route('/chart/<filename>')
@login_required
def serve_chart(filename):
    """Route pour servir les graphiques"""
    try:
        chart_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(chart_path):
            from flask import send_file
            return send_file(chart_path, mimetype='image/png')
        else:
            return "Graphique non trouvé", 404
    except Exception as e:
        return f"Erreur: {str(e)}", 500

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:
        # Exclure profile car il a sa propre route dans authentication
        if template == 'profile' or template == 'profile.html':
            return redirect(url_for('authentication_blueprint.profile'))

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Friendly titles for breadcrumb based on template name
        page_titles = {
            'api-view.html': 'Manuel',
            'index.html': 'Tableau de Bord',
            'upload.html': 'Analyse de Fichiers CSV',
            'analysis.html': 'Analyse de Fichier'
        }

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template(
            "home/" + template,
            segment=segment,
            page_title=page_titles.get(template),
            API_GENERATOR=len(API_GENERATOR)
        )

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
