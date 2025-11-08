# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
import pandas as pd
import base64
from collections import Counter
from werkzeug.utils import secure_filename
from flask import render_template, request, jsonify, redirect, url_for, session
from flask_login import login_required
from jinja2 import TemplateNotFound
from apps.config import API_GENERATOR
from apps.home import blueprint


# ------------------------------
# Configuration & constantes
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pcap', 'pcapng'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_NAMES = [
    'Botnet', 'Brute Force', 'DDoS', 'Exploit', 'Normal',
    'Other', 'Port Scan', 'Shellcode', 'Worm'
]

# ------------------------------
# Fonctions utilitaires
# ------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_saved_accuracies():
    """Charge les précisions sauvegardées ou renvoie les valeurs par défaut."""
    return {
        'rf': 91.35,
        'xgb': 91.91,
        'dnn': 80.21,
        'cnn': 83.69,
        'ensemble': 91.55
    }

# ------------------------------
# Routes principales
# ------------------------------

@blueprint.route('/dashboard')
@login_required
def dashboard():
    accuracies = load_saved_accuracies()
    return render_template(
        'home/dashboard.html',
        segment='dashboard',
        page_title='Tableau de Bord SafeNet',
        API_GENERATOR=len(API_GENERATOR),
        accuracies=accuracies
    )

@blueprint.route('/upload')
@login_required
def upload():
    return render_template(
        'home/upload.html',
        segment='upload',
        page_title='Analyse de Fichiers Réseau',
        API_GENERATOR=len(API_GENERATOR)
    )

@blueprint.route('/analysis')
@login_required
def analysis():
    """Affiche les résultats de la dernière analyse."""
    results = session.get('analysis_results', {})

    if not results or not results.get('success', False):
        return render_template(
            'home/analysis.html',
            segment='analysis',
            page_title='Analyse de Fichier',
            error="Aucun résultat trouvé. Lancez une analyse pour voir les prédictions.",
            results=None
        )

    # Calcul des statistiques pour le template
    total_flows = results.get('total_samples', 0)
    class_dist = results.get('class_distribution', {})
    normal_flows = class_dist.get('Normal', 0)
    attack_flows = total_flows - normal_flows
    threat_percentage = (attack_flows / total_flows * 100) if total_flows > 0 else 0

    # Préparation du résumé des attaques
    attack_summary = []
    for attack_type, count in class_dist.items():
        if attack_type != 'Normal' and count > 0:
            attack_summary.append({
                'type': attack_type,
                'count': count,
                'percentage': round((count / total_flows * 100), 2) if total_flows > 0 else 0
            })

    # Charger les graphiques depuis les fichiers si disponibles
    attack_chart_base64 = None
    feature_chart_base64 = None
    
    if results.get('attack_chart_file'):
        chart_path = os.path.join(UPLOAD_FOLDER, results['attack_chart_file'])
        if os.path.exists(chart_path):
            with open(chart_path, 'rb') as f:
                attack_chart_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    if results.get('feature_chart_file'):
        chart_path = os.path.join(UPLOAD_FOLDER, results['feature_chart_file'])
        if os.path.exists(chart_path):
            with open(chart_path, 'rb') as f:
                feature_chart_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Ajouter les graphiques aux résultats pour le template
    results_with_charts = results.copy()
    if attack_chart_base64:
        results_with_charts['attack_chart'] = attack_chart_base64
    if feature_chart_base64:
        results_with_charts['feature_chart'] = feature_chart_base64

    return render_template(
        'home/analysis.html',
        segment='analysis',
        page_title="Résultats de l'Analyse",
        results=results_with_charts,
        total_flows=total_flows,
        normal_flows=normal_flows,
        attack_flows=attack_flows,
        threat_percentage=round(threat_percentage, 2),
        attack_summary=attack_summary,
        metrics=results.get('metrics', {}),
        confusion_matrix=results.get('confusion_matrix', None)
    )

# ------------------------------
# Upload de fichiers
# ------------------------------

@blueprint.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Téléversement et extraction de caractéristiques depuis un fichier PCAP."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier sélectionné.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Format non supporté. Seuls .pcap et .pcapng sont acceptés.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Extraction des caractéristiques
        from apps.home.extractor import extract_features_from_pcap
        features_df = extract_features_from_pcap(filepath)

        if features_df is None or len(features_df) == 0:
            return jsonify({'error': 'Aucun flux réseau valide détecté dans le fichier PCAP.'}), 400

        if len(features_df) > 1000:
            features_df = features_df.head(1000)

        temp_file = os.path.join(UPLOAD_FOLDER, f"temp_{filename}.pkl")
        features_df.to_pickle(temp_file)

        session['temp_file'] = temp_file
        session['filename'] = filename
        session['selected_model'] = request.form.get('model', 'ensemble')

        return jsonify({
            'success': True,
            'message': 'Fichier PCAP traité et features extraites avec succès.',
            'filename': filename,
            'flows_detected': len(features_df)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Erreur lors du téléversement : {str(e)}'}), 500

# ------------------------------
# Analyse et prédiction
# ------------------------------

@blueprint.route('/analyze-data', methods=['POST'])
@login_required
def analyze_data():
    try:
        if 'temp_file' not in session:
            return jsonify({'error': 'Aucune donnée à analyser'}), 400

        df = pd.read_pickle(session['temp_file'])
        accuracies = load_saved_accuracies()

        from apps.home.extractor import predict_with_model
        model_type = request.form.get('model') or session.get('selected_model', 'ensemble')

        prediction_results = predict_with_model(df, model_type)
        if not prediction_results:
            return jsonify({'error': "Résultats de prédiction vides."}), 500

        if "error" in prediction_results:
            return jsonify({'error': f"Erreur pendant la prédiction : {prediction_results['error']}"}), 500

        # predict_with_model retourne un dictionnaire avec 'predicted_classes' comme dict
        class_distribution = prediction_results.get('predicted_classes', {})
        
        if class_distribution and len(class_distribution) > 0:
            results = {
                'success': True,
                'total_samples': prediction_results.get('total_samples', len(df)),
                'model_used': prediction_results.get('model_used', model_type),
                'model_accuracy': accuracies.get(model_type, 0),
                'class_distribution': class_distribution,
                'predictions': prediction_results.get('samples_preview', [])
            }

            # ✅ Génération des visualisations (convertir dict en liste pour les graphiques)
            # Note: On ne stocke PAS les graphiques dans la session (trop volumineux pour les cookies)
            # Ils seront régénérés dans la route /analysis si nécessaire
            from apps.home.extractor import plot_attack_distribution, plot_feature_importance
            try:
                # Convertir le dictionnaire en liste pour les graphiques
                preds_list = []
                for cls, count in class_distribution.items():
                    preds_list.extend([cls] * count)
                
                if preds_list:
                    # Stocker les graphiques dans des fichiers temporaires au lieu de la session
                    import base64
                    charts_dir = os.path.join(UPLOAD_FOLDER, 'charts')
                    os.makedirs(charts_dir, exist_ok=True)
                    
                    session_id = session.get('session_id', str(hash(str(session))))
                    attack_chart = plot_attack_distribution(preds_list)
                    if attack_chart:
                        attack_file = os.path.join(charts_dir, f'attack_{session_id}.png')
                        with open(attack_file, 'wb') as f:
                            f.write(base64.b64decode(attack_chart))
                        results['attack_chart_file'] = f'charts/attack_{session_id}.png'
                    
                    feature_chart = plot_feature_importance(df, preds_list[:len(df)])
                    if feature_chart:
                        feature_file = os.path.join(charts_dir, f'feature_{session_id}.png')
                        with open(feature_file, 'wb') as f:
                            f.write(base64.b64decode(feature_chart))
                        results['feature_chart_file'] = f'charts/feature_{session_id}.png'
            except Exception as e:
                print(f"[⚠️] Erreur génération graphiques : {e}")
                import traceback
                traceback.print_exc()

        else:
            results = {
                'success': False,
                'message': "Aucune prédiction valide trouvée.",
                'total_samples': len(df)
            }

        # ✅ Stockage des résultats SANS les graphiques (trop volumineux pour les cookies)
        # On stocke seulement les métadonnées
        results_for_session = {
            'success': results.get('success'),
            'total_samples': results.get('total_samples'),
            'model_used': results.get('model_used'),
            'model_accuracy': results.get('model_accuracy'),
            'class_distribution': results.get('class_distribution'),
            'attack_chart_file': results.get('attack_chart_file'),
            'feature_chart_file': results.get('feature_chart_file')
        }
        session['analysis_results'] = results_for_session

        # ✅ Réponse JSON valide vers le frontend
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[❌] Erreur analyse_data : {e}")
        return jsonify({'error': f'Erreur analyse : {str(e)}'}), 500


# ------------------------------
# Gestion session
# ------------------------------

@blueprint.route('/clear-data')
@login_required
def clear_data():
    if 'temp_file' in session:
        try:
            os.remove(session['temp_file'])
        except:
            pass
    session.pop('temp_file', None)
    session.pop('analysis_results', None)
    session.pop('filename', None)
    return jsonify({'success': True, 'message': 'Données effacées'})

# ------------------------------
# Gestion du template
# ------------------------------

@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if template == 'profile' or template == 'profile.html':
            return redirect(url_for('authentication_blueprint.profile'))

        if not template.endswith('.html'):
            template += '.html'

        segment = get_segment(request)
        page_titles = {
            'dashboard.html': 'Tableau de Bord',
            'upload.html': 'Analyse de Fichiers Réseau',
            'analysis.html': 'Analyse de Fichier'
        }

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

def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None
