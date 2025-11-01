# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import json
from datetime import datetime

# from flask_restx import Resource, Api

import flask
from flask import render_template, redirect, request, url_for, flash
from flask_login import (
    current_user,
    login_user,
    logout_user,
    login_required
)

from flask_dance.contrib.github import github

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm, UpdateProfileForm, ChangePasswordForm
from apps.authentication.models import Users

from apps.authentication.util import verify_pass, generate_token, hash_pass

# Bind API -> Auth BP (désactivé)
# api = Api(blueprint)

@blueprint.route('/')
def route_default():
    from flask_login import current_user
    if current_user.is_authenticated:
        return redirect(url_for('home_blueprint.upload'))
    else:
        return redirect(url_for('authentication_blueprint.login'))

# Login & Registration

@blueprint.route("/github")
def login_github():
    """ Github login """
    if not github.authorized:
        return redirect(url_for("github.login"))

    res = github.get("/user")
    return redirect(url_for('home_blueprint.upload'))

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)

    if flask.request.method == 'POST':

        # read form data
        username = request.form['username']
        password = request.form['password']

        #return 'Login: ' + username + ' / ' + password

        # Locate user
        user = Users.query.filter_by(username=username).first()
        print(f"Recherche utilisateur: {username}")
        print(f"Utilisateur trouvé: {user is not None}")

        # Check the password
        if user and verify_pass(password, user.password):
            login_user(user)
            print(f"Utilisateur {username} connecté avec succès, redirection vers /upload")
            return redirect('/upload')
        else:
            print(f"Échec de connexion pour {username}")
            # Something (user or pass) is not ok
            return render_template('accounts/login.html',
                                   msg='Wrong user or password',
                                   form=login_form)

    if current_user.is_authenticated:
        return redirect(url_for('home_blueprint.upload'))
    else:
        return render_template('accounts/login.html',
                               form=login_form) 


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username already registered',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email already registered',
                                   success=False,
                                   form=create_account_form)

        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        # Delete user from session
        logout_user()

        return render_template('accounts/register.html',
                               msg='User created successfully.',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)

@blueprint.route('/login/jwt/', methods=['POST'])
def login_jwt():
    try:
        data = request.get_json()
        if not data:
            return {
                'message': 'username or password is missing',
                "data": None,
                'success': False
            }, 400

        username = data.get('username')
        password = data.get('password')
       

        # Recherche utilisateur dans la base
        user = Users.query.filter_by(username=username).first()

        if user and verify_pass(password, user.password):
            # Génère un token s'il n'existe pas encore
            if not user.api_token or user.api_token == '':
                user.api_token = generate_token(user.id)
                user.api_token_ts = int(datetime.utcnow().timestamp())
                db.session.commit()

            return {
                "message": "Successfully fetched auth token",
                "success": True,
                "data": user.api_token
            }
        else:
            return {
                'message': 'username or password is wrong',
                'success': False
            }, 403

    except Exception as e:
        return {
            "error": "Something went wrong",
            "success": False,
            "message": str(e)
        }, 500



@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))


@blueprint.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Route pour afficher et modifier le profil utilisateur"""
    update_profile_form = UpdateProfileForm()
    change_password_form = ChangePasswordForm()
    password_message = None
    
    if request.method == 'POST':
        if 'update_profile' in request.form:
            # Traitement de la mise à jour du profil
            if update_profile_form.validate_on_submit():
                current_user.full_name = update_profile_form.full_name.data
                current_user.email = update_profile_form.email.data
                current_user.mobile = update_profile_form.mobile.data
                current_user.location = update_profile_form.location.data
                
                db.session.commit()
                flash('Profil mis à jour avec succès!', 'success')
                return redirect(url_for('authentication_blueprint.profile'))
        
        elif 'change_password' in request.form:
            # Traitement du changement de mot de passe
            if change_password_form.validate_on_submit():
                old_password = change_password_form.old_password.data
                new_password = change_password_form.new_password.data
                
                # Vérifier l'ancien mot de passe
                if verify_pass(old_password, current_user.password):
                    # Mettre à jour le mot de passe
                    current_user.password = hash_pass(new_password)
                    db.session.commit()
                    password_message = "Mot de passe changé avec succès!"
                else:
                    password_message = "Ancien mot de passe incorrect!"
    
    return render_template('home/profile.html', 
                         update_profile_form=update_profile_form,
                         change_password_form=change_password_form,
                         password_message=password_message,
                         page_title='Profil',
                         segment='profile')


@blueprint.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    """Route pour mettre à jour le profil utilisateur"""
    update_profile_form = UpdateProfileForm(request.form)
    
    if update_profile_form.validate_on_submit():
        current_user.full_name = update_profile_form.full_name.data
        current_user.email = update_profile_form.email.data
        current_user.mobile = update_profile_form.mobile.data
        current_user.location = update_profile_form.location.data
        
        db.session.commit()
        flash('Profil mis à jour avec succès!', 'success')
    else:
        flash('Erreur lors de la mise à jour du profil', 'error')
    
    return redirect(url_for('authentication_blueprint.profile'))


@blueprint.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Route pour changer le mot de passe"""
    change_password_form = ChangePasswordForm(request.form)
    password_message = None
    
    if change_password_form.validate_on_submit():
        old_password = change_password_form.old_password.data
        new_password = change_password_form.new_password.data
        
        # Vérifier l'ancien mot de passe
        if verify_pass(old_password, current_user.password):
            # Mettre à jour le mot de passe
            current_user.password = hash_pass(new_password)
            db.session.commit()
            password_message = "Mot de passe changé avec succès!"
        else:
            password_message = "Ancien mot de passe incorrect!"
    else:
        password_message = "Erreur de validation du formulaire"
    
    return redirect(url_for('authentication_blueprint.profile', password_message=password_message)) 

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
