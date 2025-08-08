# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import json
from datetime import datetime

# from flask_restx import Resource, Api

import flask
from flask import render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_user,
    logout_user
)

from flask_dance.contrib.github import github

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users

from apps.authentication.util import verify_pass, generate_token

# Bind API -> Auth BP (désactivé)
# api = Api(blueprint)

@blueprint.route('/')
def route_default():
    from flask_login import current_user
    if current_user.is_authenticated:
        return redirect(url_for('home_blueprint.index'))
    else:
        return redirect(url_for('authentication_blueprint.login'))

# Login & Registration

@blueprint.route("/github")
def login_github():
    """ Github login """
    if not github.authorized:
        return redirect(url_for("github.login"))

    res = github.get("/user")
    return redirect(url_for('home_blueprint.index'))

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
            print(f"Utilisateur {username} connecté avec succès, redirection vers /index")
            return redirect('/index')
        else:
            print(f"Échec de connexion pour {username}")
            # Something (user or pass) is not ok
            return render_template('accounts/login.html',
                                   msg='Wrong user or password',
                                   form=login_form)

    if current_user.is_authenticated:
        return redirect(url_for('home_blueprint.index'))
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
