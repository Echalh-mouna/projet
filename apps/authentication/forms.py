# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import Email, DataRequired, EqualTo, Length

# login and registration


class LoginForm(FlaskForm):
    username = StringField('Username',
                         id='username_login',
                         validators=[DataRequired()])
    password = PasswordField('Password',
                             id='pwd_login',
                             validators=[DataRequired()])


class CreateAccountForm(FlaskForm):
    username = StringField('Username',
                         id='username_create',
                         validators=[DataRequired()])
    email = StringField('Email',
                      id='email_create',
                      validators=[DataRequired(), Email()])
    password = PasswordField('Password',
                             id='pwd_create',
                             validators=[DataRequired()])


class UpdateProfileForm(FlaskForm):
    username = StringField('Nom d\'utilisateur', render_kw={'readonly': True})
    full_name = StringField('Nom complet', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    mobile = StringField('Téléphone')
    location = StringField('Localisation')


class ChangePasswordForm(FlaskForm):
    old_password = PasswordField('Ancien mot de passe', 
                               validators=[DataRequired()])
    new_password = PasswordField('Nouveau mot de passe', 
                                validators=[DataRequired(), Length(min=6, message='Le mot de passe doit contenir au moins 6 caractères')])
    confirm_password = PasswordField('Confirmer le mot de passe', 
                                   validators=[DataRequired(), EqualTo('new_password', message='Les mots de passe ne correspondent pas')])
