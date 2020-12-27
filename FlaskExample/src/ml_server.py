import os
import pickle

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect, send_from_directory, flash

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField

import ensembles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='html', static_folder="static")
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)

randomforest = ensembles.RandomForestMSE(0)
boosting = ensembles.GradientBoostingMSE

class ModelFormForest(FlaskForm):
    n_estimators = StringField('Количество деревьев', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля подпространства признаков для одного дерева(число от 0 до 1)', validators=[DataRequired()])
    train_path = FileField('Путь до файла с датасетом для обучения', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    target_train = FileField('Путь до файла с целевыми значениями на обучающей выборке', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    test_path = FileField('Путь до файла с датасетом для предсказания', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Start')


class ModelFormBoosting(FlaskForm):
    n_estimators = StringField('Количество деревьев', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля подпространства признаков для одного дерева(число от 0 до 1)',
                                         validators=[DataRequired()])
    learning_rate = StringField('Темп обучения(число от 0 до 1)', validators=[DataRequired()])
    train_path = FileField('Путь до файла с датасетом для обучения', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    target_train = FileField('Путь до файла с целевыми значениями на обучающей выборке', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    test_path = FileField('Путь до файла с датасетом для предсказания', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Start')


class ResponseInfoForest(FlaskForm):
    n_estimators = StringField('Количество деревьев', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля подпространства признаков для одного дерева(число от 0 до 1)', validators=[DataRequired()])
    rmse = StringField('Конечный RMSE', validators=[DataRequired()])
    submit = SubmitField('Скачать предсказанные целевые значения')

class ResponseInfoBoosting(FlaskForm):
    n_estimators = StringField('Количество деревьев', validators=[DataRequired()])
    max_depth = StringField('Максимальная глубина', validators=[DataRequired()])
    feature_subsample_size = StringField('Доля подпространства признаков для одного дерева',
                                         validators=[DataRequired()])
    learning_rate = StringField('Темп обучения', validators=[DataRequired()])
    rmse = StringField('Конечный RMSE', validators=[DataRequired()])
    submit = SubmitField('Скачать предсказанные целевые значения')

class FileForm(FlaskForm):
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Open File')


def plot_rmse(rmse):
    fig = plt.figure(figsize=(15, 7))

    plt.plot(np.arange(rmse.size)+1, rmse)

    plt.title("Зависимость RMSE от количества деревьев", fontsize=20)
    plt.grid()
    plt.xlabel("Количество деревьев", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)
    pdf_req = "./static/RandForDepth.pdf"
    fig.savefig(pdf_req)

def model_train_forest(n_estimators, max_depth, feature_subsample_size, train_path, test_path, target_train):
    try:
        global randomforest
        randomforest = ensembles.RandomForestMSE(n_estimators, max_depth, feature_subsample_size)
        X_train = pd.read_csv(train_path).to_numpy()
        y_train = pd.read_csv(target_train).to_numpy().ravel()
        X_test = pd.read_csv(test_path).to_numpy()
        rmse_list = randomforest.fit(X_train, y_train, X_train, y_train)
        pred_y = randomforest.predict(X_test)
        temp = pd.DataFrame(pred_y)
        temp.to_csv("./predictions/predicted.csv", index=False)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        rmse_list = -1

    return rmse_list


def model_train_boosting(n_estimators, learning_rate, max_depth,
                         feature_subsample_size, train_path, test_path, target_train):
    try:
        global randomforest
        boosting = ensembles.GradientBoostingMSE(n_estimators, learning_rate, max_depth, feature_subsample_size)
        X_train = pd.read_csv(train_path).to_numpy()
        y_train = pd.read_csv(target_train).to_numpy().ravel()
        X_test = pd.read_csv(test_path).to_numpy()
        rmse_list = boosting.fit(X_train, y_train, X_train, y_train)
        pred_y = boosting.predict(X_test)
        temp = pd.DataFrame(pred_y)
        temp.to_csv("./predictions/predicted.csv", index=False)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        rmse_list = -1

    return rmse_list


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/new_model_forest', methods=['GET', 'POST'])
def new_model_forest():
    try:
        model_params = ModelFormForest()

        if model_params.validate_on_submit():
            n_estimators = int(model_params.n_estimators.data)
            max_depth = int(model_params.max_depth.data)
            feature_subsample_size = float(model_params.feature_subsample_size.data)
            train_path = model_params.train_path.data
            test_path = model_params.test_path.data
            target_train = model_params.target_train.data
            rmse_list = model_train_forest(n_estimators, max_depth, feature_subsample_size, train_path,
                                   test_path, target_train)
            return redirect(url_for('model_info_forest', n_estimators=n_estimators, max_depth=max_depth,
                                    feature_subsample_size=feature_subsample_size,
                                    rmse_list=rmse_list))

        return render_template('from_form.html', form=model_params)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/new_model_boosting', methods=['GET', 'POST'])
def new_model_boosting():
    try:
        model_params = ModelFormBoosting()

        if model_params.validate_on_submit():
            n_estimators = int(model_params.n_estimators.data)
            max_depth = int(model_params.max_depth.data)
            feature_subsample_size = float(model_params.feature_subsample_size.data)
            learning_rate = float(model_params.learning_rate.data)
            train_path = model_params.train_path.data
            test_path = model_params.test_path.data
            target_train = model_params.target_train.data
            rmse_list= model_train_boosting(n_estimators, learning_rate, max_depth,
                                                   feature_subsample_size, train_path,
                                                   test_path, target_train)
            return redirect(url_for('model_info_boosting', n_estimators=n_estimators, learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    feature_subsample_size=feature_subsample_size,
                                    rmse_list=rmse_list))

        return render_template('from_form.html', form=model_params)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/result_forest', methods=['GET', 'POST'])
def model_info_forest():
    try:
        response_form = ResponseInfoForest()

        if response_form.validate_on_submit():
            return redirect(url_for('download_file'))
        print(request.args.get('rmse_list'))
        rmse_list = np.fromstring(request.args.get('rmse_list')[1:-1], sep=" ")
        plot_rmse(rmse_list)
        response_form.n_estimators.data = str(request.args.get('n_estimators'))
        response_form.max_depth.data = str(request.args.get('max_depth'))
        response_form.feature_subsample_size.data = request.args.get('feature_subsample_size')
        response_form.rmse.data = rmse_list[-1]
        return render_template('from_form_result.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/result_boosting', methods=['GET', 'POST'])
def model_info_boosting():
    try:
        response_form = ResponseInfoBoosting()

        if response_form.validate_on_submit():
            return redirect(url_for('download_file'))
        rmse_list = np.fromstring(request.args.get('rmse_list')[1:-1], sep=" ")

        plot_rmse(rmse_list)
        response_form.n_estimators.data = str(request.args.get('n_estimators'))
        response_form.max_depth.data = str(request.args.get('max_depth'))
        response_form.feature_subsample_size.data = request.args.get('feature_subsample_size')
        response_form.learning_rate.data = request.args.get('learning_rate')
        response_form.rmse.data = rmse_list[-1]
        return render_template('from_form_result.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/predictions')
def download_file():
    file_path = "predictions/predicted.csv"
    if os.path.exists(file_path):
        return send_from_directory("predictions", "predicted.csv", as_attachment=True)
    else:
        flash("Произошла ошибка скачивания")