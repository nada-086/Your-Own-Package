import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from pycaret.classification import ClassificationExperiment
import streamlit as st


def reading_data(path):
    if not os.path.exists(path):
        raise "Error Reading The File!!"
    data = pd.read_csv(path)
    return data, data.describe()


def dealing_with_null_values(data):
    mean_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    for column in data.columns:
        if data[column].dtype in ['str', 'object']:
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column] = mean_imputer.fit_transform(data[column].values.reshape(-1, 1))
    return data


def encoding(data, method):
    if method == 'Label Encoder':
        encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype in ['str', 'object']:
                data[column] = encoder.fit_transform(data[column])
    else:
        for column in data.columns:
            if data[column].dtype in ['str', 'object']:
                one_hot = pd.get_dummies(data[column])
                data = pd.concat([data, one_hot], axis=1)
    return data


def scaling(data, method):
    if method == 'MinMax Scaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    for column in data.columns:
        if data[column].dtype not in ['str', 'object']:
            data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))
    return data


def preprocess(data, encoder, scaler, chosen):
    # Dropping The ID Column
    for column in chosen:
        data = data.drop(column, axis=1)
    # Apply 
    data = dealing_with_null_values(data)
    data = encoding(data, encoder)
    data = scaling(data, scaler)
    return data


def modeling(data, target):
    cs = ClassificationExperiment()
    cs = cs.setup(data, target=target)
    best = cs.compare_models()
    models = cs.pull()
    plot = cs.plot_model(best, plot='auc')
    return best, models, plot
