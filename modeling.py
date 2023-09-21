import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from pycaret.classification import ClassificationExperiment
from pycaret.regression import *
import streamlit as st


def reading_data(path):
    if not os.path.exists(path):
        raise "Error Reading The File!!"
    data = pd.read_csv(path)
    return data, data.describe()


def dealing_with_null_values(data, null_values):
    if null_values == 'User Imputer':
        mean_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        for column in data.columns:
            if data[column].dtype in ['str', 'object']:
                data[column].fillna(data[column].mode()[0], inplace=True)
                data[column] = mean_imputer.fit_transform(data[column].values.reshape(-1, 1))
    else:
        data.dropna(inplace=True)
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
                data = data.drop(column, axis=1)
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


def preprocess(data, encoder, scaler, chosen, null_values):
    # Detecting Columns with Only Two Unique Values
    for column in data.columns:
        if len(data[column].unique()) == 2:
            data[column] = data[column].astype('str')
    # Dropping The ID Column
    pre_data = data.drop(chosen, axis=1)
    # Apply 
    pre_data = dealing_with_null_values(pre_data, null_values)
    pre_data = scaling(pre_data, scaler)
    pre_data = encoding(pre_data, encoder)
    return pre_data


def modeling(data, target):
    if len(data[target].unique()) == 2:
        cs = ClassificationExperiment()
        cs = cs.setup(data, target=target)
        best = cs.compare_models()
        models = cs.pull()
    else:
        s = setup(data=data, target=target)
        best = compare_models()
        models = pull()
    return best, models
