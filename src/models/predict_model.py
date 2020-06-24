# -*- coding: utf-8 -*-
from pathlib import Path
import pickle
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import seaborn as sns
import matplotlib
import statsmodels.api as sm
from matplotlib import pyplot as plt
from cycler import cycler
import datetime
import io
from functools import reduce
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'macro-forecasting1301' # s3 bucket name

class AllModelsPredictor():

    '''Loads up test set and pre-trained classical and ML models with optimized hyperparameters. Predicts on test set,
    calculates errors across models and graphs forecasts'''
    
    def __init__(self, logger):
        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve().joinpath('test_set').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

    def get_data(self):

        '''Reads in csv from s3'''
        pth = Path(self.data_path, 'test').with_suffix('.csv')
        self.test_set = pd.read_csv(pth)
        self.logger.info('loaded data...')

    def predict_AR(self):

        '''Reads in model and scaler, de-means and standardizes y, predicts out of sample values for AR model.
        Stores predicted values, errors in memory for later cross-model comparison and graphing.'''
    