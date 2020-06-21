# -*- coding: utf-8 -*-
from pathlib import Path
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

class ClassicalModels():

    '''Loads up features, splits into train/test, de-means and standardizes.
    Fits baseline univariate SARIMA model, then full Dynamic Factor Models, saves error metrics'''
    
    def __init__(self, logger):
        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()

    def get_data(self):

        '''Reads in csv from s3'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='current.csv')
        self.features_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data...')

    def splice_test_data(self):

        '''Sets aside test data up front, saves to s3 for evaluation later. Remainder will be used in walk-forward validation to train+tune parameters.'''
        nobs = len(self.features_df)
        n_init_training_val = int(nobs * 0.8)
        self.test_df = self.features_df.iloc[n_init_training_val:, :]
        self.train_val_df = self.features_df.iloc[0:n_init_training_val, :]
        pth = Path(self.data_path, 'test').with_suffix('.csv')
        self.test_df.to_csv(pth)
    
    def train_sARIMA(self):

        '''Trains sARIMA model as baseline, including walk forward validation and standardization'''
        # Setup forecasts
        nforecasts = 3
        endog = self.train_val_df['BAAFFM']
        forecasts = {}

        # Get the number of initial training observations
        nobs = len(endog)
        n_init_training = int(nobs * 0.6)
        scaler = StandardScaler()

        # Create model for initial training sample, fit parameters
        training_endog = endog.iloc[:n_init_training]
        training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values.reshape(-1, 1)))
        mod = sm.tsa.SARIMAX(training_endog_preprocessed, order=(1, 0, 0), trend='c') # 1 lag, already stationary
        res = mod.fit()

        # Save initial forecast
        forecasts[training_endog.index[-1]] = res.forecast(steps=nforecasts)

        # Step through the rest of the sample
        for t in range(n_init_training, nobs):
            # Update the results by appending the next observation
            endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
            updated_endog = endog.iloc[t:t+1]
  
            mod = sm.tsa.SARIMAX(endog_preprocessed, order=(1, 0, 0), trend='c') 
            res = mod.fit() # re-fit

            # Save the new set of forecasts
            forecasts[updated_endog.index[0]] = res.forecast(steps=nforecasts)

        # Combine all forecasts into a dataframe
        forecasts = pd.concat(forecasts, axis=1)

        print(forecasts.iloc[:5, :5])

    def execute_analysis(self):
        self.get_data()
        self.splice_test_data()
        self.train_sARIMA()

def main():
    """ Runs training of classical models and hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info('training classical models')

    Models = ClassicalModels(logger)
    Models.execute_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
