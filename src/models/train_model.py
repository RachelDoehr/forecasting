# -*- coding: utf-8 -*-
from pathlib import Path
from sklearn.metrics import explained_variance_score, mean_squared_error
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
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='features.csv')
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
    
    def train_SS_AR(self):

        '''Trains state space SARIMAX model (specified as purely AR process, already integrated) as baseline, including walk forward validation and standardization'''
        # Setup forecasts
        endog = self.train_val_df['BAAFFM']
        forecasts = {}

        # Get the number of initial training observations
        nobs = len(endog)
        n_init_training = int(nobs * 0.8)
        scaler = StandardScaler()

        # Create model for initial training sample, fit parameters
        training_endog = endog.iloc[:n_init_training]
        training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values.reshape(-1, 1)))
        mod = sm.tsa.SARIMAX(training_endog_preprocessed, order=(3, 0, 0), trend='c') # 1 lag, already stationary
        res = mod.fit()

        # Save initial forecast
        forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]

        # Step through the rest of the sample
        for t in range(n_init_training, nobs):
            # Update the results by appending the next observation
            endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
            dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
  
            mod = sm.tsa.SARIMAX(endog_preprocessed, order=(3, 0, 0), trend='c') 
            res = mod.fit(disp=0) # re-fit

            # Save the new set of forecasts, inverse the scaler
            forecasts[self.train_val_df.iloc[t, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]
    
        # Combine all forecasts into a dataframe
        forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
        actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
        actuals.columns = ['t_actual', 'sasdate']
        self.SS_AR_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')

    def report_errors_SS_AR(self):
        
        '''Calculates and reports error metrics from baseline model'''
        print('MSE: ', mean_squared_error(self.SS_AR_forecasts['t_actual'], self.SS_AR_forecasts['t_forecast']))
        print('Explained variance: ', explained_variance_score(self.SS_AR_forecasts['t_actual'], self.SS_AR_forecasts['t_forecast']))

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
        ax1.plot(self.SS_AR_forecasts['sasdate'], self.SS_AR_forecasts['t_actual'], color='gray', label='actual')
        ax1.plot(self.SS_AR_forecasts['sasdate'], self.SS_AR_forecasts['t_forecast'], color='dodgerblue', label='forecast')

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('BAAFFM Spread', color='tab:red', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
        ax1.grid(alpha=.4)

        fig.tight_layout()
        plt.legend()
        plt.show()
        

    def execute_analysis(self):
        self.get_data()
        self.splice_test_data()
        self.train_SS_AR()
        self.report_errors_SS_AR()

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
