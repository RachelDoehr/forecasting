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
    
    def train_ss_AR(self):

        '''Trains state space SARIMAX model (specified as purely AR process, already integrated) as baseline, including walk forward validation and standardization.
        Optimizes lag number by calculating RMSE on validation set during walk-forward training (auto-lag optimization through scipy only available for AIC).'''
        self.error_metrics_AR = {}
        self.forecasts_AR = {}

        for ll in range(1, 13):

            endog = self.train_val_df['BAAFFM']
            forecasts = {}
            
            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * 0.8)
            scaler = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training]
            training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values.reshape(-1, 1)))
            mod = sm.tsa.SARIMAX(training_endog_preprocessed, order=(ll, 0, 0), trend='c') # 1 lag, already stationary
            res = mod.fit(disp=0)

            # Save initial forecast
            forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]

            # Step through the rest of the sample
            for t in range(n_init_training, nobs):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
                dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
    
                mod = sm.tsa.SARIMAX(endog_preprocessed, order=(ll, 0, 0), trend='c') 
                res = mod.fit(disp=0) # re-fit

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
            actuals.columns = ['t_actual', 'sasdate']
            self.SS_AR_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')
            self.SS_AR_forecasts['sasdate'] = pd.to_datetime(self.SS_AR_forecasts['sasdate'])
            # error storage
            self.error_metrics_AR[ll] = mean_squared_error(self.SS_AR_forecasts['t_actual'], self.SS_AR_forecasts['t_forecast'])
            # forecast storage
            self.forecasts_AR['lag_'+str(ll)] = {
                'df': self.SS_AR_forecasts
            }
            self.logger.info('completed training for AR baseline model with lag: '+str(ll))

    def train_MarkovSwitch_AR(self):
    
        '''Trains Markov Switching autoregression on univariate series. Allows for time-varying transition probabilities, incorporated here as the VIX.
        Uses walk forward validation to tune lag order similar to ssAR fx.'''
        self.error_metrics_Markov = {}
        self.forecasts_Markov = {}

        for ll in range(1, 2):

            endog = self.train_val_df['BAAFFM']
            leading_ind = self.train_val_df['VXOCLSx']
            forecasts = {}
            
            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * 0.8)
            scaler = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training]
            training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values.reshape(-1, 1)))
            training_leading_preprocessed = pd.DataFrame(scaler.fit_transform(leading_ind.iloc[:n_init_training].values.reshape(-1, 1)))
            mod = sm.tsa.MarkovAutoregression(training_endog_preprocessed,
                                        k_regimes=2,
                                        order=4,
                                        switching_variance=True,
                                        exog_tvtp=sm.add_constant(training_leading_preprocessed)
                                        )
            
            np.random.seed(123)
            res = mod.fit(search_reps=10)

            # Save initial forecast
            forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]

            # Step through the rest of the sample
            for t in range(n_init_training, nobs):
                print('Iter: ', t)
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
                leading_preprocessed = pd.DataFrame(scaler.fit_transform(leading_ind.iloc[0:t+1].values.reshape(-1, 1))) 
                dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
    
                mod = sm.tsa.MarkovAutoregression(endog_preprocessed,
                                        k_regimes=2,
                                        order=4,
                                        switching_variance=True,
                                        exog_tvtp=sm.add_constant(leading_preprocessed)
                )
                res = mod.fit(search_reps=10)

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1]
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
            actuals.columns = ['t_actual', 'sasdate']
            self.SS_AR_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')
            self.SS_AR_forecasts['sasdate'] = pd.to_datetime(self.SS_AR_forecasts['sasdate'])
            # error storage
            self.error_metrics_AR[ll] = mean_squared_error(self.SS_AR_forecasts['t_actual'], self.SS_AR_forecasts['t_forecast'])
            # forecast storage
            self.forecasts_AR['lag_'+str(ll)] = {
                'df': self.SS_AR_forecasts
            }
            self.logger.info('completed training for Markov Switching AR model with lag: '+str(ll))



    def plot_forecasts_ss_AR(self, chosen_lag):
        
        '''Plots and reports forecatss / error metrics from baseline model'''
        df = self.forecasts_AR['lag_'+str(chosen_lag)]['df']

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
        
        ax1.plot(df['sasdate'], df['t_actual'], color='dodgerblue', label='actual')
        ax1.plot(df['sasdate'], df['t_forecast'], color='navy', label='forecast', linestyle=":")

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('BAAFFM Spread', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)
        plt.title('Forecast vs. Realized Values: BAAFFM', weight='bold')

        fig.tight_layout()
        plt.legend()
        pth = Path(self.graphics_path, 'yhat_y_AR').with_suffix('.png')
        fig.savefig(pth)

        # MSE
        df_error = pd.DataFrame(self.error_metrics_AR.items())
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        plt.scatter(df_error.iloc[:, 0], df_error.iloc[:, 1], color='blue', s=6)
        
        # Decorations
        ax1.set_xlabel('Lag', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Validation Set RMSE', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )

        fig.tight_layout()
        plt.legend()
        plt.title('Error Metrics: AR', fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'AR_errors').with_suffix('.png')
        fig.savefig(pth)

    def train_ss_DFM(self):
    
        '''Trains Dynamic Factor model, including walk forward validation and standardization.
        Optimizes lag number & factor number by calculating RMSE on validation set during walk-forward training.'''
        self.error_metrics_DFM = {}
        self.forecasts_DFM = {}

        for ll in range(1, 2):

            endog = self.train_val_df.iloc[:, 2:]

            # create correlation matrix
            corr_matrix = endog.corr().abs()
            # select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # find index of feature columns with correlation greater than 0.25
            to_drop = [column for column in upper.columns if any(upper[column] > 0.2) and column != 'BAAFFM']
            # drop correlated (post stationarized) features for computational time purposes 
            endog = endog.drop(endog[to_drop], axis=1)

            BAAFFM_iloc = endog.columns.get_loc('BAAFFM')
            forecasts = {}
            
            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * 0.8)
            scaler = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training, :]
            training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values))
            mod = sm.tsa.DynamicFactor(training_endog_preprocessed, k_factors=2, factor_order=2)
            res = mod.fit(low_memory=True)
            
            # Save initial forecast
            forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1][BAAFFM_iloc]

            # Step through the rest of the sample
            for t in range(n_init_training, nobs):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[:t+1, :].values)) # re fit
                dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
    
                mod = sm.tsa.DynamicFactor(endog_preprocessed, k_factors=2, factor_order=2)
                res = mod.fit(low_memory=True)

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1][BAAFFM_iloc]
                
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
            print(actuals)
            actuals.columns = ['t_actual', 'sasdate']
            self.SS_DFM_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')
            self.SS_DFM_forecasts['sasdate'] = pd.to_datetime(self.SS_DFM_forecasts['sasdate'])
            # error storage
            self.error_metrics_DFM[ll] = mean_squared_error(self.SS_DFM_forecasts['t_actual'], self.SS_DFM_forecasts['t_forecast'])
            # forecast storage
            self.forecasts_DFM['lag_'+str(ll)] = {
                'df': self.SS_DFM_forecasts
            }
            self.logger.info('completed training for DFM model with order: '+str(ll)+' and 1 factors')
            print(self.forecasts_DFM)

    def execute_analysis(self):
        self.get_data()
        self.splice_test_data()
        #self.train_ss_AR()
        #self.plot_forecasts_ss_AR(chosen_lag=1)
        #self.train_ss_DFM()
        self.train_MarkovSwitch_AR()

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
