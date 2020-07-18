# -*- coding: utf-8 -*-
import itertools
from matplotlib import cm
from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import pickle
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from io import StringIO
import logging
from statsmodels.tsa.stattools import adfuller
from dotenv import find_dotenv, load_dotenv
import boto3
import matplotlib
import statsmodels.api as sm
import umap
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from cycler import cycler
import datetime
from mpl_toolkits.mplot3d import Axes3D
import umap.plot as up
import io
from functools import reduce
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'macro-forecasting1301' # s3 bucket name
TRAINING_SAMPLE_PERCENT = 0.8 # for both classical and ml models, % of sample to use as training/val. 1-% is test set.
VALIDATION_SAMPLE_PERCENT = 0.8 # for both classical and ml models, % of the training data to use as validation set in walk-forward validation

class ClassicalModels():

    '''Loads up features, splits into train/test, de-means and standardizes.
    Fits univariate AR model, then a 2 regime univariate Markov Switching model with time varying variance,
    saves validation set forecasts, models, and error metrics'''
    
    def __init__(self, logger):
        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

    def get_data(self):

        '''Reads in csv from s3'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='features.csv')
        self.features_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data...')

    def splice_test_data(self):

        '''Sets aside test data up front, saves to s3 for evaluation later. Remainder will be used in walk-forward validation to train+tune parameters.'''
        nobs = len(self.features_df)
        n_init_training_val = int(nobs * TRAINING_SAMPLE_PERCENT)
        self.test_df = self.features_df.iloc[n_init_training_val:, :]
        self.train_val_df = self.features_df.iloc[0:n_init_training_val, :]
        pth = Path(self.data_path, 'test_classical').with_suffix('.csv')
        self.test_df.to_csv(pth)

    def examine_autocorr_stationary(self):

        '''The stationarity procedures per the authors of this dataset should have de-trended / made stationary. Visual examination of autocorrelation.'''
        yvar = self.train_val_df['BAAFFM']
        fig = plot_acf(yvar, lags=36)
        pth = Path(self.graphics_path, 'acf_plot').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of autocorrelation plot of BAAFFM')

        result = adfuller(yvar)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')

        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[1] < 0.05:
            print('series is stationary')
        else:
            print('series is still not stationary')
        self.logger.info('calculated augmented Dickey-Fuller test for stationary')

    def train_ss_AR(self):

        '''Trains state space SARIMAX model (specified as purely AR process, already integrated) as baseline, including walk forward validation and standardization.
        Optimizes lag number by calculating RMSE on validation set during walk-forward training (auto-lag optimization through scipy only available for AIC).'''
        self.error_metrics_AR = {}
        self.forecasts_AR = {}
        self.AR_models = {}

        def __train_one_lag(ll):

            endog = self.train_val_df['BAAFFM']
            forecasts = {}
            
            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT)
            scaler = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training]
            training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values.reshape(-1, 1)))
            mod = sm.tsa.SARIMAX(training_endog_preprocessed, order=(ll, 0, 0), trend='c') # 1 lag, already stationary
            res = mod.fit(disp=0)

            # Save initial forecast
            forecasts[self.train_val_df.iloc[n_init_training, 1]] = scaler.inverse_transform(res.predict(start=len(training_endog_preprocessed), end=len(training_endog_preprocessed)))[0]

            # Step through the rest of the sample
            for t in range(n_init_training, nobs-1):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
    
                mod = sm.tsa.SARIMAX(endog_preprocessed, order=(ll, 0, 0), trend='c')
                res = mod.fit(disp=0) # re-fit

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t+1, 1]] = scaler.inverse_transform(res.predict(start=len(endog_preprocessed), end=len(endog_preprocessed)))[0]
                # save the model at end of time series
                if t == nobs-2:
                    self.AR_models['lag_'+str(ll)] = res
                    self.scaler_AR = scaler
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = self.train_val_df.tail(forecasts.shape[0])[['sasdate', 'BAAFFM']]
            actuals.columns = ['sasdate', 't_actual']
            self.SS_AR_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')
            self.SS_AR_forecasts['sasdate'] = pd.to_datetime(self.SS_AR_forecasts['sasdate'])
            # error storage
            self.error_metrics_AR[ll] = mean_squared_error(self.SS_AR_forecasts['t_actual'], self.SS_AR_forecasts['t_forecast'])
            # forecast storage
            self.forecasts_AR['lag_'+str(ll)] = {
                'df': self.SS_AR_forecasts
            }
            self.logger.info('completed training for AR model with lag: '+str(ll))

        [__train_one_lag(lag_value) for lag_value in range(1, 13)]

        # save dictionary of models to disk for later use
        pth = Path(self.models_path, 'AR_models').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.AR_models, handle)
        pth = Path(self.models_path, 'scaler_AR').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.scaler_AR, handle)

    def train_MarkovSwitch_AR(self):
    
        '''Trains Markov Switching autoregression on univariate series.
        Allows for time varying covariance. Uses walk forward validation to tune lag order similar to AR.'''
        self.error_metrics_Markov = {}
        self.forecasts_Markov = {}
        self.MKV_models = {}

        def __train_one_lag(ll):

            endog = self.train_val_df['BAAFFM']
            forecasts = {}
            
            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT) 
            scaler_y = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training]
            training_endog_preprocessed = pd.DataFrame(scaler_y.fit_transform(training_endog.values.reshape(-1, 1)))
            mod = sm.tsa.MarkovAutoregression(training_endog_preprocessed,
                                        k_regimes=2,
                                        order=ll,
                                        switching_variance=True,
                                        )
            
            np.random.seed(123)
            res = mod.fit(search_reps=20)

            # Save initial forecast
            print(training_endog_preprocessed)
            print(res.predict(start=0, end=382, probabilities='smoothed', conditional=1))
            forecasts[self.train_val_df.iloc[n_init_training, 1]] = scaler_y.inverse_transform(res.predict(start=len(training_endog_preprocessed), end=len(training_endog_preprocessed), probabilities='predicted'))[0]
            # Step through the rest of the sample
            for t in range(n_init_training, nobs-1):
                scaler_y = StandardScaler()
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler_y.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
    
                mod = sm.tsa.MarkovAutoregression(endog_preprocessed,
                                        k_regimes=2,
                                        order=ll,
                                        switching_variance=True
                )
                res = mod.fit(search_reps=20)

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t+1, 1]] = scaler_y.inverse_transform(res.predict(start=len(endog_preprocessed), end=len(endog_preprocessed), probabilities='predicted'))[0]
                # save the model at end of time series
                if t == nobs-2:
                    self.MKV_models['lag_'+str(ll)] = res
                    self.standard_scaler_Markov = scaler_y
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = self.train_val_df.tail(forecasts.shape[0])[['sasdate', 'BAAFFM']]
            actuals.columns = ['sasdate', 't_actual']
            self.Markov_fcasts = pd.merge(forecasts, actuals, on='sasdate', how='inner').dropna()
            self.Markov_fcasts['sasdate'] = pd.to_datetime(self.Markov_fcasts['sasdate'])
            
            # error storage
            self.error_metrics_Markov[ll] = mean_squared_error(self.Markov_fcasts['t_actual'], self.Markov_fcasts['t_forecast'])
            # forecast storage
            self.forecasts_Markov['lag_'+str(ll)] = {
                'df': self.Markov_fcasts
            }
            self.logger.info('completed training for Markov Switching AR model with lag: '+str(ll))
        
        [__train_one_lag(lag_value) for lag_value in range(2, 7)]

        # save dictionary of models to disk for later use
        pth = Path(self.models_path, 'MKV_models').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.MKV_models, handle)
        pth = Path(self.models_path, 'scaler_Markov').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.standard_scaler_Markov, handle)

    def train_exponential_smoother(self):
        
        '''Trains Holt's Exponential Smoothing model on univariate series.
        Allows for dampened trend, no seasonality. Uses walk forward validation to tune lag order similar to AR.'''
        self.error_metrics_exp = {}
        self.forecasts_exp = {}
        self.EXP_models = {}

        endog = self.train_val_df['BAAFFM']
        forecasts = {}
        
        # Get the number of initial training observations
        nobs = len(endog)
        n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT) 
        scaler_y = StandardScaler()

        # Create model for initial training sample, fit parameters
        training_endog = endog.iloc[:n_init_training]
        training_endog_preprocessed = pd.DataFrame(scaler_y.fit_transform(training_endog.values.reshape(-1, 1)))
        mod = ExponentialSmoothing(training_endog_preprocessed,
                                    trend=None,
                                    seasonal=None
                                    )
        res = mod.fit()

        # Save initial forecast
        forecasts[self.train_val_df.iloc[n_init_training, 1]] = scaler_y.inverse_transform(res.predict(start=len(training_endog_preprocessed), end=len(training_endog_preprocessed)))[0]
        # Step through the rest of the sample
        for t in range(n_init_training, nobs-1):
        
            scaler_y = StandardScaler()
            # Update the results by appending the next observation
            endog_preprocessed = pd.DataFrame(scaler_y.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit

            mod = ExponentialSmoothing(endog_preprocessed,
                                    trend=None,
                                    seasonal=None
                                    )
            res = mod.fit()

            # Save the new set of forecasts, inverse the scaler
            forecasts[self.train_val_df.iloc[t+1, 1]] = scaler_y.inverse_transform(res.predict(start=len(training_endog_preprocessed), end=len(training_endog_preprocessed)))[0]
            # save the model at end of time series
            if t == nobs-2:
                self.EXP_models['exp_weigh_lag_struct'] = res
                self.standard_scaler_Expsmooth = scaler_y
        
        # Combine all forecasts into a dataframe
        forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
        actuals = self.train_val_df.tail(forecasts.shape[0])[['sasdate', 'BAAFFM']]
        actuals.columns = ['sasdate', 't_actual']
        self.Expsmooth_fcasts = pd.merge(forecasts, actuals, on='sasdate', how='inner').dropna()
        self.Expsmooth_fcasts['sasdate'] = pd.to_datetime(self.Expsmooth_fcasts['sasdate'])
        
        # error storage
        self.error_metrics_exp['exp_weigh_lag_struct'] = mean_squared_error(self.Expsmooth_fcasts['t_actual'], self.Expsmooth_fcasts['t_forecast'])
        # forecast storage
        print('Exponential smoothing MSE: ', self.error_metrics_exp)
        self.forecasts_exp['exp_weigh_lag_struct'] = {
            'df': self.Expsmooth_fcasts
        }
        self.logger.info('completed training for Exponential Smoothing model')

        # save dictionary of models to disk for later use
        pth = Path(self.models_path, 'EXP_models').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.EXP_models, handle)
        pth = Path(self.models_path, 'scaler_ExponSmooth').with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(self.standard_scaler_Expsmooth, handle)

    def train_ss_DFM(self):
    
        '''Trains Dynamic Factor model, including walk forward validation and standardization.
        Optimizes lag number & factor number by calculating RMSE on validation set during walk-forward training.'''
        self.error_metrics_DFM = {}
        self.forecasts_DFM = {}

        for ll in range(1, 3):

            endog = self.train_val_df[['BAAFFM', 'INDPRO']]
            forecasts = {}

            # Get the number of initial training observations
            nobs = len(endog)
            n_init_training = int(nobs * 0.8)
            scaler = StandardScaler()

            # Create model for initial training sample, fit parameters
            training_endog = endog.iloc[:n_init_training, :]
            training_endog_preprocessed = pd.DataFrame(scaler.fit_transform(training_endog.values))
            mod = sm.tsa.DynamicFactor(training_endog_preprocessed, k_factors=1, factor_order=ll) ######## CHECK IF YOU NEED ERROR ORDER TOO

            res = mod.fit(low_memory=True, disp=False)

            # Save initial forecast
            forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1][0]

            # Step through the rest of the sample
            for t in range(n_init_training, nobs):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler.fit_transform(endog.iloc[:t+1, :].values)) # re fit
                dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
    
                mod = sm.tsa.DynamicFactor(endog_preprocessed, k_factors=1, factor_order=ll)
                res = mod.fit(low_memory=True, disp=False)
                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t, 1]] = scaler.inverse_transform(res.predict())[len(res.predict())-1][0]
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])

            actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
            actuals = actuals[['BAAFFM', 0]]
            actuals.columns = ['t_actual', 'sasdate']
            self.SS_DFM_forecasts = pd.merge(forecasts, actuals, on='sasdate', how='inner')
            self.SS_DFM_forecasts['sasdate'] = pd.to_datetime(self.SS_DFM_forecasts['sasdate'])
            # error storage
            self.error_metrics_DFM[ll] = mean_squared_error(self.SS_DFM_forecasts['t_actual'], self.SS_DFM_forecasts['t_forecast'])
            # forecast storage
            self.forecasts_DFM['lag_'+str(ll)] = {
                'df': self.SS_DFM_forecasts
            }
            self.logger.info('completed training for DFM model with order: '+str(ll)+' and 2 factors')

    def plot_errors_AR(self):

        '''For validation / lag tuning, plots the errors of the different lag terms of AR once trained'''
        df_error = pd.DataFrame(self.error_metrics_AR.items())
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        plt.scatter(df_error.iloc[:, 0], df_error.iloc[:, 1], color='blue', s=10)
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
        self.logger.info('plotted and saved png file in /reports/figures of AR errors at various lags')

    def plot_errors_Markov(self):
    
        '''For validation / lag tuning, plots the errors of the different lag terms + switching variance/constant once trained'''
        df_error = pd.DataFrame(self.error_metrics_Markov.items())
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        plt.scatter(df_error.iloc[:, 0], df_error.iloc[:, 1], color='blue', s=10)
        
        # Decorations
        ax1.set_xlabel('Lag', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Validation Set RMSE', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        fig.tight_layout()
        plt.legend()
        plt.title('Error Metrics: Markov switching', fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'MKV_errors').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of Markov errors at various parameters')

    def plot_errors_DFM(self):
        
        '''For validation / lag tuning, plots the errors of the different lag terms + switching variance/constant once trained'''
        df_error = pd.DataFrame(self.error_metrics_DFM.items())
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        plt.scatter(df_error.iloc[:, 0], df_error.iloc[:, 1], color='blue', s=10)
        
        # Decorations
        ax1.set_xlabel('Lag', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Validation Set RMSE', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        fig.tight_layout()
        plt.legend()
        plt.title('Error Metrics: Dynamic Factor Model', fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'DFM_errors').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of Dynamic Factor Model errors at various parameters')

    def plot_forecasts_Classical(self, chosen_lag_AR, chosen_lag_Markov, chosen_lag_DFM):
        
        '''Plots and reports forecats (t+1) for both classical models'''
        #df_Markov = self.forecasts_Markov['lag_'+str(chosen_lag_Markov)]['df']
        df_AR = self.forecasts_AR['lag_'+str(chosen_lag_AR)]['df']
        df_ExpSmo = self.forecasts_exp['exp_weigh_lag_struct']['df']
        #df_DFM = self.forecasts_DFM['lag_'+str(chosen_lag_DFM)]['df']

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        
        ax1.plot(df_AR['sasdate'], df_AR['t_actual'], color='dodgerblue', label='actual')
        ax1.plot(df_AR['sasdate'], df_AR['t_forecast'], color='navy', label=('state space AR, lag='+str(chosen_lag_AR)+' forecast'), linestyle=":")
        #ax1.plot(df_Markov['sasdate'], df_Markov['t_forecast'], color='crimson', label=('Markov switching model, lag='+str(chosen_lag_Markov)+' forecast'), linestyle=":")
        ax1.plot(df_ExpSmo['sasdate'], df_ExpSmo['t_forecast'], color='gray', label=('exponential smoothing model forecast'), linestyle=":")
        #ax1.plot(df_DFM['sasdate'], df_DFM['t_forecast'], color='lightskyblue', label=('dynamic factor model, lag='+str(chosen_lag_DFM)+' forecast'), linestyle=":")

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
        pth = Path(self.graphics_path, 'yhat_y_Classical').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of forecasts of Classical models vs. actuals')

    def execute_analysis(self):
        self.get_data()
        self.splice_test_data()
        #self.examine_autocorr_stationary()
        #self.train_ss_AR()
        #self.plot_errors_AR()
        self.train_MarkovSwitch_AR()
        #self.plot_errors_Markov()
        #self.train_exponential_smoother()
        #self.train_ss_DFM()
        #self.plot_forecasts_Classical(chosen_lag_AR=9, chosen_lag_Markov=6, chosen_lag_DFM=2)


def main():
    """ Runs training of machine learning models and hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info('running classical models...')
    UnivariateClassicalModels = ClassicalModels(logger)
    UnivariateClassicalModels.execute_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
