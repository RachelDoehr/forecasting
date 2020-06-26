# -*- coding: utf-8 -*-
import itertools
from sklearn.linear_model import ElasticNet
from pathlib import Path
import pickle
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
TRAINING_SAMPLE_PERCENT = 0.8 # for both classical and ml models, % of sample to use as training/val. 1-% is test set.
VALIDATION_SAMPLE_PERCENT = 0.95 # for both classical and ml models, % of the training data to use as validation set in walk-forward validation

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
                # save the model at end of time series
                if t == nobs-1:
                    self.AR_models['lag_'+str(ll)] = res
                    self.scaler_AR = scaler
        
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
            forecasts[self.train_val_df.iloc[n_init_training-1, 1]] = scaler_y.inverse_transform(res.predict())[len(res.predict())-1]
            # Step through the rest of the sample
            for t in range(n_init_training, nobs):
                scaler_y = StandardScaler()
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(scaler_y.fit_transform(endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
                dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
    
                mod = sm.tsa.MarkovAutoregression(endog_preprocessed,
                                        k_regimes=2,
                                        order=ll,
                                        switching_variance=True
                )
                res = mod.fit(search_reps=20)

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t, 1]] = scaler_y.inverse_transform(res.predict())[len(res.predict())-1]
                # save the model at end of time series
                if t == nobs-1:
                    self.MKV_models['lag_'+str(ll)] = res
                    self.standard_scaler_Markov = scaler_y
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = pd.concat([endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
            actuals.columns = ['t_actual', 'sasdate']
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

    def plot_forecasts_Classical(self, chosen_lag_AR, chosen_lag_Markov):
        
        '''Plots and reports forecats (t+1) for both classical models'''
        df_Markov = self.forecasts_Markov['lag_'+str(chosen_lag_Markov)]['df']
        df_AR = self.forecasts_AR['lag_'+str(chosen_lag_AR)]['df']

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        
        ax1.plot(df_AR['sasdate'], df_AR['t_actual'], color='dodgerblue', label='actual')
        ax1.plot(df_AR['sasdate'], df_AR['t_forecast'], color='navy', label=('state space AR, lag='+str(chosen_lag_AR)+' forecast'), linestyle=":")
        ax1.plot(df_Markov['sasdate'], df_Markov['t_forecast'], color='crimson', label=('Markov switching model, lag='+str(chosen_lag_Markov)+' forecast'), linestyle=":")

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
        self.logger.info('plotted and saved png file in /reports/figures of forecasts of Classical models vs. actuals')

    def execute_analysis(self):
        self.get_data()
        self.splice_test_data()
        self.train_ss_AR()
        self.plot_errors_AR()
        self.train_MarkovSwitch_AR()
        self.plot_errors_Markov()
        self.plot_forecasts_Classical(chosen_lag_AR=9, chosen_lag_Markov=6)

class MLModels():

    '''Loads up features, splits into train/test, de-means and standardizes. Uses full multivariate dataset with different regularization strategies to feature
    select on high dimensional data. Fits elastic net, Adaboost (decision tree), SVM regressor with / without non-linear kernel, and ANN.
    Saves validation set forecasts, models, and error metrics'''
    
    def __init__(self, logger):
        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

        self.error_metrics_EN = {}
        self.forecasts_EN = {}
        self.EN_models = {}

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
        pth = Path(self.data_path, 'test_ml').with_suffix('.csv')
        self.test_df.to_csv(pth)

    def _series_to_supervised(self, df, desired_lags, desired_forecasts, dropnan):
        
        """Frame a time series as a supervised learning dataset."""
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(desired_lags, 0, -1):
            cols.append(df.shift(i))
        for ll in range(desired_lags, 0, -1):
            names.extend([str(col) + '_(t-'+str(ll)+')' for col in df.columns])
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, desired_forecasts):
            cols.append(df.shift(-i))
        names.extend([str(col) + '_(t+'+str(desired_forecasts-1)+')' for col in df.columns])
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def reframe_train_val_df(self):

        '''Uses series to supervised function to turn the transformed training/val dataset into a supervised learning problem, by
        adding lag terms as new columns to the dataframe, as well as t+1 forecast column of desired prediction variables (just one, BAAFFM).'''
        pre_flipped_df = self.train_val_df.copy()
        self.en_lags=1
        reframed = self._series_to_supervised(pre_flipped_df, desired_lags=self.en_lags, desired_forecasts=1, dropnan=True) # note here b/c of time reframing issue, lags=2 -> t and t-1 are used, for 2 time steps back use lags=3
        variables = [c for c in reframed.columns if '(t+0)' not in c and 'sasdate' not in c]
        self.reframed_df = pd.concat([reframed[[v for v in variables if 'Unnamed' not in v]], reframed[['BAAFFM_(t+0)', 'sasdate_(t+0)']]], axis=1)
    
    def train_elastic_net(self):

        '''Trains state space SARIMAX model (specified as purely AR process, already integrated) as baseline, including walk forward validation and standardization.
        Optimizes lag number by calculating RMSE on validation set during walk-forward training (auto-lag optimization through scipy only available for AIC).'''
        forecasts = {
            'l1_ratio':  list(),
            'alpha': list(),
            'forecast': list(),
            'forecast_t': list()
        }
        EN_params = {
            'l1_ratio': np.linspace(0.1,1,11),
            'alpha': np.linspace(0.1,1,11)
        }
        EN_models = list()

        Y = self.reframed_df['BAAFFM_(t+0)']
        X = self.reframed_df[[c for c in self.reframed_df if '(t+0)' not in c]] # removes date col and Y
        dates = self.reframed_df['sasdate_(t+0)']
        
        # Get the number of initial training observations
        nobs = len(Y)
        n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT)

        for time_step in range(0, (nobs-n_init_training-self.en_lags)):
            scaler_X, scaler_y = StandardScaler(), StandardScaler()

            # Create model for training sample, gridsearch and fit parameters
            training_X = X.iloc[:n_init_training+time_step]
            training_X_preprocessed = pd.DataFrame(scaler_X.fit_transform(training_X.values))
            test_X = X.iloc[n_init_training+1+time_step, :]
            test_X_preprocessed = pd.DataFrame(scaler_X.transform(test_X.values.reshape(1, -1)))
            training_Y = Y.iloc[:n_init_training+time_step]
            training_Y_preprocessed = pd.DataFrame(scaler_y.fit_transform(training_Y.values.reshape(-1, 1)))

            for ratios in itertools.product(EN_params['l1_ratio'], EN_params['alpha']):
                regr = ElasticNet(random_state=0, l1_ratio=ratios[0], alpha=ratios[1])
                regr.fit(training_X_preprocessed, training_Y_preprocessed)
                forecasts['l1_ratio'].append(ratios[0])
                forecasts['alpha'].append(ratios[1])
                forecasts['forecast_t'].append(dates.iloc[n_init_training+1+time_step])
                forecasts['forecast'].append(scaler_y.inverse_transform(regr.predict(test_X_preprocessed))[0])

                # save models
                if time_step == (nobs-n_init_training+1-self.en_lags)-1:
                    EN_models.append({
                        'l1_ratio': ratios[0],
                        'alpha': ratios[1],
                        'model': regr
                        })
            print('time step', time_step)

        forecasts = pd.DataFrame(forecasts)
        forecasts.to_csv('a.csv')
        '''
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
        '''
        # save dictionary of models to disk for later use
        with open(Path(self.models_path, 'elastic_net_models').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(self.EN_models, handle)
        with open(Path(self.models_path, 'scaler_elastic_net').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(scaler_X, handle) # last one in memory; full val/train sample
        self.logger.info('completed training for Elastic Net models')

    def execute_analysis(self):

        '''Executes training, scaling, tuning, and error analysis for ml models, saves final models to disk.'''
        self.get_data()
        self.splice_test_data()
        self.reframe_train_val_df()
        self.train_elastic_net()

def main():
    """ Runs training of machine learning models and hyperparameter tuning.
    """
    logger = logging.getLogger(__name__)
    #logger.info('running classical models...')
    #UnivariateClassicalModels = ClassicalModels(logger)
    #UnivariateClassicalModels.execute_analysis()

    logger.info('running ML models...')
    MultivariateMLModels = MLModels(logger)
    MultivariateMLModels.execute_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
