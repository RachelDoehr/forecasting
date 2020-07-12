# -*- coding: utf-8 -*-
import itertools
from matplotlib import cm
from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import matplotlib
import statsmodels.api as sm
import umap
from matplotlib import pyplot as plt
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
        pth = Path(self.graphics_path, 'yhat_y_Classical').with_suffix('.png')
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

        self.error_metrics_RF = {}
        self.forecasts_RF = {}
        self.RF_models = {}

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

    def _series_to_supervised(self, df, n_in, n_out, dropnan=True):

        """Frame a time series as a supervised learning dataset."""
        n_vars = 1 if type(df) is list else df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def reframe_train_val_df(self, maxlags):

        '''Uses series to supervised function to turn the transformed training/val dataset into a supervised learning problem, by
        adding lag terms as new columns to the dataframe, as well as t+1 forecast column of desired prediction variables (just one, BAAFFM).'''
        pre_flipped_df = self.train_val_df.copy()
        self.reframed_dfs = {}

        for lag_val in range(1,  maxlags+1):
            reframed = self._series_to_supervised(pre_flipped_df, lag_val, 1, dropnan=True)
            # removing the variables: sasdate, 'Unnamed' (the indx), and everything at time (t) except for BAAFFM (Y var) and sasdate
            var_num_unnamed = [c for c in reframed.columns if 'var1(' in c or 'var2(' in c] # the index, to remove
            var_num_extra_current = [c for c in reframed.columns if '(t)' in c] # only predicting one var, remove these
            # then add back:
            var_num_sasdate = [c for c in reframed.columns if 'var2(' in c and '(t)' in c] # sasdate, the correct current one
            var_num_baaffm = [c for c in reframed.columns if ('var'+str(pre_flipped_df.columns.get_loc('BAAFFM')+1)+'(') in c and '(t)' in c] # keep this Y var, tack on at the end

            var_num_unnamed.extend(var_num_extra_current)
            var_num_sasdate.extend(var_num_baaffm)
            self.reframed_dfs['lag_'+str(lag_val)] = pd.concat([reframed[[c for c in reframed.columns if c not in var_num_unnamed]], reframed[[c for c in reframed.columns if c in var_num_sasdate]]], axis=1)
        self.logger.info('reframed dataset as supervised learning problem')

    def _mse(self, y):
        r2 = r2_score(y['t_actual'], y['forecast'])
        mse = mean_squared_error(y['t_actual'], y['forecast'])
        return pd.Series(dict(r2=r2, mse=mse))

    def explore_dim_reduce(self, maxlags):
    
        '''Explores dimensionality reduction on stationary, transformed data for potential use in pipeline before supervised model training.'''

        exp_var = []
        for lag_val in range(1, 3):

            Y = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -1]
            X = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, 0:-2]
            dates = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -2]
            
            # Get the number of initial training observations
            nobs = len(Y)
            n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT)
            
            for time_step in range(0, (nobs-n_init_training-1)):
                scaler_X, scaler_y = StandardScaler(), StandardScaler()

                # Create model for training sample, gridsearch and fit parameters
                training_X, training_Y = X.iloc[:n_init_training+time_step], Y.iloc[:n_init_training+time_step]
                training_X_preprocessed, training_Y_preprocessed = pd.DataFrame(scaler_X.fit_transform(training_X.values)), pd.DataFrame(scaler_y.fit_transform(training_Y.values.reshape(-1, 1)))
                
                for N_COMP in range(1, 3):
                    reducer_X = umap.UMAP(random_state=42, n_components=N_COMP)
                    reducer_X.fit(training_X_preprocessed)
                    embedding = reducer_X.transform(training_X_preprocessed)

                    reg = LinearRegression().fit(embedding, training_Y_preprocessed)
                    R2 = reg.score(embedding, training_Y_preprocessed)
                    exp_var.append({
                        'n_comp': N_COMP,
                        'lag': lag_val,
                        'time_step': time_step,
                        'R2': R2
                    })

                if time_step == (nobs-n_init_training-2):
                    fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
                    colormap = plt.get_cmap('seismic')
                    ax1.set_prop_cycle(cycler('color', [colormap(k) for k in np.linspace(0, 1, 4)]) +
                            cycler('linestyle', ['-', ':', '-', ':']))
                    for nn in range(0, N_COMP):
                        plt.plot(pd.to_datetime(dates.iloc[0:n_init_training+time_step]), embedding[:, nn], linewidth=0.5)
    
                    plt.title('UMAP projection of the dataset', fontsize=24)
                    fig.savefig(Path(self.graphics_path, 'umap_projection').with_suffix('.png'))

        summary_exp_var = pd.DataFrame(exp_var)
        summary_exp_var = summary_exp_var.groupby(['n_comp', 'lag'])['R2'].mean().reset_index()

        fig, ax2 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        colormap = plt.get_cmap('seismic')
        ax2.set_prop_cycle(cycler('color', [colormap(k) for k in np.linspace(0, 1, 4)]) +
                cycler('linestyle', ['-', ':', '-', ':']))

        for ll in range(1, 3):
            lag = summary_exp_var[summary_exp_var.lag == ll]
            plt.plot(lag['n_comp'], lag['R2'], '-o', linewidth=0.5, label='lag='+str(ll))

        plt.title('Explained Variance of Y using UMAP at Various n_components / lags', fontsize=24)
        plt.legend()
        plt.xlabel('Number of UMAP components')
        plt.ylabel('Explained variance of Y')
        fig.savefig(Path(self.graphics_path, 'dimension_reduction_selection').with_suffix('.png'))

        self.logger.info('ran uniform manifold approximation/projection')

    def train_elastic_net(self, maxlags):

        '''Trains elastic net models using walk forward validation across a range of hyperparameters l1, l2, and lag order. For each lag value, steps through time
        steps and trains l1/l2 variations, stores forecasts. Calculates MSE for each l1/l2/lag order possibility.'''
        forecasts = {
            'l1_ratio':  list(),
            'alpha': list(),
            'lag': list(),
            'forecast': list(),
            'forecast_t': list()
        }
        EN_params = {
            'l1_ratio': np.linspace(0.001, 1.0, 11),#np.linspace(0.1,1,11),
            'alpha': np.linspace(0.0001,1.0,11)
        }
        EN_models = list()
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        for lag_val in range(1, maxlags+1):

            Y = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -1]
            X = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, 0:-2]
            X = X[[c for c in X.columns if 'var94' in c or 'var8(t-1)' in c or 'var75(t-1)' in c]]

            dates = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -2]
            
            # Get the number of initial training observations
            nobs = len(Y)
            n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT)
            
            for time_step in range(0, (nobs-n_init_training-1)):
                
                # Create model for training sample, gridsearch and fit parameters
                training_X, training_Y = X.iloc[:n_init_training+time_step], Y.iloc[:n_init_training+time_step]
                training_X_preprocessed, training_Y_preprocessed = pd.DataFrame(scaler_X.fit_transform(training_X.values)), pd.DataFrame(scaler_y.fit_transform(training_Y.values.reshape(-1, 1)))
                test_X = X.iloc[n_init_training+1+time_step, :]
                test_X_preprocessed = pd.DataFrame(scaler_X.transform(test_X.values.reshape(1, -1)))
                
                for ratios in itertools.product(EN_params['l1_ratio'], EN_params['alpha']):
                    regr = ElasticNet(random_state=42, l1_ratio=ratios[0], alpha=ratios[1], fit_intercept=False)
                    regr.fit(training_X_preprocessed, training_Y_preprocessed)

                    values = [
                        ratios[0],
                        ratios[1],
                        lag_val,
                        dates.iloc[n_init_training+1+time_step],
                        scaler_y.inverse_transform(regr.predict(test_X_preprocessed))[0]
                    ]
                    [forecasts[k].append(v) for k, v in zip(['l1_ratio', 'alpha', 'lag', 'forecast_t', 'forecast'], values)]

                    # save models
                    if time_step == (nobs-n_init_training-2):
                        EN_models.append({
                            'l1_ratio': ratios[0],
                            'alpha': ratios[1],
                            'lag': lag_val,
                            'model': regr
                            })
            self.logger.info('completed a full hyperparamter search / training for a lag val of elastic net')

        forecasts = pd.DataFrame(forecasts)
        actuals = pd.concat([Y, dates], axis=1)
        actuals.columns = ['t_actual', 'sasdate']
        self.EN_forecasts = pd.merge(forecasts, actuals, left_on='forecast_t', right_on='sasdate', how='left')
        
        self.EN_forecasts['forecast_t'] = pd.to_datetime(self.EN_forecasts['forecast_t'])
        # error storage
        self.error_metrics_EN = self.EN_forecasts.groupby(['l1_ratio', 'alpha', 'lag']).apply(self._mse).reset_index()
        # save dictionary of models to disk for later use
        with open(Path(self.models_path, 'elastic_net_models').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(self.EN_models, handle)
        with open(Path(self.models_path, 'scaler_elastic_net').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(scaler_X, handle) # last one in memory; full val/train sample

    def plot_EN_metrics(self, maxlags):

        '''Plots the error metrics of elastic nets, 1 plot per lag that shows the l1/l2 MSE'''
        for jj in range(1, maxlags+1):

            fig = plt.figure()
            one_lag = self.error_metrics_EN[self.error_metrics_EN.lag == jj]
            X_one, Y_one = np.array(one_lag.l1_ratio), np.array(one_lag.alpha)
            Z_one = np.array(one_lag.mse)

            ax = fig.gca(projection='3d')

            # Plot the surface.
            surf1 = ax.plot_trisurf(X_one, Y_one, Z_one, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.8)

            # Add a color bar which maps values to colors.
            fig.colorbar(surf1, shrink=0.5, aspect=5)

            plt.xlabel('l1 ratio', fontsize=9)
            plt.ylabel('l2 ratio (alpha)', fontsize=9)
            plt.suptitle('Grid search for elastic net, lag order=' + str(jj))
            plt.title('MSE')

            fig.tight_layout()
            plt.legend()
            pth = Path(self.graphics_path, 'elasticnet_hyperparams_'+str(jj)).with_suffix('.png')
            fig.savefig(pth)
            plt.close()

        self.logger.info('plotted and saved png files of error metrics in /reports/figures for elastic net hyperparameters')
 
    def train_random_forests(self, maxlags):
    
        '''Trains random forests models using walk forward validation across a range of hyperparameters max depth, estimator count, and lag order. For each lag value, steps through time
        steps and trains depth/estimator variations, stores forecasts. Calculates MSE for each depth/estimators/lag order possibility.'''
        forecasts = {
            'max_depth':  list(),
            'estimator_count': list(),
            'lag': list(),
            'forecast': list(),
            'forecast_t': list()
        }
        RF_params = {
            'max_depth': np.linspace(4, 7, 3),#np.linspace(0.1,1,11),
            'estimator_count': np.linspace(1, 3, 3)
        }
        RF_models = list()
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        for lag_val in range(1, maxlags+1):

            Y = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -1]
            X = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, 0:-2]
            X = X[[c for c in X.columns if 'var94' in c or 'var8(t-1)' in c or 'var75(t-1)']]

            dates = self.reframed_dfs['lag_'+str(lag_val)].iloc[:, -2]
            
            # Get the number of initial training observations
            nobs = len(Y)
            n_init_training = int(nobs * VALIDATION_SAMPLE_PERCENT)
            
            for time_step in range(0, (nobs-n_init_training-1)):
                
                # Create model for training sample, gridsearch and fit parameters
                training_X, training_Y = X.iloc[:n_init_training+time_step], Y.iloc[:n_init_training+time_step]
                training_X_preprocessed, training_Y_preprocessed = pd.DataFrame(scaler_X.fit_transform(training_X.values)), pd.DataFrame(scaler_y.fit_transform(training_Y.values.reshape(-1, 1)))
                test_X = X.iloc[n_init_training+1+time_step, :]
                test_X_preprocessed = pd.DataFrame(scaler_X.transform(test_X.values.reshape(1, -1)))
                
                for ratios in itertools.product(RF_params['max_depth'], RF_params['estimator_count']):
                    regr = RandomForestRegressor(random_state=42, max_depth=int(ratios[0]), n_estimators=int(ratios[1]), bootstrap=True, warm_start=True)
                    regr.fit(training_X_preprocessed, training_Y_preprocessed.values[:, 0])

                    values = [
                        ratios[0],
                        ratios[1],
                        lag_val,
                        dates.iloc[n_init_training+1+time_step],
                        scaler_y.inverse_transform(regr.predict(test_X_preprocessed))[0]
                    ]
                    [forecasts[k].append(v) for k, v in zip(['max_depth', 'estimator_count', 'lag', 'forecast_t', 'forecast'], values)]

                    # save models
                    if time_step == (nobs-n_init_training-2):
                        RF_models.append({
                            'max_depth': ratios[0],
                            'estimator_count': ratios[1],
                            'lag': lag_val,
                            'model': regr
                            })
                print('trained one time step for hyperparameters grid, time step=', time_step, ' out of: ', (nobs-n_init_training-1))
            self.logger.info('completed a full hyperparamter search / training for a lag val of random forests')

        forecasts = pd.DataFrame(forecasts)
        actuals = pd.concat([Y, dates], axis=1)
        actuals.columns = ['t_actual', 'sasdate']
        self.RF_forecasts = pd.merge(forecasts, actuals, left_on='forecast_t', right_on='sasdate', how='left')
        
        self.RF_forecasts['forecast_t'] = pd.to_datetime(self.RF_forecasts['forecast_t'])
        # error storage
        self.error_metrics_RF = self.RF_forecasts.groupby(['max_depth', 'estimator_count', 'lag']).apply(self._mse).reset_index()
        # save dictionary of models to disk for later use
        print(self.error_metrics_RF)
        with open(Path(self.models_path, 'random_forests_models').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(self.RF_models, handle)
        with open(Path(self.models_path, 'scaler_random_forests').with_suffix('.pkl'), 'wb') as handle:
            pickle.dump(scaler_X, handle) # last one in memory; full val/train sample

    def plot_RF_metrics(self, maxlags):

        '''Plots the error metrics of elastic nets, 1 plot per lag that shows the l1/l2 MSE'''
        for jj in range(1, maxlags+1):

            fig = plt.figure()
            one_lag = self.error_metrics_EN[self.error_metrics_EN.lag == jj]
            X_one, Y_one = np.array(one_lag.l1_ratio), np.array(one_lag.alpha)
            Z_one = np.array(one_lag.mse)

            ax = fig.gca(projection='3d')

            # Plot the surface.
            surf1 = ax.plot_trisurf(X_one, Y_one, Z_one, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False, alpha=0.8)

            # Add a color bar which maps values to colors.
            fig.colorbar(surf1, shrink=0.5, aspect=5)

            plt.xlabel('l1 ratio', fontsize=9)
            plt.ylabel('l2 ratio (alpha)', fontsize=9)
            plt.suptitle('Grid search for elastic net, lag order=' + str(jj))
            plt.title('MSE')

            fig.tight_layout()
            plt.legend()
            pth = Path(self.graphics_path, 'elasticnet_hyperparams_'+str(jj)).with_suffix('.png')
            fig.savefig(pth)
            plt.close()

        self.logger.info('plotted and saved png files of error metrics in /reports/figures for elastic net hyperparameters')

    def plot_ML_forecasts(self):

        '''Plots and reports forecats (t+1) for all ML models'''
    
        best_en = self.error_metrics_EN.iloc[self.error_metrics_EN['mse'].idxmin()].reset_index().T
        best_en.columns = best_en.iloc[0]
        print('Found best EN hypers at: ', best_en)

        best_en_forecasts = self.EN_forecasts[
                                (self.EN_forecasts.l1_ratio == best_en.l1_ratio.iloc[1]) &
                                (self.EN_forecasts.alpha == best_en.alpha.iloc[1]) &
                                (self.EN_forecasts.lag == best_en.lag.iloc[1])
                            ]

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        best_en_forecasts['sasdate'] = pd.to_datetime(best_en_forecasts.sasdate)
        
        ax1.plot(best_en_forecasts['sasdate'], best_en_forecasts['t_actual'], color='dodgerblue', label='actual')
        ax1.plot(best_en_forecasts['sasdate'], best_en_forecasts['forecast'], color='navy', label='elastic net forecast', linestyle=":")
        #ax1.plot(df_Markov['sasdate'], df_Markov['t_forecast'], color='crimson', label=('Markov switching model, lag='+str(chosen_lag_Markov)+' forecast'), linestyle=":")

        # Decorations
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('BAAFFM Spread', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)
        plt.title('Forecast vs. Realized Values: BAAFFM', weight='bold')

        fig.tight_layout()
        plt.legend()
        pth = Path(self.graphics_path, 'yhat_y_ML').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of forecasts of ML models vs. actuals')

    def execute_analysis(self):

        '''Executes training, scaling, tuning, and error analysis for ml models, saves final models to disk.
        All model optimizition is done programmatically with hyperparameter grid search (unlike manual selection of lag order in Classical models)'''
        self.get_data()
        self.splice_test_data()
        self.reframe_train_val_df(maxlags=20)
        #self.explore_dim_reduce(maxlags=1)
        #self.train_elastic_net(maxlags=9)
        #self.plot_EN_metrics(maxlags=9)
        self.train_random_forests(maxlags=2)
        #self.plot_ML_forecasts()

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
