# -*- coding: utf-8 -*-
'''
Trains the example run of the OOS Markov-Switching Autoregression forecasting on Federal Reserve data.

Note that the t/t+1 prediction timing conventions are *not* the exact same between statsmodels' Autoregression,
Markov Autoregression, and Exponential Smoother, so walk-forward validation is handled in separate methods for the 3.
'''

import io
import logging
import pickle
from pathlib import Path
import boto3
import matplotlib.lines as mlines
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# custom built class for OOS t+1 prediction
from MarkovExtension import MSARExtension

BUCKET = 'macro-forecasting1301' # s3 bucket name
DATA_SAMPLE_PERCENT = 0.825 # % of sample to use
VALIDATION_SAMPLE_PERCENT = 0.7 # 1-% to use for validation in walk-fwd
VAR = 'CLAIMSx' # unemployment claims
RECESSION_START = pd.Timestamp(2007, 1, 10) # NBER defined onset of recession period

class ClassicalModels():

    '''Loads up series, standardizes.
    Fits AR model, then a 2 regime Markov Switching model, then Exp Smoother,
    saves validation set forecasts, models, and error metrics'''

    def __init__(self, logger):

        self.logger = logger
        sns.set(style="white")
        register_matplotlib_converters()

        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()
        self.data_path = Path(__file__).resolve().parents[2].joinpath('data').resolve().joinpath('processed').resolve()
        self.models_path = Path(__file__).resolve().parents[2].joinpath('models').resolve()

        self.error_metrics_exp = {}
        self.forecasts_exp = {}
        self.EXP_models = {}
        self.error_metrics_AR = {}
        self.forecasts_AR = {}
        self.AR_models = {}
        self.error_metrics_Markov = {}
        self.forecasts_Markov = {}
        self.MKV_models = {}

        self.scaler = StandardScaler()

    def get_data(self):

        '''Reads in csv from s3'''

        obj = self.s3_client.get_object(Bucket=BUCKET, Key='features.csv')
        self.features_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data...')

    def save_model(self, pkl_name, item):

        '''Helper function for saving the latest model after train/val'''

        pth = Path(self.models_path, pkl_name).with_suffix('.pkl')
        with open(pth, 'wb') as handle:
            pickle.dump(item, handle)

    def filter_data(self):

        '''Removes the post-08 data up front.'''

        nobs = len(self.features_df)
        n_init_training_val = int(nobs * DATA_SAMPLE_PERCENT)
        self.train_val_df = self.features_df.iloc[0:n_init_training_val, :]
        self.endog = self.train_val_df[VAR]

        # Get the number of initial training observations
        self.nobs = len(self.endog)
        self.n_init_training = int(self.nobs * VALIDATION_SAMPLE_PERCENT)

    def examine_autocorr_stationary(self):

        '''The stationarity procedures per the authors of this dataset should have
        made stationary. Visual examination of autocorrelation.'''

        fig = plot_acf(self.endog, lags=36)
        pth = Path(self.graphics_path, 'acf_plot').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info(
            'plotted and saved png file in /reports/figures of autocorrelation plot of variable'
            )

        result = adfuller(self.endog)
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

    def train_AR(self):

        '''Trains state space AR process already integrated as baseline.
        Optimizes lag number by calculating MSE on validation during walk-forward training.'''

        def __train_one_lag(ll):

            forecasts = {}

            # Create model for initial training sample, fit parameters
            training_endog = self.endog.iloc[:self.n_init_training]
            training_endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(training_endog.values.reshape(-1, 1)))
            mod = sm.tsa.SARIMAX(training_endog_preprocessed, order=(ll, 0, 0), trend='c') # 1 lag, already stationary
            res = mod.fit(disp=0)

            # Save initial forecast
            forecasts[self.train_val_df.iloc[self.n_init_training, 1]] = self.scaler.inverse_transform(
                res.predict(
                    start=len(training_endog_preprocessed),
                    end=len(training_endog_preprocessed)
                    )
                )[0]

            # Step through the rest of the sample
            for t in range(self.n_init_training, self.nobs-1):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(self.endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
    
                mod = sm.tsa.SARIMAX(endog_preprocessed, order=(ll, 0, 0), trend='c')
                res = mod.fit(disp=0) # re-fit

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t+1, 1]] = self.scaler.inverse_transform(res.predict(start=len(endog_preprocessed), end=len(endog_preprocessed)))[0]
                # save the model at end of time series
                if t == self.nobs-2:
                    self.AR_models['lag_'+str(ll)] = res
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = self.train_val_df.tail(forecasts.shape[0])[['sasdate', VAR]]
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

        self.save_model('AR_models', self.AR_models)

    def train_MarkovSwitch_AR(self):
    
        '''Trains Markov Switching autoregression on univariate series.
        Allows for time varying covariance. Uses walk forward validation to tune lag order similar to AR.'''     

        def __train_one_lag(ll):

            forecasts = {}

            # Create model for initial training sample, fit parameters
            training_endog = self.endog.iloc[:self.n_init_training]
            training_endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(training_endog.values.reshape(-1, 1)))
            mod = sm.tsa.MarkovAutoregression(training_endog_preprocessed,
                                              k_regimes=2,
                                              order=ll,
                                              switching_variance=False,
                                              )
            
            res = mod.fit()
            res_extended = MSARExtension(res) # pass the trained model to OOS forecaster
            yhat = res_extended.predict_out_of_sample()

            # Save initial forecast
            forecasts[self.train_val_df.iloc[self.n_init_training, 1]] = self.scaler.inverse_transform(yhat.ravel())[0]
            # Step through the rest of the sample
            for t in range(self.n_init_training, self.nobs-1):
                # Update the results by appending the next observation
                endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(self.endog.iloc[0:t+1].values.reshape(-1, 1))) # re fit
    
                mod = sm.tsa.MarkovAutoregression(endog_preprocessed,
                                                  k_regimes=2,
                                                  order=ll,
                                                  switching_variance=False
                                                  )
                res = mod.fit()
                res_extended = MSARExtension(res)
                yhat = res_extended.predict_out_of_sample()

                # Save the new set of forecasts, inverse the scaler
                forecasts[self.train_val_df.iloc[t+1, 1]] = self.scaler.inverse_transform(yhat.ravel())[0]
                # save the model at end of time series
                if t == self.nobs-2:
                    self.MKV_models['lag_'+str(ll)] = res
        
            # Combine all forecasts into a dataframe
            forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
            actuals = self.train_val_df.tail(forecasts.shape[0])[['sasdate', VAR]]
            actuals.columns = ['sasdate', 't_actual']
            self.Markov_fcasts = pd.merge(forecasts, actuals, on='sasdate', how='inner').dropna()
            self.Markov_fcasts['sasdate'] = pd.to_datetime(self.Markov_fcasts['sasdate'])
            
            # error storage
            self.error_metrics_Markov[ll] = mean_squared_error(self.Markov_fcasts['t_actual'], self.Markov_fcasts['t_forecast'])
            # forecast storage
            self.forecasts_Markov['lag_'+str(ll)] = {
                'df': self.Markov_fcasts
            }
            self.logger.info('completed training for Markov Switching model with lag: '+str(ll))
        
        [__train_one_lag(lag_value) for lag_value in range(1, 13)]

        self.save_model('MKV_models', self.MKV_models)

    def train_exponential_smoother(self):
        
        '''Trains Holt's Exponential Smoothing model. Allows for dampened trend, seasonality.'''

        forecasts = {}

        # Create model for initial training sample, fit parameters
        training_endog = self.endog.iloc[:self.n_init_training]
        training_endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(training_endog.values.reshape(-1, 1)))
        mod = ExponentialSmoothing(training_endog_preprocessed,
                                   trend='add',
                                   seasonal='add',
                                   seasonal_periods=12
                                   )
        res = mod.fit()

        # Save initial forecast
        forecasts[self.train_val_df.iloc[self.n_init_training+1, 1]] = self.scaler.inverse_transform(
            res.predict()
        )[len(res.predict())-1]
        # Step through the rest of the sample
        for t in range(self.n_init_training, self.nobs-1):
        
            # Update the results by appending the next observation
            endog_preprocessed = pd.DataFrame(self.scaler.fit_transform(self.endog.iloc[0:t+1].values.reshape(-1, 1)))
            dates = pd.DataFrame(self.train_val_df.iloc[0:t+1, 1].values.reshape(-1, 1))
            mod = ExponentialSmoothing(endog_preprocessed,
                                       trend='add',
                                       seasonal='add',
                                       seasonal_periods=12
                                    )
            res = mod.fit()

            # Save the new set of forecasts, inverse the scaler
            forecasts[self.train_val_df.iloc[t+1, 1]] = self.scaler.inverse_transform(res.predict())[len(res.predict())-1]
            # save the model at end of time series
            if t == self.nobs-1:
                self.EXP_models['exp_weigh_lag_struct'] = res
        
        # Combine all forecasts into a dataframe
        forecasts = pd.DataFrame(forecasts.items(), columns=['sasdate', 't_forecast'])
        actuals = pd.concat([self.endog.tail(forecasts.shape[0]), dates.tail(forecasts.shape[0])], axis=1)
        actuals.columns = ['t_actual', 'sasdate']
        self.Expsmooth_fcasts = pd.merge(forecasts, actuals, on='sasdate', how='inner').dropna()
        self.Expsmooth_fcasts['sasdate'] = pd.to_datetime(self.Expsmooth_fcasts['sasdate'])
        
        # error storage
        self.error_metrics_exp['exp_weigh_lag_struct'] = mean_squared_error(self.Expsmooth_fcasts['t_actual'], self.Expsmooth_fcasts['t_forecast'])
        # forecast storage
        self.forecasts_exp['exp_weigh_lag_struct'] = {
            'df': self.Expsmooth_fcasts
        }
        self.logger.info('completed training for Exponential Smoothing model')

        self.save_model('EXP_models', self.EXP_models)

    def plot_errors_lags(self, which_errors, plt_title, png_title):

        '''For validation / lag tuning, plots the errors of the different lag terms of AR once trained'''
        df_error = pd.DataFrame(which_errors.items())
        
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
        plt.scatter(df_error.iloc[:, 0], df_error.iloc[:, 1], color='blue', s=30)
        # Decorations
        ax1.set_xlabel('Lag', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Validation Set RMSE', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black')
        fig.tight_layout()
        plt.legend()
        plt.title('Error Metrics: '+plt_title, fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, png_title).with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of errors at various lags')

    def plot_forecasts_Classical(self, chosen_lag_AR, chosen_lag_Markov):
        
        '''Plots and reports forecats (t+1) for both classical models'''
        df_Markov = self.forecasts_Markov['lag_'+str(chosen_lag_Markov)]['df']
        df_AR = self.forecasts_AR['lag_'+str(chosen_lag_AR)]['df']
        df_ExpSmo = self.forecasts_exp['exp_weigh_lag_struct']['df']

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
        
        ax1.plot(df_AR['sasdate'], df_AR['t_actual'], color='dodgerblue', label='actual')
        ax1.plot(df_AR['sasdate'], df_AR['t_forecast'], color='navy', label=('state space AR, lag='+str(chosen_lag_AR)+' forecast'), linestyle=":")
        ax1.plot(df_Markov['sasdate'], df_Markov['t_forecast'], color='crimson', label=('Markov switching model, lag='+str(chosen_lag_Markov)+' forecast'), linestyle=":")
        ax1.plot(df_ExpSmo['sasdate'], df_ExpSmo['t_forecast'], color='gray', label=('exponential smoothing model forecast'), linestyle=":")

        # Decorations
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Unemployment Claims', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black')
        ax1.grid(alpha=.4)
        plt.title('Forecast vs. Realized Values', weight='bold')

        fig.tight_layout()
        plt.legend()
        pth = Path(self.graphics_path, 'yhat_y_Classical').with_suffix('.png')
        fig.savefig(pth)
        self.logger.info('plotted and saved png file in /reports/figures of forecasts of Classical models vs. actuals')

    def compare_error_forecasts(self, chosen_lag_AR, chosen_lag_Markov):

        '''Creates charts to compare the forecasting abilities of MS-AR model to others, on 
        an aggregate time period as well as during recessions vs. expansions.'''

        df_M = self.forecasts_Markov['lag_'+str(chosen_lag_Markov)]['df']
        df_AR = self.forecasts_AR['lag_'+str(chosen_lag_AR)]['df']
        df_E = self.forecasts_exp['exp_weigh_lag_struct']['df']

        errors = []

        def _calc_mse(df):

            '''Helper function, calculates MSE by dataframe'''
            errors.append({
                'recession': mean_squared_error(df[df.sasdate >= RECESSION_START]['t_actual'], df[df.sasdate >= RECESSION_START]['t_forecast']),
                'expansion': mean_squared_error(df[df.sasdate < RECESSION_START]['t_actual'], df[df.sasdate < RECESSION_START]['t_forecast']),
                'aggregate': mean_squared_error(df['t_actual'], df['t_forecast'])
            })
        
        [_calc_mse(d) for d in [df_AR, df_E, df_M]]
        e = pd.DataFrame(errors)
        e['model'] = ['Autoregression', 'Exponential Smoothing', 'Markov Autoregression']
        # draw line
        def newline(p1, p2, color='black'):
            ax = plt.gca()
            l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='blue', marker='o', markersize=6)
            ax.add_line(l)
            return l

        fig, ax = plt.subplots(1, 1,figsize=(14, 14), dpi=80)

        ax.vlines(x=1, ymin=0.00025, ymax=0.00055, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
        ax.vlines(x=3, ymin=0.00025, ymax=0.00055, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

        # Points
        ax.scatter(y=e['expansion'], x=np.repeat(1, e.shape[0]), s=10, color='black', alpha=0.7)
        ax.scatter(y=e['recession'], x=np.repeat(3, e.shape[0]), s=10, color='black', alpha=0.7)

        # Line Segmentsand Annotation
        for p1, p2, c in zip(e['expansion'], e['recession'], e['model']):
            newline([1, p1], [3, p2])
            ax.text(1-0.05, p1, c + ', ' + str(round(p1, 5)), horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
            ax.text(3+0.05, p2, c + ', ' + str(round(p2, 5)), horizontalalignment='left', verticalalignment='center', fontdict={'size':14})

        # 'Before' and 'After' Annotations
        ax.text(1-0.05, 0.00054, 'EXPANSION', horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
        ax.text(3+0.05, 0.00054, 'RECESSION', horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})

        # Decoration
        ax.set_title("Slopechart: Comparing t+1 Forecast MSE between Economic Expansions ('normalcy') vs Recession ('shock')", fontdict={'size':16})
        ax.set(xlim=(0, 4), ylim=(0.00025, 0.00055), ylabel='Mean Squared Error on t+1 out-of-sample forecasts')
        ax.set_xticks([1, 3])
        ax.set_xticklabels(["Expansion (pre-'07)", "Recession (late '07-'10)"])
        plt.yticks(np.arange(0.00025, 0.00055, 0.000025), fontsize=12)
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.0)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.0)

        pth = Path(self.graphics_path, 'error_summary').with_suffix('.png')
        fig.savefig(pth)

        self.logger.info('plotted and saved error metric comparison in /reports/figures of forecasts of Classical models vs. actuals')

    def execute_analysis(self):

        '''Runs the necessary methods'''

        self.get_data()
        self.filter_data()
        self.examine_autocorr_stationary()
        self.train_AR()
        self.plot_errors_lags(
            which_errors=self.error_metrics_AR,
            plt_title='AR',
            png_title='AR_errors')
        self.train_MarkovSwitch_AR()
        self.plot_errors_lags(
            which_errors=self.error_metrics_Markov,
            plt_title='MS-AR',
            png_title='MKV_errors')
        self.train_exponential_smoother()
        self.plot_forecasts_Classical(chosen_lag_AR=3, chosen_lag_Markov=3)
        self.compare_error_forecasts(chosen_lag_AR=3, chosen_lag_Markov=3)

def main():

    """ Runs training of models and hyperparameter tuning, saves graphs, logs progress."""

    logger = logging.getLogger(__name__)
    logger.info('running classical models...')
    UnivariateClassicalModels = ClassicalModels(logger)
    UnivariateClassicalModels.execute_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
