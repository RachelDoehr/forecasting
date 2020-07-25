# -*- coding: utf-8 -*-
from pathlib import Path
from io import StringIO
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler
import datetime
import io
from functools import reduce
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'macro-forecasting1301' # s3 bucket name

class DatasetMaker():

    '''Loads up train and test dataset stored in s3. Data from Federal Reserve.
    Performs initial data aggregation and exploratory visualizations.

    Also feature creation and saves copy of features locally and to s3.'''
    
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
        self.raw_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.logger.info('loaded data, now cleaning...')
    
    def prelim_null_analysis(self):

        '''Removes older dates driving most of the nulls'''
        self.logger.info('now norming / transforming into stationary variables...')
        # remove 1960's due to significant missing values
        self.interim_df = self.raw_df.iloc[1:].iloc[:-1] # last row + transforms row
        self.interim_df['sasdate'] = pd.to_datetime(self.interim_df['sasdate'], format="%m/%d/%Y")
        self.interim_df = self.interim_df[self.interim_df.sasdate.dt.year >= 1970]

        null_counts =  pd.DataFrame(self.interim_df.isnull().sum(axis = 0))
        null_counts.columns = ['null_count']
        nulls = null_counts[null_counts.null_count > 0]
        print(nulls)

        # ensure numeric
        self.interim_df.iloc[:, 1:].apply(pd.to_numeric)

    def transform_stationary(self):

        '''Uses suggested transformations to make time series stationary, see https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md/Appendix_Tables_Update.pdf'''
        dfs = []
        for transform_num in list(range(1, 8)):
            dfs.append(self.interim_df[self.raw_df.columns[self.raw_df.iloc[0, :] == transform_num].tolist()])
        dfs.append(self.interim_df.iloc[:, 0])

        # ---------- apply transforms
        df1_transform = dfs[0] # none 
        
        df2_transform = dfs[1].diff(axis=0, periods=1) # first difference

        df4_transform = (dfs[3].apply(np.log10)) # log

        df5_tmp = (dfs[4].apply(np.log10))
        df5_transform = df5_tmp.diff(axis=0, periods=1) # first difference of log(xt)

        df6_tmp = (dfs[5].apply(np.log10))
        df6_tmp2 = df6_tmp.diff(axis=0, periods=1)
        df6_transform = df6_tmp2.pow(2) # squared first difference of log(xt)

        df7_transform = dfs[6].pct_change() # percentage change

        frames= [df1_transform, df2_transform, df4_transform, df5_transform, df6_transform, df7_transform, dfs[7]] # no type 3 transforms in Appdx
        self.transformed_df = reduce(lambda  left, right: left.join(right, how='outer'), frames)
        self.transformed_df = self.transformed_df[[c for c in self.raw_df.columns]] # put back in original order for grouping plotting

    def _plot_group_variables(self, indices, plot_title, png_title):

        '''sub function for using within time series visualization function'''
        df = self.transformed_df.iloc[:, [i-2 for i in indices]] # Appdx key is off by 3
        # graph
        x =self.transformed_df.sasdate
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)

        colormap = plt.get_cmap('seismic')
        ax1.set_prop_cycle(cycler('color', [colormap(k) for k in np.linspace(0, 1, 10)]) +
                   cycler('linestyle', ['-', ':', '-', ':', '-', '-', ':', '-', ':', '-']))

        for col in df.columns:
            ax1.plot(x, df[col], label=col)
        
        # Decorations
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Transformed Series Values', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)

        fig.tight_layout()
        plt.legend()
        plt.title(plot_title, fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'time_series_transformed_'+png_title).with_suffix('.png')
        fig.savefig(pth)

    def visualize_transformed_data(self):

        '''Visualizations of select time series variables, by group/type'''
        # housing variables
        indx = list(range(50, 60)) # from Appdnx key
        self._plot_group_variables(indx, 'Housing Variables: Transformed Time Series', 'housing')

        indx = list(range(6, 21)) # from Appdnx key
        self._plot_group_variables(indx, 'Output Variables: Transformed Time Series', 'output')

        indx = [3, 4, 5, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69] # from Appdnx key
        self._plot_group_variables(indx, 'Consumption Variables: Transformed Time Series', 'consumption')

        indx = list(range(66, 69)) # from Appdnx key
        self._plot_group_variables(indx, 'Money Variables: Transformed Time Series', 'money')

        # note extreme outlier in nonborrowed reserves in '08 - considering excluding from final set
        #indx = list(range(66, 74)) # from Appdnx key
        #self._plot_group_variables(indx, 'Money Variables: Transformed Time Series', 'money')

        indx = list(range(80, 97)) # from Appdnx key
        self._plot_group_variables(indx, 'Interest Rate Variables: Transformed Time Series', 'interestrates')

        indx = list(range(97, 102)) # from Appdnx key
        self._plot_group_variables(indx, 'Exchange Rate Variables: Transformed Time Series', 'fxrates')

        indx = list(range(23, 44)) # from Appdnx key
        self._plot_group_variables(indx, 'Labor Variables: Transformed Time Series', 'labor')

        indx = list(range(107, 121)) # from Appdnx key
        self._plot_group_variables(indx, 'Price Variables: Transformed Time Series', 'prices')

    def explore_correlations(self):

        '''Visualize correlations of post-transformed data'''
        correl = self.transformed_df.iloc[1:,60:]
        corr = correl.corr()
        # generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(17, 17))

        # generate a custom diverging colormap
        cmap = sns.diverging_palette(10, 220, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
                    square=True, linewidths=.2, cbar_kws={"shrink": 0.5})
        pth = Path(self.graphics_path, 'correlation_matrix_1').with_suffix('.png')
        f.savefig(pth)
        self.logger.info('plotted and save figures in /reports/figures/')

    def plot_y_var(self):

        '''plotting dependent variable visually post and pre-transform'''
        nobs = np.round(0.825*self.transformed_df.shape[0], 0)
        df = self.transformed_df.copy().iloc[0:int(nobs), :]
        # graph
        x =df.sasdate
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)

        ax1.fill_between(x, df['CLAIMSx'], label='Unemployment Claims', alpha=0.4, color='dodgerblue')
        
        # Decorations
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Transformed Series Values', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)

        fig.tight_layout()
        plt.legend()
        plt.title('U.S. Monthly Unemployment Claims', fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'y_var_time_series').with_suffix('.png')
        fig.savefig(pth)

    def final_prep_and_save(self):
        '''Removes the columns with significant null values. Can also remove outlier series, nonborrowed reserves.
        Then saves copies.'''
        self.logger.info('removing nulls and outliers, saving...')
        to_remove = ['ACOGNO', 'S&P PE ratio', 'TWEXAFEGSMTHx', 'UMCSENTx'] # leaving reserves in for now

        self.features_df = self.transformed_df.drop(to_remove, axis=1).iloc[1:, :]

        pth = Path(self.data_path, 'features').with_suffix('.csv')
        self.features_df.to_csv(pth)
        # upload to s3
        csv_buffer = StringIO()
        self.features_df.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(BUCKET, 'features.csv').put(Body=csv_buffer.getvalue())

    def execute_dataprep(self):
        self.get_data()
        self.prelim_null_analysis()
        self.transform_stationary()
        self.visualize_transformed_data()
        self.explore_correlations()
        self.plot_y_var()
        self.final_prep_and_save()

def main():
    """ Runs data processing scripts to turn raw data from s3 into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    EconData = DatasetMaker(logger)
    EconData.execute_dataprep()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
