# -*- coding: utf-8 -*-
from pathlib import Path
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
    
    def __init__(self):
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()

    def get_data(self):

        '''Reads in csv from s3'''
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='current.csv')
        self.raw_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    def prelim_null_analysis(self):

        '''Removes older dates driving most of the nulls'''
        # remove 1960's due to significant missing values
        self.interim_df = self.raw_df.iloc[1:].iloc[:-1] # last row + transforms row
        self.interim_df['sasdate'] = pd.to_datetime(self.interim_df['sasdate'], format="%m/%d/%Y")
        self.interim_df = self.interim_df[self.interim_df.sasdate.dt.year >= 1970]

        print('Total obs: ', self.interim_df.shape[0])
        null_counts =  pd.DataFrame(self.interim_df.isnull().sum(axis = 0))
        null_counts.columns = ['null_count']
        nulls = null_counts[null_counts.null_count > 0]
        print(nulls)

        # will likely simply remove the 4 variables w/nulls, first visualizing raw + suggested transformed/stationary
        # ensure numeric
        self.interim_df.iloc[:, 1:].apply(pd.to_numeric)

    def _plot_group_variables(self, indices, plot_title, png_title):

        '''sub function for using within time series visualization function'''
        df = self.transformed_df.iloc[:, [i-2 for i in indices]] # Appdx key is off by 3
        # graph
        x =self.transformed_df.sasdate
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)

        colormap = plt.get_cmap('coolwarm')
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
        plt.show()
        
        pth = Path(self.graphics_path, 'time_series_transformed_'+png_title).with_suffix('.png')
        fig.savefig(pth)

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

    def visualize_transformed_data(self):

        # housing variables
        housing_indx = list(range(50, 60)) # from Appdnx key
        self._plot_group_variables(housing_indx, 'Housing Variables: Transformed Time Series', 'housing')


    def visualize_store_types(self):
        grouped_df = self.raw_df.groupby(['Date', 'Type', 'Store']).sum().reset_index() # adding up the Dept sales for each store / date, keeping Type along for the ride
        grouped_df = grouped_df.groupby(['Date', 'Type']).mean().reset_index()[['Weekly_Sales', 'Type', 'Date']] # avg Total weekly sales across the stores by type / date
        
        # graph
        pv = pd.pivot_table(grouped_df, index='Date', columns='Type', values='Weekly_Sales').reset_index()
        x = pd.to_datetime(pv['Date']) 
        y1 = pv['A']
        y2 = pv['B']
        y3 = pv['C']

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        ax1.plot(x, y1, color='gray', label='Type: A')
        ax1.plot(x, y2, color='deepskyblue', label='Type: B', linestyle='--')
        ax1.plot(x, y3, color='crimson', linestyle=':', label='Type: C')

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Average Weekly Total Store Sales', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)

        fig.tight_layout()
        plt.legend()
        plt.title('Time Series Plot of Weekly Sales by Store Type', fontsize=12, fontweight='bold')

        pth = Path(self.graphics_path, 'time_series_store_type').with_suffix('.png')
        fig.savefig(pth)

    def visualize_departments(self):
        df = self.raw_df.groupby(['Dept', 'Type']).mean().reset_index()[['Weekly_Sales', 'Dept', 'Type']] # adding up the sales for each dept / date
        
        # Draw Plot
        f, ax = plt.subplots(1,1, figsize=(16,10), dpi= 80)

        sns.kdeplot(df.loc[df['Type'] == 'A', "Weekly_Sales"], shade=True, color="gray", label="Store Type=A", alpha=.7, ax=ax)
        sns.kdeplot(df.loc[df['Type'] == 'B', "Weekly_Sales"], shade=True, color="crimson", label="Store Type=B", alpha=.7, ax=ax)
        sns.kdeplot(df.loc[df['Type'] == 'C', "Weekly_Sales"], shade=True, color="dodgerblue", label="Store Type=C", alpha=.7, ax=ax)

        # Decoration
        plt.title('Distribution of Average Weekly Sales Across Departments (by Store Type)', fontsize=22)
        plt.legend()

        pth = Path(self.graphics_path, 'kernel_density_plot_depts').with_suffix('.png')
        f.savefig(pth)

    def visualize_markdowns(self):
        grouped_df = self.raw_df.groupby(['Date']).mean().reset_index()[['Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] # avg 5 markdown columns
        
        # graph
        x = pd.to_datetime(grouped_df['Date'])
        
        fig, ax1 = plt.subplots(1,1, figsize=(16,9), dpi= 80)
        ax1.set_prop_cycle(cycler('color', ['gray', 'crimson', 'dodgerblue', 'b', 'navy']) +
                   cycler('linestyle', ['-', ':', '-', ':', '-']))

        for col in grouped_df.columns[1:]:
            ax1.plot(x, grouped_df[col], label=col)
        
        # Decorations
        ax1.set_xlabel('Date', fontsize=20)
        ax1.tick_params(axis='x', rotation=0, labelsize=12)
        ax1.set_ylabel('Average Weekly MarkDown (excl. NULLs)', color='black', fontsize=20)
        ax1.tick_params(axis='y', rotation=0, labelcolor='black' )
        ax1.grid(alpha=.4)

        fig.tight_layout()
        plt.legend()
        plt.title('Time Series Plot of Weekly Avg. MarkDown Values', fontsize=12, fontweight='bold')
        
        pth = Path(self.graphics_path, 'time_series_markdowns').with_suffix('.png')
        fig.savefig(pth)

    def execute_dataprep(self):
        self.get_data()
        self.prelim_null_analysis()
        self.transform_stationary()
        self.visualize_transformed_data()
        #self.explore_correlations()

def main():
    """ Runs data processing scripts to turn raw data from s3 into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    EconData = DatasetMaker()
    EconData.execute_dataprep()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
