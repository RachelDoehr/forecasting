# -*- coding: utf-8 -*-
from pathlib import Path
import logging
from dotenv import find_dotenv, load_dotenv
import boto3
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler
import io
import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters

BUCKET = 'forecast1301' # s3 bucket name

class DatasetMaker():

    '''Loads up train and test dataset stored in s3. Data from Kaggle.
    Performs initial data aggregation and exploratory visualizations.

    Also feature creation and saves copy of features locally and to s3.'''
    
    def __init__(self):
        sns.set(style="white")
        register_matplotlib_converters()
        self.s3_client = boto3.client('s3')
        self.graphics_path = Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()

    def get_data(self):
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='data/WMT_train.csv')
        self.train_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        obj = self.s3_client.get_object(Bucket=BUCKET, Key='data/WMT_test.csv')
        self.test_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        self.train_df['source'] = 'train'
        self.test_df['source'] = 'test'
        self.raw_df = pd.concat([self.train_df, self.test_df], axis=0, sort=True)
    
    def explore_correlations(self):
        corr = self.raw_df.corr()

        f, ax = plt.subplots(figsize=(12, 11))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, cmap=sns.diverging_palette(10, 220, n=9, as_cmap=True), vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        pth = str(Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()) + '\\' + 'correlation_matrix.png'
        f.savefig(pth)

        f1, ax1 = plt.subplots(figsize=(10, 10))
        a = self.raw_df[['CPI', 'Temperature', 'Fuel_Price', 'Unemployment', 'Weekly_Sales']]
        pd.plotting.scatter_matrix(a, ax=ax1, marker='o',
                                 hist_kwds={'bins': 20}, s=5, alpha=.8)
        pth = str(Path(__file__).resolve().parents[2].joinpath('reports').resolve().joinpath('figures').resolve()) + '\\' + 'scatter_pairs_histograms.png'
        f1.savefig(pth)

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

    def 

    def execute_dataprep(self):
        self.get_data()
        #self.explore_correlations()
        #self.visualize_store_types()
        #self.visualize_departments()
        #self.visualize_markdowns()


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    WMTData = DatasetMaker()
    WMTData.execute_dataprep()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
