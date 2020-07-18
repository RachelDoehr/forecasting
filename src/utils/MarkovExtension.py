# -*- coding: utf-8 -*-
'''Custom class that extends the statsmodels Markov Autoregression to include out-of-sample forecasting.

Currently supports t+1 forecasts only and time invariant transition probabilities. Assumes no exogenous vars.
Usage:
    --> Pass in statsmodel fitted MS-AR to initialize
    --> Call oos forecasting method to generate t+1 forecasts
'''

import itertools
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

class MSARExtension():

    '''Custom extension for out-of-sample forecasts'''
    
    
    def __init__(self, fitted_statsmodel):
        
        self.model = fitted_statsmodel
        self.last_nobs = self.model.data.orig_endog.tail(fitted_statsmodel.order)

    def _gen_perRegime_forecasts(self):

        '''Pull up fitted transition matrix of markov chain process'''

        parameters = pd.DataFrame(self.model.params)
        parameters.reset_index(level=0, inplace=True)
        parameters.columns = ['param', 'estimate']
        self.perRegime_forecasts = []

        for regime in range(1, self.model.k_regimes+1):

            r_params = parameters[parameters['param'].str.contains('['+str(regime-1)+']', regex=False)]
            
            mat1 = r_params[~r_params.param.str.contains('sigma2')]
            
            mat2 = self.last_nobs.iloc[::-1]
            mat2 = pd.concat([pd.Series([1]), mat2]).tolist() # adding constant
            
            mat1['value'] = mat2
            mat1['product'] = mat1.value * mat1.estimate # dot product step 1

            self.perRegime_forecasts.append(mat1['product'].sum())
        
        # also pull up transition matrix
        self.transition_matrix = parameters[parameters['param'].str.contains('p[', regex=False)]
    
    def gen_forecasts(self):

        '''tbd'''

        self._gen_perRegime_forecasts()
        print(self.perRegime_forecasts)

        t_prob = self.model.filtered_marginal_probabilities.iloc[-1, :]
        print(t_prob)

        print(self.transition_matrix)
        tmp_mat = self.transition_matrix.iloc[0:int(self.transition_matrix.shape[0]/2), :]
        print(tmp_mat)

        


scaler = StandardScaler()
# sample usage
dta = pd.read_csv('data.csv').iloc[0:480, :]
b = scaler.fit_transform(dta.BAAFFM.values.reshape(-1, 1))
b = b.ravel()
dta_hamilton = pd.Series(b)
mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=3, order=3, switching_variance=True)
res_hamilton = mod_hamilton.fit()

OOSMSAR = MSARExtension(fitted_statsmodel=res_hamilton)
OOSMSAR.gen_forecasts()

