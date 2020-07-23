
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/example_markov_chain.gif?raw=true)

# Univariate Classical Models Out-of-Sample Forecasting


 *A custom extension of the statsmodels Markov Autoregression package for OOS forecasting*

**COMPARISON OF PERFORMANCE IN PREDICTING U.S. UNEMPLOYMENT CLAIMS IN 'NORMAL' AND 'SHOCK' TIME PERIODS**

> -> Builds on the Markov AR available in the latest statsmodels package available <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression" target="_blank">here</a>

> -> Model comparison to basic Autoregression and Holt's Exponential Smoothing

> -> Mathematical approach continues the strategy used in statsmodels, e.g. Kim, Chang-Jin, and Charles R. Nelson. “State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications”. MIT Press Books. The MIT Press.

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/markov_ar_specification.PNG?raw=true)

**Motivation & Project Summary**

Markov Autoregressions are well-regarded with respect to their abilities to identify regime shifts in time series data. Given their semi-parametric framework, MS-AR models are often able to provider superior in-sample forecasts (Clements and Krolzig, 1998, & Dacco and Satchell, 1999). Researchers have also found the relative out-of-sample forecast performance of regime-switching models depends on the regime present at the time the forecast is made. 

This project started out as a comparison of statsmodels' univariate time series methods using a walk-forward validation strategy. However, while out-of-sample ('OOS') prediction is available for Autoregressions and Exponential Smoothers, <a href="https://github.com/statsmodels/statsmodels/blob/ebe5e76c6c8055dddb247f7eff174c959acc61d2/statsmodels/tsa/regime_switching/markov_switching.py#L702-L703" target="_blank">it is not yet implemented for MS-AR</a>

Here I leverage the statsmodels MS-AR framework to build an extension which generates OOS forecasts for MS-ARs. Currently the extension is limited to the context in which I needed it (2 regimes, no exogenous vars), but it could be easily extended to complement the full range of **kwargs MS-AR offers. 

The extension is demonstrated on Federal Reserve monthly economic data, showing forecast performance for U.S. unemployment claims in "normal" times (~2000 - 2007) and a crisis/recession (~2008 - 2010). **Consistent with the literature, the MS-AR offers comparable out of sample forecasting for normal times and outperforms in the non-dominant regime.**

> ***ReadMe Table of Contents***

- INSTALLATION & SETUP
- RESULTS
- MATHEMATICAL APPROACH

---

## Installation & Setup

### Clone

- Clone this repo to your local machine using `https://github.com/RachelDoehr/forecasting.git`

### Setup

- Install the required packages

> Requires python3. Suggest the use of an Anaconda environment for easy package management.

```shell
$ pip install -r requirements.txt
```

### Example Use on Federal Reserve Data

- Recommend running train_model as background processes that can be returned to later if running locally. Given the walk-forward validation method, significant training time is incurred
- Estimated runtime will vary by computer, but on an Intel(R) Core(TM) i5-6200U CPU @2.30GHz with 8.00 GB memory, searching up to lag_order=12 takes 1-2 hours

```shell
$ python /src/data/build_features.py > /logs/features_log.txt
$ nohup python /src/models/train_model.py > /logs/models_log.txt &
```

---

## Results

As noted at the start, the OOS MS-AR t+1 forecasts for U.S. unemployment claims perform comparably with an AR and exponential smoothing in normal periods, and outperforms in a recession. 

We begin by transforming the data as suggested by the authors of the dataset to make it stationary https://research.stlouisfed.org/econ/mccracken/fred-databases/. The full dataset contains ~140 different series with corresponding strategies for transformation (differencing, logs, logs+differences, etc.). The data can be rougly categorized into different economic factors. For example, the housing variables contain many housing series:

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/time_series_transformed_housing.png?raw=true)


The grouped nature of the variables can also be seen in a visual examination of the correlation matrix (again post-stationarizing):

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/correlation_matrix_2.png?raw=true)


Isolating the chosen variable of interest, unemployment claims:

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/correlation_matrix_2.png?raw=true)

- All the `code` required to get started
- Images of what it should look like


---

## Mathematical Approach

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>