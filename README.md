
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
- MATHEMATICAL APPROACH
- RESULTS

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

- Recommend running train_model as a background process that can be returned to later if running locally. Given the walk-forward validation method, significant training time is incurred
- Estimated runtime will vary by computer, but on an Intel(R) Core(TM) i5-6200U CPU @2.30GHz with 8.00 GB memory, searching up to lag_order=12 for the models takes 1-2 hours

```shell
$ python /src/data/build_features.py > /logs/features_log.txt
$ nohup python /src/models/train_model.py > /logs/models_log.txt &
```

---
## Mathematical Approach

The extension follows the approach taken in statsmodels, i.e. the model is an autoregression where the coefficients, the mean of the process (possibly including trend or regression effects) and the variance of the error term may be switching across regimes
- A detailed discussion of Markov ARs can be found in Hamilton's original paper
- Additional detail on MS-AR forecasting <a target="_blank" href="https://warwick.ac.uk/fac/soc/economics/research/workingpapers/1995-1998/twerp489.pdf/">here</a>, which also follows Hamilton


**Model Forecast Specification**
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/math.PNG?raw=true)

The parameters needed to calculate this are available in memory in statsmodel in the fitted MS-AR model. The extension takes in the fitted model, and accesses the learned parameters as follows:

1. For each regime, calculate the t+1 forecast using the AR coefficients and the lags of the data available to the user at time t (e.g., t, t-1... etc.)
2. Fill out the learned Markov transition matrix so that the probabilities for any regime k sum to 1.0 (this is simply bookeeping to ease the later calculations)
3. Pull up the filtered probabilities of which regime the user (time=t) is currently in when making the forecast
4. Using the transition matrix and the filtered probabilities, calculate the probabilities of which regime the user will be in in time t+1 (one step in the Markov chain)
5. Weight the forecasts calculated in (1) by the new probabilities, and sum
---

## Results

**Preliminary Data Transforms**

*The OOS MS-AR t+1 forecasts for U.S. unemployment claims perform comparably with an AR and exponential smoothing in normal periods, and outperforms in a recession.*

We begin by transforming the data as suggested by the authors of the dataset to make it stationary https://research.stlouisfed.org/econ/mccracken/fred-databases/. The full dataset contains ~140 different series with corresponding strategies for transformation (differencing, logs, logs+differences, etc.). The data can be rougly categorized into different economic factors. For example, the housing variables contain many housing series:

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/time_series_transformed_housing.png?raw=true)

The below graph isolates the chosen variable of interest, unemployment claims. The series is post-making stationary. In the walk-forward validation, the initial training sample is 1970 through late 1990's, while 1998-2007 is considered "normal" economic times, and late 2007-2010 is considered "shock/recession."

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/y_var_time_series.png?raw=true)

A visual examination of the lag structure using an ACF plot:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/acf_plot.png?raw=true)
 

### **Cross-Model Comparison in Various Regimes / Time Periods**

- Using the optimal hyperparameters (lags) selected above, the below graph shows the MSE of the OOS forecasts
- The MS-AR extension performs comparably with the other univariate statsmodels OOS methods
- It outperforms in the non-dominant regime, consistent with literature

![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/error_summary.png?raw=true)

The t+1 forecasts plotted with the actuals show the improved ability to model the more extreme values in '08:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/yhat_y_Classical.png?raw=true)


**Detail on Lag Selection/Parameter Tuning**

The baseline models used are an autoregression and exponential smoothing. The mean squared error for the t+1 forecasts using walk-forward validation across various lag orders are:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/AR_errors.png?raw=true)

Given the above, a 3rd order lag for the AR is used. Holt's Exponential Smoothing baseline model, on the other hand, uses a weighted lag combination. No hyperparameters are needed to be optimized in the statsmodels implementation.

The MS-AR's mse for the same period (using the custom extension to allow for walk-forward validation) is:
![Alt Text](https://github.com/RachelDoehr/forecasting/blob/master/reports/figures/MKV_errors.png?raw=true)

A 3rd order Markov model is used as well.

---


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> 