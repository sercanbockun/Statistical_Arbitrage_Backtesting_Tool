import itertools
from statsmodels.tsa.stattools import coint, adfuller
import numpy as np
import pandas as pd
import tqdm
import logging
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from statsmodels.regression.rolling import RollingOLS

def Sharpe_Ratio(series, timeframe, interest_rate = 0.05): # Generally input return on Trades or the Wallet (It genuinely should be the continuously updated# version of the Wallet)
    
    # if market_serie is np.nan:
    if timeframe == "_1M":
        year_div_timeframe = 365*24*60
    if timeframe == "_5M":
        year_div_timeframe = 365*24*12
    if timeframe == "_15M":
        year_div_timeframe = 365*24*4
    if timeframe == "_1H":
        year_div_timeframe = 365*24
    if timeframe == "_4H":
        year_div_timeframe = 365*6
    if timeframe == "_1D":
        year_div_timeframe = 365
        
    #risk_free_rate = interest_rate / year_div_timeframe
    risk_free_rate = (1 + interest_rate) ** (1 / year_div_timeframe) - 1
    series = series.dropna()
      
    
    returns = series.pct_change(1)
    returns = returns.dropna()
    returns_mean = returns.mean()
    returns_std = returns.std()
    sharpe_ratio = (returns_mean - risk_free_rate )/returns_std
    #excess_returns = returns - risk_free_rate
    #sharpe_ratio = np.sqrt(year_div_timeframe) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def find_cointegrated_pairs(price_matrix, type_,trend_ , significant_level = 0.05 ): #type_ = "raw", "normalized", "log"
   #https://hudsonthames.org/an-introduction-to-cointegration/
   # One of the most time consuming functions: Determining the pairs is a huge task to do. 
   # 1- Implement other pair finding algoritghms. There are several academic work on this. 
   # 2- Client should also be able to pass in their selected pairs.


    cointegrated_pairs = []
    symbols = price_matrix.columns.values.tolist()
    symbol_pairs = list(itertools.combinations(symbols, 2))
    printed_download_percentiles = [0]
    i = 0
    for pair in symbol_pairs:
        i+= 1
        if (i / len(symbol_pairs) - printed_download_percentiles[-1]) > 0.1:  
            print(i / len(symbol_pairs))
            printed_download_percentiles.append(i / len(symbol_pairs))
        symbol_1, symbol_2 = pair
        pair_df = price_matrix[[symbol_1, symbol_2]]
        pair_df = pair_df[pair_df[symbol_1].notna()]
        pair_df = pair_df[pair_df[symbol_2].notna()]
        # print(pair_df)
        if len(pair_df) >= len(price_matrix)*0.5:
            
            if type_ == "raw":
                cointegration_test = coint(pair_df[symbol_1].values.astype(float), pair_df[symbol_2].values.astype(float), trend = trend_, maxlag = 1)
            

            elif type_ == "normalized":
                x1, x2 = pair_df[symbol_1].values.astype(float), pair_df[symbol_2].values.astype(float)
                p1 = (x1- np.minimum.accumulate(x1))/(np.maximum.accumulate(x1) - np.minimum.accumulate(x1))
                p2 = (x2- np.minimum.accumulate(x2))/(np.maximum.accumulate(x2) - np.minimum.accumulate(x2))
                cointegration_test = coint(p1, p2, trend = trend_, maxlag = 1)

            elif type_ == "log":
                cointegration_test = coint(np.log(pair_df[symbol_1].values.astype(float)), np.log(pair_df[symbol_2].values.astype(float)), trend =trend_, maxlag = 1)
                    
            p_value = 0
            p_value =  cointegration_test[1]
            if p_value < significant_level :
                cointegrated_pairs.append([symbol_1, symbol_2, p_value])
        else:
            cointegrated_pairs.append([symbol_1, symbol_2, 9999999])
            
            

    result = sorted(cointegrated_pairs, key=lambda cointegrated_pairs: cointegrated_pairs[2])
    
    cointegrated_pairs_df = pd.DataFrame(result, columns = ['Asset 1', 'Asset 2', 'p-value'])
    cointegrated_pairs_df = cointegrated_pairs_df[~cointegrated_pairs_df["Asset 1"].str.contains("USDUSD")]
    cointegrated_pairs_df = cointegrated_pairs_df[~cointegrated_pairs_df["Asset 1"].str.contains("USDCUSD")]
    cointegrated_pairs_df = cointegrated_pairs_df[~cointegrated_pairs_df["Asset 2"].str.contains("USDUSD")]
    cointegrated_pairs_df = cointegrated_pairs_df[~cointegrated_pairs_df["Asset 2"].str.contains("USDCUSD")]
    cointegrated_pairs_df = cointegrated_pairs_df.groupby('Asset 1').head(5).reset_index(drop=True)
    cointegrated_pairs_df = cointegrated_pairs_df.groupby('Asset 2').head(4).reset_index(drop=True)
    cointegrated_pairs_df = cointegrated_pairs_df.groupby('Asset 1').head(2).reset_index(drop=True)
    cointegrated_pairs_df = cointegrated_pairs_df.groupby('Asset 2').head(2).reset_index(drop=True)
    
    selected_pairs = cointegrated_pairs_df.iloc[:50, :]

    return selected_pairs 


def calculate_portfolio_metrics(wallet_values, frequency, risk_free_rate=0.04):
    """
    Calculate comprehensive portfolio performance metrics from a series of wallet values
    at different time frequencies.
    
    Parameters:
    wallet_values: pd.Series or array-like - Historical wallet/portfolio values
    frequency: str - Time frequency of the data ('1min', '1H', '4H', '1D')
    risk_free_rate: float - Annual risk-free rate (default 4%)
    
    Returns:
    pd.DataFrame - DataFrame containing all calculated metrics
    """
    
    # Convert to pandas Series if not already
    if not isinstance(wallet_values, pd.Series):
        wallet_values = pd.Series(wallet_values)
    
    # Define annualization factors for different frequencies
    annualization_factors = {
        '_1M': 525600,  # 365 * 24 * 60
        '_5M': 105120,
        '_15M': 35040,
        '_1H': 8760,      # 365 * 24
        '_4H': 2190,      # 365 * 24 / 4
        '_1D': 365
    }
    
    # Get the appropriate annualization factor
    if frequency not in annualization_factors:
        raise ValueError(f"Unsupported frequency. Must be one of {list(annualization_factors.keys())}")
    
    ann_factor = annualization_factors[frequency]
    
    # Calculate returns
    returns = wallet_values.pct_change().dropna()
    
    # 1. Basic Return Metrics
    total_return = (wallet_values.iloc[-1] / wallet_values.iloc[0] - 1) * 100
    
    # Annualized return (geometric mean)
    n_periods = len(wallet_values)
    annualized_return = (((1 + total_return/100) ** (ann_factor/n_periods)) - 1) * 100
    
    # 2. Risk Metrics
    volatility = returns.std() * np.sqrt(ann_factor) * 100
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # 3. Risk-Adjusted Return Metrics
    # Adjust risk-free rate to match the frequency
    period_risk_free = (1 + risk_free_rate) ** (1/ann_factor) - 1
    excess_returns = returns - period_risk_free
    
    # Sharpe Ratio with frequency adjustment
    sharpe_ratio = np.sqrt(ann_factor) * (excess_returns.mean() / returns.std())
    
    # Sortino Ratio (only considering negative returns)
    negative_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(ann_factor) * (excess_returns.mean() / negative_returns.std())
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown)
    
    # Create metrics dictionary
    metrics_dict = {
        # Return Metrics
        "total_return": total_return,
        "annualized_return": annualized_return,
        f"average_{frequency}_return": returns.mean() * 100,
        
        # Risk Metrics
        "annualized_volatility": volatility,
        "max_drawdown": max_drawdown,
        
        # Risk-Adjusted Returns
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        
        # Additional Info
        "number_of_periods": n_periods,
        "frequency": frequency,
        "annualization_factor": ann_factor
    }

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df = metrics_df.transpose()
    return metrics_df

def get_timeframe_of_the_data(price_matrix):
    day = 86400000
    hour = day/24
    minute = hour/60
    time_frame = (price_matrix.index[-2] - price_matrix.index[-3])

    if float(time_frame) == float(minute):
        return "_1M"
    
    elif float(time_frame) == float(5*minute) :
        return "_5M"
    elif float(time_frame) == float(15*minute) :
        return "_15M"
    elif float(time_frame)  == float(60*minute) :
        return "_1H"
    elif float(time_frame)  == float(60*4*minute) :
        return "_4H"
    elif float(time_frame)  == float(60*24*minute) :
        return "_1D"
    
    else: return np.nan



# Import the necessary libraries if needed but not defined above
import statsmodels.api as sm

# Example usage
#if __name__ == "__main__":