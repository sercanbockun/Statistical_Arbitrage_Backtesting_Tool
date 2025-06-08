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
import statsmodels.api as sm
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stat_arb.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StatisticalArbitragePairFinder:
    """
    A class to identify and analyze pairs for statistical arbitrage trading
    using various methods including cointegration testing.
    """
    
    def __init__(self, price_matrix, min_history_pct=0.5, trend = 'c', transform_type="log", max_pairs = 50, max_per_asset = 2):
        """
        Initialize the pair finder with price data.
        
        Args:
            price_matrix: DataFrame containing price time series for all assets
            min_history_pct: Minimum percentage of history required for pair analysis
        """
        self.price_matrix = price_matrix
        self.min_history_pct = min_history_pct
        self.trend = trend
        self.transform_type = transform_type
        self.max_pairs = max_pairs
        self.max_per_asset = max_per_asset
        self.symbols = price_matrix.columns.tolist()
        self.pairs = list(itertools.combinations(self.symbols, 2))
        logger.info(f"Initialized with {len(self.symbols)} symbols and {len(self.pairs)} potential pairs")
    
    def preprocess_prices(self, pair_df, symbol_1, symbol_2):
        """
        Preprocess price data for cointegration testing based on specified transformation.
        
        Args:
            pair_df: DataFrame containing price data for the pair
            symbol_1: First symbol in the pair
            symbol_2: Second symbol in the pair
            transform_type: Type of transformation ("raw", "normalized", "log")
            
        Returns:
            Tuple of transformed price series or None if insufficient data
        """
        transform_type = self.transform_type
        # Filter out NaN values
        pair_df = pair_df[pair_df[symbol_1].notna() & pair_df[symbol_2].notna()]
        
        # Check if we have enough data
        if len(pair_df) < len(self.price_matrix) * self.min_history_pct:
            return None
            
        x1 = pair_df[symbol_1].values.astype(float)
        x2 = pair_df[symbol_2].values.astype(float)
        
        # Transform prices based on specified method
        if transform_type == "raw":
            return x1, x2
        
        elif transform_type == "normalized":
            # Min-max normalization with accumulating extremes
            p1 = (x1 - np.minimum.accumulate(x1)) / (np.maximum.accumulate(x1) - np.minimum.accumulate(x1))
            p2 = (x2 - np.minimum.accumulate(x2)) / (np.maximum.accumulate(x2) - np.minimum.accumulate(x2))
            return p1, p2
        
        elif transform_type == "log":
            # Check for non-positive values before log transform
            if np.any(x1 <= 0) or np.any(x2 <= 0):
                logger.warning(f"Non-positive values found in {symbol_1} or {symbol_2}, skipping log transform")
                return None
            return np.log(x1), np.log(x2)
        
        else:
            logger.error(f"Unknown transform type: {transform_type}")
            return None
    
    def calculate_spread(self, pair_df, symbol_1, symbol_2):
        trend = self.trend

        if trend == 'c':
            x = pair_df[symbol_2].values
            y = pair_df[symbol_1].values
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()

            spread = pair_df[symbol_1] - model.params[1] * pair_df[symbol_2]
            spread_mean = spread.mean()
            spread_std = spread.std()
            z_score = (spread - spread_mean)/spread_std

        if trend == 'ct':
            x = pair_df[symbol_2].values                   
            time_trend = np.arange(len(x))                   
            x = np.column_stack((time_trend, x))                   
            y = pair_df[symbol_1].values                   
            x = sm.add_constant(x)        
            model = sm.OLS(y, x).fit()
            predicted_y = model.predict(x)
            spread = pair_df[symbol_1] - predicted_y                
            spread_mean = spread.mean()
            spread_std = spread.std()
            z_score = (spread - spread_mean)/spread_std
    
        if trend == 'ctt':
            x = pair_df[symbol_2].values
            time_trend = np.arange(len(x))
            time_trend_squared = time_trend ** 2
            x = np.column_stack((time_trend, time_trend_squared, x))
            y = pair_df[symbol_1].values
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            predicted_y = model.predict(x)
            spread = pair_df[symbol_1] - predicted_y   
            
            spread_mean = spread.mean()
            spread_std = spread.std()
            z_score = (spread - spread_mean)/spread_std

        return spread, spread_mean, spread_std, z_score


    def find_cointegrated_pairs(self, significance_level=0.05, max_lag=1):
        """
        Find cointegrated pairs using the Engle-Granger two-step method (serial processing).
        
        Args:
            transform_type: Price transformation method ("raw", "normalized", "log")
            trend: Trend component in cointegration test ("c", "ct", "ctt", etc.)
            significance_level: P-value threshold for statistical significance
            max_lag: Maximum lag to consider in the cointegration test
            
        Returns:
            DataFrame of cointegrated pairs sorted by p-value
        """
        trend = self.trend
        transform_type = self.transform_type
        cointegrated_pairs = []
        total_pairs = len(self.pairs)
        
        print(f"Starting cointegration tests for {total_pairs} pairs with {transform_type} transformation")
        
        # Process pairs serially with better tracking
        for i, pair in enumerate(self.pairs):
            symbol_1, symbol_2 = pair
            
            # Print progress every 100 pairs or at specific percentage milestones
            if i % 10000 == 0 or i == total_pairs - 1:
                print(f"Processing pair {i+1}/{total_pairs} ({(i+1)/total_pairs*100:.1f}%): {symbol_1}-{symbol_2}")
            
            # Get price data for this pair
            pair_df = self.price_matrix[[symbol_1, symbol_2]]
            
            # Preprocess the data
            transformed_prices = self.preprocess_prices(pair_df, symbol_1, symbol_2)
            
            if transformed_prices is None:
                # Not enough data or preprocessing failed
            #    print(f"Skipping {symbol_1}-{symbol_2}: Insufficient data or preprocessing failed")
                continue
                
            x1, x2 = transformed_prices
            
            try:
                # Perform cointegration test
                _, p_value, _ = coint(x1, x2, trend=trend, maxlag=max_lag)
                
                # Only add pairs that meet the significance threshold
                if p_value < significance_level:
                    #print(f"✓ Found cointegrated pair: {symbol_1}-{symbol_2} with p-value: {p_value:.6f}")
                    cointegrated_pairs.append([symbol_1, symbol_2, p_value])
                
            except Exception as e:
                print(f"Error testing cointegration for {symbol_1}-{symbol_2}: {str(e)}")
        
        # Create and sort dataframe of results
        if not cointegrated_pairs:
            print("No cointegrated pairs found!")
            return pd.DataFrame(columns=['Asset 1', 'Asset 2', 'p-value'])
        
        result = pd.DataFrame(cointegrated_pairs, columns=['Asset 1', 'Asset 2', 'p-value'])
        result = result.sort_values(by='p-value', ascending=True)  # Changed to ascending=True
        
        print(f"Found {len(result)} cointegrated pairs with p-value < {significance_level}")
        
        # Filter out specific symbols if needed
        result = self._filter_unwanted_pairs(result)
        print(f"After filtering: {len(result)} pairs remain")
        
        return result

    def _filter_unwanted_pairs(self, pairs_df):
        """
        Filter out unwanted pairs based on specific criteria.
        
        Args:
            pairs_df: DataFrame of pairs to filter
            
        Returns:
            Filtered DataFrame
        """
        # Filter out specific patterns (e.g., stablecoins paired with USD)
        filtered_df = pairs_df
        for pattern in ["USDUSD", "USDCUSD", "USDTUSDC", "USDTDAI", "TUSDT"]:
            filtered_df = filtered_df[~filtered_df["Asset 1"].str.contains(pattern, na=False)]
            filtered_df = filtered_df[~filtered_df["Asset 2"].str.contains(pattern, na=False)]
        
        return filtered_df
    
    def calculate_hedge_ratio(self, symbol_1, symbol_2, method="ols"):
        """
        Calculate the optimal hedge ratio between two assets.
        
        Args:
            symbol_1: First symbol (Y in the regression)
            symbol_2: Second symbol (X in the regression)
            method: Method to calculate hedge ratio ("ols" or "total_least_squares")
            
        Returns:
            Hedge ratio (float)
        """
        pair_df = self.price_matrix[[symbol_1, symbol_2]].dropna()
        
        if method == "ols":
            # Ordinary Least Squares regression
            x = pair_df[symbol_2].values
            y = pair_df[symbol_1].values
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            return model.params[1]  # beta coefficient
            
        elif method == "total_least_squares":
            # Total Least Squares (orthogonal regression)
            x = pair_df[symbol_2].values
            y = pair_df[symbol_1].values
            
            # Calculate the hedge ratio using total least squares
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_centered = x - x_mean
            y_centered = y - y_mean
            
            # SVD approach for TLS
            u, d, v = np.linalg.svd(np.column_stack((x_centered, y_centered)), full_matrices=False)
            hedge_ratio = -v[1, 0] / v[1, 1]
            
            return hedge_ratio
        
        else:
            logger.error(f"Unknown hedge ratio method: {method}")
            return None
    
    def calculate_half_life(self, spread):
        """
        Calculate the half-life of mean reversion for a spread series.
        
        Args:
            spread: Time series of the spread between two assets
            
        Returns:
            Half-life in periods (float)
        """
        spread = pd.Series(spread).dropna()
        
        # Calculate lagged spread and delta
        lagged_spread = spread.shift(1).dropna()
        delta = spread[1:] - lagged_spread
        delta = pd.Series(delta).dropna()
        
        # Regression of delta on lagged spread
        x = sm.add_constant(lagged_spread)
        model = sm.OLS(delta, x).fit()
        
        # Extract the coefficient and calculate half-life
        print(model.params)
        beta = model.params[0]
        
        if beta >= 0:
            # Not mean-reverting
            return float('inf')
            
        half_life = -np.log(2) / beta
        return half_life

    def calculate_normality_score(self, shapiro_p, jarque_bera_p, skewness, kurtosis):
        """
        Calculate a composite score for normality of the Z-score distribution.
        
        Args:
            shapiro_p: p-value from Shapiro-Wilk test
            jarque_bera_p: p-value from Jarque-Bera test
            skewness: Skewness of the distribution
            kurtosis: Excess kurtosis of the distribution
            
        Returns:
            Normality score between 0 and 1 (higher is better)
        """
        # For p-values, higher is better for normality
        p_value_score = (shapiro_p + jarque_bera_p) / 2
        
        # For skewness, closer to 0 is better
        skewness_score = 1 - min(abs(skewness), 3) / 3
        
        # For kurtosis, closer to 0 is better (for excess kurtosis)
        kurtosis_score = 1 - min(abs(kurtosis), 3) / 3
        
        # Weighted combination
        weights = {
            'p_value': 0.5,  # Statistical tests are most important
            'skewness': 0.25,
            'kurtosis': 0.25
        }
        
        normality_score = weights['p_value'] * p_value_score + weights['skewness'] * skewness_score + weights['kurtosis'] * kurtosis_score
        
        return normality_score
    
    def calculate_hurst_exponent(self, time_series, lag_range=None):
        """
        Calculate the Hurst exponent of a time series.
        
        Args:
            time_series: Time series data
            lag_range: Range of lags to use in the calculation
            
        Returns:
            Hurst exponent (float between 0 and 1)
            - H < 0.5: Mean-reverting series
            - H = 0.5: Random walk
            - H > 0.5: Trending series
        """
        time_series = pd.Series(time_series).dropna()
        n = len(time_series)
        
        if lag_range is None:
            lag_range = range(2, min(100, int(n/4)))
        
        # Calculate the array of the variances of the lagged differences
        tau = []
        var = []
        
        for lag in lag_range:
            # Calculate price difference
            pp = np.subtract(time_series[lag:].values, time_series[:-lag].values)
            # Calculate variance of the difference
            var.append(np.var(pp))
            tau.append(lag)
        
        # Calculate the slope of the log plot -> the Hurst Exponent
        m = np.polyfit(np.log(tau), np.log(var), 1)
        hurst = m[0] / 2.0
        
        return hurst
    
    def analyze_pairs(self, pairs_df, calculate_metrics=True):
        """
        Analyze selected pairs to calculate additional metrics and trading parameters.
        
        Args:
            pairs_df: DataFrame containing the selected pairs
            calculate_metrics: Whether to calculate additional metrics
            
        Returns:
            Enhanced DataFrame with metrics for each pair
        """
        trend = self.trend
        if not calculate_metrics:
            return pairs_df
            
        # Create a copy to avoid modifying the original
        enhanced_pairs = pairs_df.copy()
        
        # Add columns for metrics
        enhanced_pairs['hedge_ratio'] = np.nan
        enhanced_pairs['half_life'] = np.nan
        enhanced_pairs['spread_std'] = np.nan
        enhanced_pairs['correlation'] = np.nan
        enhanced_pairs['adf_pvalue'] = np.nan
        enhanced_pairs['shapiro_p'] = np.nan
        enhanced_pairs['jarque_bera_p'] = np.nan
        enhanced_pairs['skewness'] = np.nan
        enhanced_pairs['kurtosis'] = np.nan
        enhanced_pairs['hurst_exponent'] = np.nan
        enhanced_pairs['normality_score'] = np.nan
        enhanced_pairs['overall_score'] = np.nan

        
        # Calculate metrics for each pair
        for idx, row in enhanced_pairs.iterrows():
            symbol_1 = row['Asset 1']
            symbol_2 = row['Asset 2']
            
            try:
                # Get price data
                pair_df = self.price_matrix[[symbol_1, symbol_2]]  #.dropna()
                spread, spread_mean, spread_std, z_score =self.calculate_spread(pair_df, symbol_1, symbol_2)

                # Test for normality of Z-score (Shapiro-Wilk test)
                shapiro_stat, shapiro_p = stats.shapiro(z_score)

                # Jarque-Bera test for normality
                jb_stat, jb_p = stats.jarque_bera(z_score)
                
                # Calculate skewness and kurtosis
                skewness = stats.skew(z_score)
                kurtosis = stats.kurtosis(z_score)

                # Calculate correlation
                correlation = pair_df[symbol_1].corr(pair_df[symbol_2])
                
                # Augmented Dickey-Fuller test on spread
                adf_result = adfuller(spread.dropna())
                adf_pvalue = adf_result[1]

                # Calculate Hurst exponent (measure of mean reversion)
                #hurst_exp = self.calculate_hurst_exponent(spread)
                
                # Store results
                #enhanced_pairs.at[idx, 'hedge_ratio'] = hedge_ratio
                #enhanced_pairs.at[idx, 'half_life'] = half_life
                enhanced_pairs.at[idx, 'spread_std'] = spread_std
                enhanced_pairs.at[idx, 'correlation'] = correlation
                enhanced_pairs.at[idx, 'adf_pvalue'] = adf_pvalue
                enhanced_pairs.at[idx, 'shapiro_p'] = shapiro_p
                enhanced_pairs.at[idx, 'jarque_bera_p'] = jb_p
                enhanced_pairs.at[idx, 'skewness'] = skewness
                enhanced_pairs.at[idx, 'kurtosis'] = kurtosis
                #enhanced_pairs.at[idx, 'hurst_exponent'] = hurst_exp
                enhanced_pairs.at[idx, 'normality_score'] = self.calculate_normality_score(shapiro_p, jb_p, skewness, kurtosis),
                enhanced_pairs.at[idx, 'overall_score'] = 0
                
            except Exception as e:
                logger.error(f"Error analyzing pair {symbol_1}-{symbol_2}: {str(e)}")

        
        enhanced_pairs['coint_score'] = 1 - enhanced_pairs['p-value']
        #enhanced_pairs['mean_reversion_score'] = 1 - enhanced_pairs['hurst_exponent']

        # Calculate overall score (weighted average of individual scores)
        weights = {
            'normality_score': 0.50,
            'coint_score': 0.50,
            #'mean_reversion_score': 0.25,

        }

        # Normalize scores before weighting
        for col in ['coint_score', 'normality_score']:
            if enhanced_pairs[col].max() > enhanced_pairs[col].min():
                enhanced_pairs[f'{col}_norm'] = (enhanced_pairs[col] - enhanced_pairs[col].min()) / \
                                            (enhanced_pairs[col].max() - enhanced_pairs[col].min())
            else:
                enhanced_pairs[f'{col}_norm'] = 0.5

        # Calculate overall score
        enhanced_pairs['overall_score'] = (
            weights['normality_score'] * enhanced_pairs['normality_score_norm'] +
            weights['coint_score'] * enhanced_pairs['coint_score_norm'] 
        )
    
        # Sort by overall score (descending)
        enhanced_pairs = enhanced_pairs.sort_values('overall_score', ascending=False)
                
        return enhanced_pairs
    
    def select_diverse_pairs(self, cointegrated_pairs_df):
            """
            Select a diverse set of pairs to avoid concentration risk.
            
            Args:
                cointegrated_pairs_df: DataFrame of cointegrated pairs
                max_pairs: Maximum number of pairs to select
                max_per_asset: Maximum pairs per individual asset
                
            Returns:
                DataFrame with selected pairs
            """
            max_pairs = self.max_pairs
            max_per_asset = self.max_per_asset
            # Take only pairs with valid p-values (not placeholder values)
            valid_pairs = cointegrated_pairs_df[cointegrated_pairs_df['p-value'] < 1.0].copy()
            
            # Limit the number of pairs per asset to ensure diversity
            diverse_pairs = valid_pairs.copy()
            
            # First limit by Asset 1
            diverse_pairs = diverse_pairs.groupby('Asset 1').head(max_per_asset).reset_index(drop=True)
            
            # Then limit by Asset 2
            diverse_pairs = diverse_pairs.groupby('Asset 2').head(max_per_asset).reset_index(drop=True)
            
            # Take top pairs up to the maximum
            selected_pairs = diverse_pairs.head(max_pairs)
            
            logger.info(f"Selected {len(selected_pairs)} diverse pairs for trading")
            return selected_pairs

    def run_full_analysis(self,  significance_level=0.02 ):
        """
        Run a complete analysis workflow to find and analyze pairs.
        
        Args:
            transform_type: Price transformation method
            trend: Trend component for cointegration test
            significance_level: P-value threshold
            max_pairs: Maximum pairs to select
            max_per_asset: Maximum pairs per asset
            calculate_metrics: Whether to calculate additional metrics
            
        Returns:
            DataFrame with selected pairs and metrics
        """
        """   
        transform_type = self.transform_type
        trend = self.trend
        max_per_asset = self.max_per_asset
        max_pairs = self.max_pairs"""
        # Find cointegrated pairs
        selected_pairs = self.find_cointegrated_pairs( significance_level=significance_level)

        # Analyze selected pairs
        selected_pairs = self.analyze_pairs(selected_pairs)
        
        # Select diverse pairs
        selected_pairs = self.select_diverse_pairs( selected_pairs )
        
        return selected_pairs
    
    def find_pairs_by_distance_method(self, n_clusters=10, max_pairs=50, distance_metric='euclidean'):
        """
        Alternative method to find pairs based on price pattern similarity.
        
        Args:
            n_clusters: Number of clusters for grouping similar assets
            max_pairs: Maximum number of pairs to return
            distance_metric: Distance metric for measuring similarity
            
        Returns:
            DataFrame with pairs sorted by distance
        """
        # Normalize all price series
        normalized_prices = self.price_matrix.copy()
        
        for col in normalized_prices.columns:
            # Normalize each series to start at 1
            normalized_prices[col] = normalized_prices[col] / normalized_prices[col].iloc[0]
        
        # Fill NaN values with column means (simple imputation)
        normalized_prices = normalized_prices.fillna(normalized_prices.mean())
        
        # Transpose for clustering (each row becomes an asset's price history)
        price_matrix_T = normalized_prices.T
        
        # Calculate distance matrix
        distance_matrix = pd.DataFrame(
            index=price_matrix_T.index,
            columns=price_matrix_T.index,
            data=np.zeros((len(price_matrix_T), len(price_matrix_T)))
        )
        
        # Calculate distances between all pairs
        pairs = []
        for i, asset1 in enumerate(price_matrix_T.index):
            for j, asset2 in enumerate(price_matrix_T.index):
                if i < j:  # Only calculate unique pairs
                    if distance_metric == 'euclidean':
                        dist = np.sqrt(np.sum((price_matrix_T.loc[asset1] - price_matrix_T.loc[asset2])**2))
                    elif distance_metric == 'correlation':
                        corr = np.corrcoef(price_matrix_T.loc[asset1], price_matrix_T.loc[asset2])[0, 1]
                        dist = 1 - abs(corr)  # Convert correlation to distance
                    
                    pairs.append((asset1, asset2, dist))
        
        # Create DataFrame and sort by distance
        pairs_df = pd.DataFrame(pairs, columns=['Asset 1', 'Asset 2', 'distance'])
        pairs_df = pairs_df.sort_values(by='distance')
        
        # Filter unwanted pairs
        pairs_df = self._filter_unwanted_pairs(pairs_df)
        
        # Select diverse pairs
        diverse_pairs = pairs_df.copy()
        diverse_pairs = diverse_pairs.groupby('Asset 1').head(2).reset_index(drop=True)
        diverse_pairs = diverse_pairs.groupby('Asset 2').head(2).reset_index(drop=True)
        
        # Return top pairs
        return diverse_pairs.head(max_pairs)
    
    def visualize_pair(self, symbol_1, symbol_2):
        """
        Visualize a trading pair, including normalized prices and spread.
        
        Args:
            symbol_1: First symbol
            symbol_2: Second symbol
            transform_type: Price transformation method
        """

        pair_df = self.price_matrix[[symbol_1, symbol_2]].dropna()
        
        if len(pair_df) == 0:
            logger.error(f"No valid data for pair {symbol_1}-{symbol_2}")
            return
        
        # Preprocess prices
        transformed_prices = self.preprocess_prices(pair_df, symbol_1, symbol_2)
        
        if transformed_prices is None:
            logger.error(f"Failed to preprocess prices for pair {symbol_1}-{symbol_2}")
            return
            
        x1, x2 = transformed_prices
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Normalize for plotting
        x1_norm = x1 / x1[0]
        x2_norm = x2 / x2[0]
        
        # Plot normalized prices
        ax1.plot(x1_norm, label=symbol_1)
        ax1.plot(x2_norm, label=symbol_2)
        ax1.set_title(f"Normalized Prices: {symbol_1} vs {symbol_2}")
        ax1.legend()
        ax1.grid(True)
        
        # Calculate and plot spread
        
        spread, spread_mean, spread_std, spread_zscore =self.calculate_spread(pair_df, symbol_1, symbol_2)

        ax2.plot(spread_zscore, label='Spread Z-Score')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(-1.0, color='green', linestyle='--', alpha=0.3)
        ax2.axhline(2.0, color='red', linestyle='--', alpha=0.3)
        ax2.axhline(-2.0, color='green', linestyle='--', alpha=0.3)
        ax2.set_title(f"Spread Z-Score (Half-Life: {self.calculate_half_life(spread):.1f} periods)")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

