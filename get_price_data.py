from binance import Client
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("binance_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    def __init__(self, api_key, api_secret, data_directory="data"):
        """Initialize the Binance data downloader with API credentials."""
        self.client = Client(api_key, api_secret)
        self.data_directory = data_directory
        
        # Create data directory if it doesn't exist
        os.makedirs(data_directory, exist_ok=True)
        
        # Available time intervals
        self.time_intervals = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }

    def get_tradable_symbols(self, quote_asset="USDT", margin_only=True):
        """
        Get a list of trading symbols based on filters.
        
        Args:
            quote_asset: Filter for specific quote asset (e.g., "USDT", "BTC")
            margin_only: If True, only return margin tradable pairs
            
        Returns:
            List of symbol strings
        """
        exchange_info = self.client.get_exchange_info()
        symbols_df = pd.DataFrame(exchange_info['symbols'])
        
        filtered_symbols = []
        for _, data in symbols_df.iterrows():
            # Apply filters
            if quote_asset in str(data['symbol']):
                if not margin_only or (margin_only and data.get('isMarginTradingAllowed', True)):
                    filtered_symbols.append(data['symbol'])
        
        logger.info(f"Found {len(filtered_symbols)} {quote_asset} trading pairs")
        return filtered_symbols

    def get_latest_data_date(self, symbol, timeframe):
        """
        Get the latest date in the existing data file.
        
        Returns None if file doesn't exist or is empty.
        """
        filepath = os.path.join(self.data_directory, f"{symbol}_{timeframe}.json")
        
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if not data:
                return None
                
            # Get the latest timestamp (close_time) from the data
            latest_timestamp = max(entry[6] for entry in data)  # 6 is the index for close time
            latest_date = datetime.fromtimestamp(latest_timestamp / 1000)  # Convert from ms to seconds
            
            return latest_date
            
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            logger.error(f"Error reading existing data for {symbol}: {e}")
            return None

    def download_and_update_data(self, symbols=None, timeframes=None, start_date=None):
        """
        Download and update price data for specified symbols and timeframes.
        
        Args:
            symbols: List of symbols to download data for. If None, uses all USDT margin pairs.
            timeframes: List of timeframe codes (e.g., ["1h", "4h"]).
                        If None, uses 1-hour timeframe.
            start_date: Default start date for new downloads. If None, uses "1 January, 2024".
                       For updates, uses the latest date in existing data.
        """
        # Set defaults
        if symbols is None:
            symbols = self.get_tradable_symbols(quote_asset="USDT", margin_only=True)
        
        if timeframes is None:
            timeframes = ["1h"]
            
        if start_date is None:
            start_date = "1 January, 2024"
            
        total_symbols = len(symbols)
        
        for timeframe_code in timeframes:
            if timeframe_code not in self.time_intervals:
                logger.warning(f"Unknown timeframe: {timeframe_code}, skipping.")
                continue
                
            binance_interval = self.time_intervals[timeframe_code]
            logger.info(f"Processing timeframe: {timeframe_code}")
            
            for idx, symbol in enumerate(symbols, 1):
                try:
                    file_path = os.path.join(self.data_directory, f"{symbol}_{timeframe_code}.json")
                    progress = (idx / total_symbols) * 100
                    
                    # Check if we have existing data and need to update
                    latest_date = self.get_latest_data_date(symbol, timeframe_code)
                    
                    if latest_date:
                        # Add a small buffer to avoid duplicate entries
                        start_time = latest_date + timedelta(minutes=1)
                        start_time_str = start_time.strftime("%d %B, %Y %H:%M:%S")
                        
                        # Get only the new data
                        new_klines = self.client.get_historical_klines(
                            symbol, binance_interval, start_time_str
                        )
                        
                        if not new_klines:
                            logger.info(f"No new data for {symbol} since {start_time_str}")
                            continue
                            
                        # Load existing data
                        with open(file_path, 'r') as f:
                            existing_data = json.load(f)
                            
                        # Append new data and save
                        updated_data = existing_data + new_klines
                        
                        with open(file_path, 'w') as f:
                            json.dump(updated_data, f)
                            
                        logger.info(f"Updated {symbol} with {len(new_klines)} new records. Progress: {progress:.2f}%")
                        
                    else:
                        # Download full history
                        klines = self.client.get_historical_klines(
                            symbol, binance_interval, start_date
                        )
                        
                        with open(file_path, 'w') as f:
                            json.dump(klines, f)
                            
                        logger.info(f"Downloaded {symbol} with {len(klines)} records. Progress: {progress:.2f}%")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
    
    def create_price_matrix(self, data_type="Close", timeframe="1h"):
        """
        Create a price matrix for all downloaded symbols.
        
        Args:
            data_type: "Close" for raw prices or "Price % Change" for percentage changes
            timeframe: The timeframe to use for the matrix
            
        Returns:
            DataFrame with price data for all symbols
        """
        # Get all files in the data directory for the specified timeframe
        file_pattern = f"_{timeframe}.json"
        all_files = [f for f in os.listdir(self.data_directory) if file_pattern in f]
        
        if not all_files:
            logger.error(f"No data files found for timeframe {timeframe}")
            return pd.DataFrame()
        
        # Start with BTC as the reference
        try:
            btc_file = f"BTCUSDT_{timeframe}.json"
            btc_path = os.path.join(self.data_directory, btc_file)
            
            with open(btc_path, 'r') as f:
                btc_data = json.load(f)
                
            # Create DataFrame from BTC data
            columns = ["Open time", "Open", "High", "Low", "Close", "Volume", 
                      "Close time", "Quote asset volume", "Number of trades", 
                      "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
                      
            df = pd.DataFrame(btc_data, columns=columns).astype(float)
            df = df.set_index("Close time")
            df = df.drop(columns=["Ignore"])
            
            # Rename the Close column to the symbol name
            df.rename(columns={"Close": "BTCUSDT"}, inplace=True)
            
            # Calculate percentage change if requested
            if data_type == "Price % Change":
                btc_close = df["BTCUSDT"].copy()
                df = pd.DataFrame()
                df["BTCUSDT %"] = btc_close.pct_change() * 100
                df = df.dropna()
                
            else:  # data_type == "Close"
                df = pd.DataFrame(df["BTCUSDT"])
            
            # Process all other symbols
            for filename in all_files:
                symbol = filename.split("_")[0]
                
                if symbol == "BTCUSDT":
                    continue
                    
                try:
                    file_path = os.path.join(self.data_directory, filename)
                    
                    with open(file_path, 'r') as f:
                        symbol_data = json.load(f)
                        
                    # Create DataFrame for this symbol
                    symbol_df = pd.DataFrame(symbol_data, columns=columns).astype(float)
                    symbol_df = symbol_df.set_index("Close time")
                    symbol_df = symbol_df.drop(columns=["Ignore"])
                    
                    # Rename and process
                    symbol_df.rename(columns={"Close": symbol}, inplace=True)
                    
                    if data_type == "Price % Change":
                        symbol_close = symbol_df[symbol].copy()
                        pct_change_df = pd.DataFrame()
                        pct_change_df[f"{symbol} %"] = symbol_close.pct_change() * 100
                        pct_change_df = pct_change_df.dropna()
                        df = df.join(pct_change_df)
                    else:  # data_type == "Close"
                        df = df.join(symbol_df[symbol])
                        
                    logger.info(f"Added {symbol} to price matrix")
                    
                except Exception as e:
                    logger.error(f"Error adding {symbol} to price matrix: {str(e)}")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error creating price matrix: {str(e)}")
            return pd.DataFrame()
            
    def save_price_matrix(self, matrix, filename):
        """Save the price matrix to a CSV file."""
        filepath = os.path.join(self.data_directory, filename)
        matrix.to_csv(filepath)
        logger.info(f"Price matrix saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Your Binance API keys (consider using environment variables for security)
    API_KEY = "5QG5zhhbtHrIKUtiEHOGClNf0fcBFCPF8cF4WluN6aELXpBkHA2xzgmh2RurN8e0"
    API_SECRET = "ycR4u0IyBxjm2QcwVi5zEmw4YeriAkXHQwjS5SucjKozm5pZyiuAkL3l3cwtfY4U"
    
    # Initialize the downloader
    downloader = BinanceDataDownloader(API_KEY, API_SECRET, data_directory="binance_data")
    
    # Get all margin tradable USDT pairs
    symbols = downloader.get_tradable_symbols(quote_asset="USDT", margin_only=True)
    
    # Download/update data for 1-hour timeframe
    downloader.download_and_update_data(symbols=symbols, timeframes=[ "1d"])
    
    # Create price matrices
   # close_matrix = downloader.create_price_matrix(data_type="Close", timeframe="4h")
   # pct_matrix = downloader.create_price_matrix(data_type="Price % Change", timeframe="4h")
    close_matrix = downloader.create_price_matrix(data_type="Close", timeframe="1d")
    #pct_matrix = downloader.create_price_matrix(data_type="Price % Change", timeframe="1d")
    
    # Save the matrices
    #downloader.save_price_matrix(close_matrix, "price_matrix_4h_close.csv")
    #downloader.save_price_matrix(pct_matrix, "price_matrix_4h_pct_change.csv")
    downloader.save_price_matrix(close_matrix, "price_matrix_1d_close.csv")
    #downloader.save_price_matrix(pct_matrix, "price_matrix_1d_pct_change.csv")