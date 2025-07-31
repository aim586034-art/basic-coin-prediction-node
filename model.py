import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY


binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    # Helper function to fix corrupted timestamps
    def fix_timestamps(timestamps, source=""):
        """Fix corrupted years in timestamps by applying mathematical correction"""
        try:
            return pd.to_datetime(timestamps, unit="ms")
        except (pd._libs.tslibs.np_datetime.OutOfBoundsDatetime, OverflowError) as e:
            print(f"Date conversion error {source}: {e}")
            current_year = pd.Timestamp.now().year
            print(f"Using current year: {current_year}")
            
            current_year_start = pd.Timestamp(f'{current_year}-01-01').value // 1000000
            next_year_end = pd.Timestamp(f'{current_year + 1}-12-31').value // 1000000
            
            fixed = []
            for i, ts in enumerate(timestamps):
                try:
                    if ts > next_year_end * 100: 
                        # Apply correction by scaling down
                        scale_factor = ts // current_year_start  
                        if scale_factor > 1000: 
                            corrected_ts = ts // scale_factor
                            fixed.append(corrected_ts)
                            print(f"Scaled down {source} timestamp by factor ~{scale_factor}")
                        else:
                            # Use sequential dates starting from current year
                            sequential_ts = current_year_start + i * 24 * 60 * 60 * 1000  # Add days
                            fixed.append(sequential_ts)
                            print(f"Replaced {source} timestamp with sequential date")
                    else:
                        # Timestamp seems reasonable, test conversion
                        test_conversion = pd.to_datetime(ts, unit="ms")
                        fixed.append(ts)
                except (pd._libs.tslibs.np_datetime.OutOfBoundsDatetime, OverflowError):
                    # For any conversion errors, use sequential dates
                    sequential_ts = current_year_start + i * 24 * 60 * 60 * 1000  # Add days
                    fixed.append(sequential_ts)
                    print(f"Replaced out-of-bounds {source} timestamp with sequential date")
            
            return pd.to_datetime(fixed, unit="ms")

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)
            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time", "open", "high", "low", "close", "volume",
                "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"
            ]
            
            # Use helper function to fix timestamps
            df.index = fix_timestamps(df["end_time"], "binance")
            df.index.name = "date"
            df["date"] = df.index
            price_df = pd.concat([price_df, df])

        if not price_df.empty:
            price_df.sort_index().to_csv(training_price_data_path, index=True)
            
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = ["timestamp", "open", "high", "low", "close"]
                
                # Use helper function to fix timestamps
                df["date"] = fix_timestamps(df["timestamp"], "coingecko")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                df["date"] = df.index
                price_df = pd.concat([price_df, df])

        if not price_df.empty:
            price_df.sort_index().to_csv(training_price_data_path, index=True)


def load_frame(price_data, timeframe):
    frame = pd.DataFrame(price_data)
    
    # Fix incorrect year in dates
    def fix_date(date_str):
        if isinstance(date_str, str):
            current_year = pd.Timestamp.now().year
            if date_str.startswith('570'):
                # Extract everything after the year and prepend current year
                return f'{current_year}-' + date_str[date_str.find('-')+1:]
            elif date_str.startswith('57048-'):
                # Handle the specific case of 57048 year showing up in binance data
                return date_str.replace('57048-', f'{current_year}-')
        return date_str

    print(f"Loading data...")
    df = frame.copy()
    
    # Debug: print column names and first few rows
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head:\n{df.head()}")
    
    # Handle different data formats
    if 'date' in df.columns:
        # Format with explicit date column
        df['date'] = df['date'].apply(fix_date)
        df['date'] = pd.to_datetime(df['date'], format='ISO8601')
        df = df.loc[:,['open','high','low','close']].dropna()
        df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
        df.set_index('date', inplace=True)
    elif df.index.dtype == 'datetime64[ns]':
        # Data already has datetime index
        df = df.loc[:,['open','high','low','close']].dropna()
        df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
    else:
        # Try to convert index to datetime if it's not already
        try:
            df.index = pd.to_datetime(df.index)
            df = df.loc[:,['open','high','low','close']].dropna()
            df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
        except:
            raise ValueError(f"Cannot process data format. Available columns: {df.columns.tolist()}")
    
    df.sort_index(inplace=True)
    
    # Show data before resampling
    print(f"Before resampling (raw data): {df.shape}")
    
    # Apply resampling to specified timeframe
    resampled_df = df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()
    
    # Show data after resampling
    print(f"After resampling to {timeframe}: {resampled_df.shape}")
    print(f"Resampled data:\n{resampled_df.head()}")
    
    return resampled_df

def train_model(timeframe):
    # Load the price data
    price_data = pd.read_csv(training_price_data_path, index_col=0)
    print(f"Raw CSV data shape: {price_data.shape}")
    
    df = load_frame(price_data, timeframe)
    print(f"After load_frame and resampling (timeframe={timeframe}): {df.shape}")

    if len(df) < 3:
        raise ValueError(f"Insufficient data for training. Need at least 3 rows, got {len(df)}. "
                        f"Try using a smaller timeframe (e.g., '1H', '30min') or increase TRAINING_DAYS.")

    print(df.tail())

    # Calculate log returns as target variable instead of raw prices
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Shift log returns to predict next period, then align data properly
    shifted_log_returns = log_returns.shift(-1)
    
    # Drop rows where we don't have both features and target
    # We need both current period features and next period log returns
    valid_mask = ~(log_returns.isna() | shifted_log_returns.isna())
    print(f"Valid data points: {valid_mask.sum()} out of {len(valid_mask)}")
    
    X_train = df[valid_mask][['open', 'high', 'low', 'close']]
    y_train = shifted_log_returns[valid_mask].values
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    
    if len(X_train) == 0:
        raise ValueError(f"No valid training data after alignment. Original shape: {df.shape}, Valid mask sum: {valid_mask.sum()}")
    
    print(f"Log returns stats - Mean: {np.mean(y_train):.6f}, Std: {np.std(y_train):.6f}")

    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    # Add more models here
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    model.fit(X_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")


def process_current_data(data, data_provider, timeframe):
    """Process current day data for inference."""
    df = data.copy()
    
    print(f"Processing current data...")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head:\n{df.head()}")
    
    if data_provider == "binance":
        # Binance current day data - already has date column
        df = df.loc[:,['date','open','high','low','close']].dropna()
        df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif data_provider == "coingecko":
        # Coingecko current day data - already has date column
        df = df.loc[:,['date','open','high','low','close']].dropna()
        df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    df.sort_index(inplace=True)
    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()


def get_inference(token, timeframe, region, data_provider):
    """Load model and predict log returns of closing price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price data
    if data_provider == "coingecko":
        current_data = download_coingecko_current_day_data(token, CG_API_KEY)
        X_new = process_current_data(current_data, data_provider, timeframe)
    else:
        current_data = download_binance_current_day_data(f"{TOKEN}USDT", region)
        X_new = process_current_data(current_data, data_provider, timeframe)
    
    print(X_new.tail())
    print(X_new.shape)

    # Predict and return log returns only
    predicted_log_return = loaded_model.predict(X_new)
    
    print(f"Predicted log return: {predicted_log_return[0]:.6f}")
    
    return predicted_log_return[0]