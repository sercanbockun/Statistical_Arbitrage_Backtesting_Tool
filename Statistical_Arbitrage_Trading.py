import pandas as pd
import numpy as np
# from get_price_data import *
# from Indicators_ import *
#client     = Client("iFC5F3IsZ8CIuaVLwxJM4fBcjsYm8LSkIbW4m2tjPuGVg9r3aSLQVf0sEMJzxJBG", "aGMNNmLA0CZlw2LO0HoEU2LM6FNE8I9ZayS2nXOmC3rpO13zuRi6LAiDkc0uduni")
import matplotlib.pyplot as plt
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
from statsmodels.regression.rolling import RollingOLS
from itertools import product
import warnings
import datetime
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


from helper_funcs import *
from pair_finder import StatisticalArbitragePairFinder

class Statistical_Arbitrage():

    # Parameters are used to find the best parameters in the grid search that maximizes sharpe_ratio
    parameter_search_range= {"z_score_thresh": list(map(lambda x: x/10.0, range(5,36,5))), "lookback_window": range(10,131,10)}
    params_ = list(product(*parameter_search_range.values()))    

    def __init__(self, price_matrix, fee_rate, time_trend, quadratic_time_trend, max_pairs = 50, max_per_asset = 2 ):
        self.price_matrix = price_matrix
        self.fee_rate = fee_rate
        self.time_trend = time_trend
        self.quadratic_time_trend = quadratic_time_trend
        self.max_pairs = max_pairs
        self.max_per_asset = max_per_asset
        
    
    
    def Trade_Pair_Arbitrage(self, start_time, end_time, pair_1, pair_2, z_score_thresh, lookback_window,  show_on_graph  ): 
         
        price_matrix  = self.price_matrix
        fee_rate = self.fee_rate
        time_trend = self.time_trend
        quadratic_time_trend = self.quadratic_time_trend           

                                                                                                               #time_trend = Should time be a factor in the regression? timeframe _1H, if fee rate is %1 input 1 
        trades_df = pd.DataFrame(columns = [ "Price_1", "Price_2", "Price_1 % Change", "Price_2 % Change", 
                                            "distance", "std", "Z Score",                                          
                                     "Return on Long", 
                                     "Active Position", "Wallet" , "Hodling Wallet" ])
         
        max_period = lookback_window
        indexes = price_matrix.index.values
        index_steps = indexes[-1]-indexes[-2]    
    
        price_matrix_ = price_matrix.loc[(start_time - 2*index_steps*max_period):end_time, :]
        # Pozisyona girilmediği zamanlarda günlük faize yatırdığımızı varsatyalım?
        
        trades_df["Price_1"] = price_matrix_[pair_1]
        trades_df["Price_2"] = price_matrix_[pair_2] 
        trades_df["Hodling Wallet"] = price_matrix_["BTCUSDT"] 
        #trades_df["Hodling Wallet"] = price_matrix_["AAPL"] 
        trades_df = trades_df[trades_df["Price_1"].notna()]
        trades_df = trades_df[trades_df["Price_2"].notna()]
    
        trades_df["Price_1 % Change"] = (trades_df["Price_1"].diff()).div(trades_df["Price_1"].shift(+1))*100
        trades_df["Price_2 % Change"] = (trades_df["Price_2"].diff()).div(trades_df["Price_2"].shift(+1))*100
    
        trades_df["Return on Long"] = (trades_df["Price_1 % Change"] - trades_df["Price_2 % Change"])/2 
    
        if time_trend == False and quadratic_time_trend == False:
            roll_reg = RollingOLS.from_formula('Price_1 ~ Price_2', window= lookback_window, data=trades_df)
            model = roll_reg.fit()
            parameters = model.params
          #  print(parameters)
            trades_df["distance"] = trades_df["Price_1"] - trades_df["Price_2"]*parameters["Price_2"] - parameters["Intercept"]
            trades_df["std"] = trades_df["distance"].rolling(lookback_window).std()
            trades_df["Z Score"] = (trades_df["distance"] -  trades_df["distance"].rolling(lookback_window).mean()) /trades_df["std"]
            
        if time_trend == True and quadratic_time_trend == False:
            trades_df['Time'] = range(len(trades_df))
            roll_reg = RollingOLS.from_formula('Price_1 ~ Price_2 + Time', window= lookback_window, data=trades_df)
            model = roll_reg.fit()
            parameters = model.params
         #   print(parameters)
            trades_df["distance"] = trades_df["Price_1"] - trades_df["Price_2"]*parameters["Price_2"] - trades_df["Time"]*parameters["Time"] - parameters["Intercept"]
            trades_df["std"] = trades_df["distance"].rolling(lookback_window).std()
            trades_df["Z Score"] = (trades_df["distance"] -  trades_df["distance"].rolling(lookback_window).mean()) /trades_df["std"]
            
    
        elif quadratic_time_trend == True:
            trades_df['Time'] = range(len(trades_df))
            trades_df['Time_squared'] = trades_df['Time'] ** 2
            roll_reg = RollingOLS.from_formula('Price_1 ~ Price_2 + Time + Time_squared', window= lookback_window, data=trades_df)
            model = roll_reg.fit()
            parameters = model.params
           # print(parameters)
            trades_df["distance"] = trades_df["Price_1"] - trades_df["Price_2"]*parameters["Price_2"] - trades_df["Time"]*parameters["Time"] - trades_df["Time_squared"]*parameters["Time_squared"] - parameters["Intercept"]
            trades_df["std"] = trades_df["distance"].rolling(lookback_window).std()
            trades_df["Z Score"] = (trades_df["distance"] -  trades_df["distance"].rolling(lookback_window).mean()) /trades_df["std"]
            
        
        
        trades_df = trades_df.loc[start_time:end_time, :]        
        trades_df.loc[trades_df[trades_df["Z Score"] <= -1* z_score_thresh].index,"Active Position"] = 1
        trades_df.loc[trades_df[trades_df["Z Score"] >= 1* z_score_thresh].index,"Active Position"] = -1
        trades_df.loc[trades_df[(trades_df["Z Score"] > -1* 0.25) & (trades_df["Z Score"] < 1* 0.25) ].index,"Active Position"] = 0
                
    
        trades_df["Active Position"] = trades_df["Active Position"].fillna(method="ffill")
    
        trades_df.loc[((trades_df["Active Position"] == 1) & (trades_df["Z Score"] > -0.25)), "Active Position" ] = 0
        trades_df.loc[((trades_df["Active Position"] == -1) & (trades_df["Z Score"] < 0.25)), "Active Position" ] = 0
        #print( trades_df[(trades_df["Active Position"] == 1) & (trades_df["Z Score"] > -0.25)])
        trades_df["Active Position"] = trades_df["Active Position"].shift(+1)
        trades_df["Active Position"] = trades_df["Active Position"].fillna(0)
    
    
        trades_df["Return on Position"] = (trades_df["Return on Long"])*trades_df["Active Position"] 
        
        
        trades_df["Fee Rate"] = trades_df["Active Position"] - trades_df["Active Position"].shift(+1)
        trades_df["Fee Rate 2"] = trades_df["Active Position"] - trades_df["Active Position"].shift(-1)
        trades_df["Fee Rate"] = trades_df["Fee Rate"] + trades_df["Fee Rate 2"]
        trades_df["Fee Rate"] = np.abs(trades_df["Fee Rate"])
        trades_df["Fee Rate"] = trades_df["Fee Rate"] * fee_rate
        trades_df["Fee Rate"] = trades_df["Fee Rate"].fillna(0)
      # trades_df = trades_df.drop(columns = ["Fee Rate 2"])
     
        #Determining Entry and Exit Points
        trades_df["Entry"] = np.abs(trades_df["Active Position"]) - np.abs(trades_df["Active Position"].shift(+1).fillna(0))
        trades_df.loc[trades_df["Entry"] == -1, "Entry"] = 0
        trades_df.loc[(trades_df["Active Position"] - trades_df["Active Position"].shift(+1).fillna(0)) == 2, "Entry" ] = 1
        trades_df.loc[(trades_df["Active Position"] - trades_df["Active Position"].shift(+1).fillna(0)) == -2, "Entry" ] = 1
            
        trades_df["Exit"] = np.abs(trades_df["Active Position"]) - np.abs(trades_df["Active Position"].shift(-1).fillna(0))
        trades_df.loc[trades_df["Exit"] == -1, "Exit"] = 0
        trades_df.loc[(trades_df["Active Position"] - trades_df["Active Position"].shift(-1).fillna(0)) == 2, "Exit" ] = 1
        trades_df.loc[(trades_df["Active Position"] - trades_df["Active Position"].shift(+-1).fillna(0)) == -2, "Exit" ] = 1
        ############
        
        
        trades_df.loc[trades_df["Return on Position"] != 0, "Return on Position"] = trades_df.loc[trades_df["Return on Position"] != 0, "Return on Position"] - trades_df.loc[trades_df["Return on Position"] != 0, "Fee Rate"]
        #print(trades_df[["Fee Rate", "Active Position", "Return on Position", "Return on Long"]].head(60))
    
        trades_df["Wallet"] =  (100 + trades_df["Return on Position"] )/100 
        trades_df.loc[trades_df.index[0], "Wallet"] = 10000
        trades_df["Wallet"] = trades_df["Wallet"].cumprod()
        trades_df["Hodling Wallet"] = trades_df["Hodling Wallet"] * 10000 / (trades_df.loc[trades_df.index[0], "Hodling Wallet"])
        
        trades_df["Pair 1"] = np.nan
        trades_df["Pair 2"] = np.nan
        trades_df.loc[ trades_df.index[0], "Pair 1"] = pair_1
        trades_df.loc[trades_df.index[0], "Pair 2"] = pair_2
        
        if show_on_graph:
            plt.figure(figsize= (120,5))
            plt.title("Z Score")
            plt.plot(trades_df["Z Score"], linewidth = 2, alpha = 0.8)
           # plt.plot(trades_df["Hodling Wallet"], linewidth = 2, alpha = 0.8)
            plt.grid()
            plt.show()
            
            plt.figure(figsize= (120,5))
            plt.title("Strategy Wallet vs BTC Hodl")
            plt.plot(trades_df["Wallet"], linewidth = 2, alpha = 0.8)
            plt.plot(trades_df["Hodling Wallet"], linewidth = 2, alpha = 0.8)
            plt.plot(trades_df["Active Position"]*10000 , linewidth = 1, alpha = 0.25, color = "black")
            plt.grid()
            plt.show()
        

        return trades_df

    
    def Find_Best_Parameters_Pair_Trading(self, timeframe, start_time, end_time, pair_1, pair_2, risk_free_rate ):
    
        parameters = self.params_
        # following is qhat the above definition returns: this is a grid search with 2 parameters
        #parameter_search_range= {"z_score_thresh": list(map(lambda x: x/10.0, range(10,36,5))), "lookback_window": range(10,131,10)}
        search_results = {}
        for params in parameters:
            
            try:
                trades_df = self.Trade_Pair_Arbitrage( start_time, end_time, pair_1, pair_2, *params,  False)
              
                objective_func =  Sharpe_Ratio(trades_df["Wallet"], timeframe, risk_free_rate)  #3(win_prob)* ((float((trades_df.loc[trades_df.index[-1],"Wallet"] - 10000)/10000)*100)**1)  
                search_results[params] = objective_func
            except Exception as e: 
                print("Location 1: ", e)
                search_results[params] = -99
            #print(params, objective_func)
            
        values = list(search_results.values())
        params = list(search_results.keys())
        best_params = params[values.index(max(values))]
        best_value = max(values)
        
        return [best_params, best_value, search_results]
    
    
    def smooth_grid_search(self, search_results, window_size=1):
        smoothed_results = {}
        params_list = list(search_results.keys())
        
        for params in params_list:
            z_score, lookback = params
            total_value = search_results[params]
            count = 1
            
            # Check neighboring points
            for dz in range(-window_size, window_size + 1):
                for dl in range(-window_size, window_size + 1):
                    neighbor = (z_score + dz * 0.5, lookback + dl * 10)  # Adjust steps based on your grid
                    if neighbor in search_results and neighbor != params:
                        total_value += search_results[neighbor]
                        count += 1
            
            smoothed_results[params] = total_value / count
        
        # Find best smoothed parameters
        best_params = max(smoothed_results.items(), key=lambda x: x[1])[0]
        best_value = smoothed_results[best_params]
        return [best_params, best_value, smoothed_results]
    
    
    def plot_grid_search_results(self, search_results):
        # Extract parameters and values from the dictionary
        z_score_thresh = [params[0] for params in search_results.keys()]
        lookback_window = [params[1] for params in search_results.keys()]
        sharpe_ratios = list(search_results.values())
        
        # Create a meshgrid for the surface plot
        X = np.array(lookback_window)
        Y = np.array(z_score_thresh)
        Z = np.array(sharpe_ratios)
        
        # Get unique values for the grid
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
        
        # Reshape Z values to match the grid
        Z_grid = np.zeros(X_grid.shape)
        for i, y in enumerate(y_unique):
            for j, x in enumerate(x_unique):
                # Find corresponding Z value
                for k, (z_score, lookback) in enumerate(search_results.keys()):
                    if z_score == y and lookback == x:
                        Z_grid[i, j] = sharpe_ratios[k]
                        break
        
        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                            cmap='viridis',
                            edgecolor='none',
                            alpha=0.8)
        
        # Add a color bar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('Lookback Window')
        ax.set_ylabel('Z-score Threshold')
        ax.set_zlabel('Sharpe Ratio')
        ax.set_title('Grid Search Results - Parameter Optimization')
        
        # Add the best parameters point
        best_params, best_value = max(search_results.items(), key=lambda x: x[1])
        ax.scatter([best_params[1]], [best_params[0]], [best_value], 
                color='red', s=100, label='Best Parameters')
        ax.legend()
        
        # Rotate the plot for better viewing
        ax.view_init(30, 45)
        
        plt.show()
    
    
    def Pair_Trading_Portfolio_Backtest_Time_Validation(self, timeframe, start_time, end_time, train_days, test_days, selected_pairs_df= []): 
        # day= 86400000
    
        price_matrix = self.price_matrix
        params_ = self.params_
        time_trend = self.time_trend
        quadratic_time_trend = self.quadratic_time_trend
        max_pairs = self.max_pairs
        max_per_asset = self.max_per_asset
        

        if time_trend and quadratic_time_trend:
            trend = 'ctt'
        elif time_trend and quadratic_time_trend == False:
            trend = 'ct'
        elif time_trend == False and quadratic_time_trend == False:
            trend = 'c'
        risk_free_rate = 0.04
        day = 86400000
        time = start_time + train_days*day
        results_df = pd.DataFrame(columns=("Train Start Time", "Train End Time", "Train Wallet", "Train Hodling Wallet", "Train Portfolio Pairs",
                                        "Test Start Time", "Test End Time", "Test Wallet", "Test Hodling Wallet", 
                                        "Beat the Market"))
        
        period = 0
        selected_pairs_df_by_period = pd.DataFrame()
        while time < end_time:
            period += 1
            print("Period: ", period, " Time: ", str(pd.to_datetime(str(time),unit='ms')))
            print("Period: ", period, " Time: ", time-train_days*day) 
          
            try:
                if time+ test_days*day >= end_time:
                    test_days = (end_time-time)/day
                
                price_matrix_ = price_matrix.loc[(time-train_days*day):time, :]

                
                if isinstance(selected_pairs_df, list):
                    #print("GİRDİ")
                    selected_pairs = find_cointegrated_pairs(price_matrix_, "log","c",  0.05)

                    #pair_finder = StatisticalArbitragePairFinder(price_matrix_, min_history_pct=1, trend= trend, transform_type="log", max_pairs = max_pairs, max_per_asset = max_per_asset)
                    #selected_pairs = pair_finder.run_full_analysis( significance_level=0.02 )[["Asset 1", "Asset 2","p-value", "coint_score_norm", "normality_score_norm", "overall_score"]]
                    selected_pairs["Start Time"] = np.nan
                    selected_pairs["End Time"] = np.nan
                    selected_pairs["Train Days"] = train_days
                    selected_pairs["Test Days"] = test_days
                    selected_pairs["Train Wallet"] = np.nan
                    selected_pairs["Test Wallet"] = np.nan
                    selected_pairs["Parameters"] = np.nan
                    selected_pairs["Train Sharpe Ratio"] = np.nan
                    selected_pairs["Test Sharpe Ratio"] = np.nan
                    print(selected_pairs)
                    
                elif isinstance(selected_pairs_df, pd.DataFrame):
                    selected_pairs = selected_pairs_df
                    selected_pairs["Start Time"] = np.nan
                    selected_pairs["End Time"] = np.nan
                    selected_pairs["Train Days"] = train_days
                    selected_pairs["Test Days"] = test_days
                    selected_pairs["Train Wallet"] = np.nan
                    selected_pairs["Test Wallet"] = np.nan
                    selected_pairs["Parameters"] = np.nan
                    selected_pairs["Train Sharpe Ratio"] = np.nan
                    selected_pairs["Test Sharpe Ratio"] = np.nan
                
                
                pair_no = 0

                
                for i in range(0,len(selected_pairs)):
                    asset_1 = selected_pairs.iloc[i]["Asset 1"]
                    asset_2 = selected_pairs.iloc[i]["Asset 2"]
                    pair_no += 1
                    print(pair_no, asset_1, asset_2)
     
                    try:
    
                        #nicest_things = self.Find_Best_Parameters_Pair_Trading( timeframe, time-train_days*day, time, asset_1, asset_2, risk_free_rate)
                        #parameters = nicest_things[0]

                        best_params, best_value, search_results = self.Find_Best_Parameters_Pair_Trading( timeframe, time-train_days*day, time, asset_1, asset_2, risk_free_rate)
                        smoothed_best_params, smoothed_best_value, smoothed_search_results = self.smooth_grid_search(search_results, window_size=1)
                        parameters = smoothed_best_params

                    except Exception as e: 
                        print("FAIL POINT 1")
                        print(e)
                        parameters = [3, 30]
        
                    selected_pairs.iat[i, selected_pairs.columns.get_loc("Parameters")]  = str(list(parameters)[-2:])
                    #selected_pairs_df.loc[selected_pairs_df.index[-1],"Parameters"] = list(parameters)[-2:]
                    
                    try:
    
                        train_results_df = self.Trade_Pair_Arbitrage( time-train_days*day, time, asset_1, asset_2, *parameters, False)
                        train_wallet = train_results_df.loc[train_results_df.index[-1],"Wallet"]
                        selected_pairs.loc[selected_pairs.index[i],"Start Time"] = time-train_days*day
                        selected_pairs.loc[selected_pairs.index[i],"End Time"] = time
                        selected_pairs.loc[selected_pairs.index[i],"Train Wallet"] = train_wallet
                        #selected_pairs.loc[selected_pairs.index[i],"Train Sharpe Ratio"] = nicest_things[1]
                        selected_pairs.loc[selected_pairs.index[i],"Train Sharpe Ratio"] = smoothed_best_value
    
                    except Exception as e:
                        print("FAIL POINT 2")
                        print(e)
                        train_wallet = np.nan
                        selected_pairs.loc[selected_pairs.index[i],"Train Wallet"] = train_wallet
    
                        
                        
                    try:
    
                        test_results_df = self.Trade_Pair_Arbitrage(  time,  time + test_days*day, asset_1, asset_2, *parameters, False)
                        test_wallet = test_results_df.loc[test_results_df.index[-1],"Wallet"] 
                        selected_pairs.loc[selected_pairs.index[i],"Test Wallet"] = test_wallet
                        selected_pairs.loc[selected_pairs.index[i],"Test Sharpe Ratio"] = Sharpe_Ratio(test_results_df["Wallet"], timeframe, risk_free_rate)
    
                    except Exception as e: 
                        print("FAIL POINT 3")
                        print(e)
                        test_wallet = np.nan
                        selected_pairs.loc[selected_pairs.index[i],"Test Wallet"] = test_wallet
                
                    
    
                selected_pairs["Backtesting_Period"] = period
                selected_pairs_df_by_period = pd.concat([selected_pairs_df_by_period ,selected_pairs])

                btc_train_price = price_matrix.loc[(time-train_days*day):time, :]["BTCUSDT"]
                btc_train_hodl = 10000*( 1 + (( btc_train_price.iloc[-1] - btc_train_price.iloc[0])/btc_train_price.iloc[0] ) )
                
                btc_test_price = price_matrix.loc[   time :(time + test_days*day), :]["BTCUSDT"]     
                btc_test_hodl = 10000*( 1 + (( btc_test_price.iloc[-1] - btc_test_price.iloc[0])/btc_test_price.iloc[0] ) )      
       

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "Train Start Time" : str(pd.to_datetime(str(time-train_days*day),unit='ms')),
                    "Train End Time" :  str(pd.to_datetime(str(time),unit='ms')),
                    "Train Wallet" : selected_pairs["Train Wallet"].mean(),
                    "Train Hodling Wallet" : btc_train_hodl,
                    
        
                    "Test Start Time" :str(pd.to_datetime(str(time),unit='ms')),
                    "Test End Time" :str(pd.to_datetime(str(time + test_days*day),unit='ms')),
                    "Test Wallet" : selected_pairs["Test Wallet"].mean(),
                    "Test Hodling Wallet" :btc_test_hodl,
        
                    "Beat the Market": selected_pairs["Test Wallet"].mean() > btc_test_hodl,
        
                    }])], ignore_index=True)
                
                time += test_days*day
    
    
            except Exception as e: 

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "Train Start Time" : str(pd.to_datetime(str(time-train_days*day),unit='ms')),
                    "Train End Time" :  str(pd.to_datetime(str(time),unit='ms')),
    
        
                    "Test Start Time" :str(pd.to_datetime(str(time),unit='ms')),
                    "Test End Time" :str(pd.to_datetime(str(time + test_days*day),unit='ms')),
    
                    
                    }])], ignore_index=True)
                
                
                time += test_days*day
                print("FAIL POINT 4")
                print(e)
    
    
            
        return results_df, selected_pairs_df_by_period


    # Portfolio analysis on backtested results:

    def Create_Historical_Wallets(self, selected_pairs_df_by_period): #  Statistical_Arbitrage instance, start_time, train_days, test_days, day are required too, 
        price_matrix  = self.price_matrix
        train_wallet_dfs_by_periods = {}    
        test_wallet_dfs_by_periods = {}
        day = 86400000
        start_time__ = selected_pairs_df_by_period.iloc[0, selected_pairs_df_by_period.columns.get_loc('Start Time')]
        start_time =  datetime.datetime.strptime(start_time__, '%Y-%m-%d %H:%M:%S.%f')
        start_time = start_time.timestamp()*1000
        print(start_time)
        train_days = selected_pairs_df_by_period.iloc[0, selected_pairs_df_by_period.columns.get_loc("Train Days")]
        test_days = selected_pairs_df_by_period.iloc[0, selected_pairs_df_by_period.columns.get_loc("Test Days")]

        for period_no in range(1, selected_pairs_df_by_period["Backtesting_Period"].max()+ 1):
            
            start_time_ = start_time + test_days*(period_no-1)*day
            end_time_ = start_time + train_days*day + test_days*(period_no-1)*day
            
            train_wallet_df = pd.DataFrame()
            test_wallet_df = pd.DataFrame()
            train_wallet_df["BTCUSDT"] = price_matrix.loc[start_time_:end_time_, "BTCUSDT"]
            train_wallet_df["BTCUSDT"] = train_wallet_df["BTCUSDT"]* 10000 / (train_wallet_df.loc[train_wallet_df.index[0], "BTCUSDT"])
            test_wallet_df["BTCUSDT"] = price_matrix.loc[end_time_:end_time_ + test_days*day, "BTCUSDT"] #df.loc[end_time_:end_time_ + 50*day, "Close"]
            test_wallet_df["BTCUSDT"] = test_wallet_df["BTCUSDT"]* 10000 / (test_wallet_df.loc[test_wallet_df.index[0], "BTCUSDT"])
           
            portfolio_df = selected_pairs_df_by_period.loc[selected_pairs_df_by_period["Backtesting_Period"] == period_no]
            for i in range(0, len(portfolio_df)) :
                asset_1 = portfolio_df.iloc[i, portfolio_df.columns.get_loc("Asset 1")]
                asset_2 = portfolio_df.iloc[i, portfolio_df.columns.get_loc("Asset 2")]
            
                parameters = portfolio_df.iloc[i, portfolio_df.columns.get_loc("Parameters")]
                parameters = parameters.replace("[","")
                parameters = parameters.replace("]","")
                parameters = parameters.replace(" ","")
                parameters = parameters.split(',')
            
                z_score_thresh = float(parameters[0])
                lookback_window = int(parameters[1])
            
                print("Period", period_no, asset_1, asset_2, z_score_thresh, lookback_window )
                try:
                    train_trades_df = self.Trade_Pair_Arbitrage( start_time_, end_time_, asset_1, asset_2, z_score_thresh, lookback_window , False)
                    test_trades_df =  self.Trade_Pair_Arbitrage( end_time_, (end_time_ + test_days*day), asset_1, asset_2, z_score_thresh, lookback_window, False)
                except Exception as e: print(e)
                
                train_wallet_name = "Pair NO: " + str(i+1) + " (" +asset_1+ "/" + asset_2+ ")"  + " Train"
                test_wallet_name = "Pair NO: " + str(i+1) +" (" +asset_1+ "/" + asset_2+ ")"  + " Test"
                train_wallet_df[train_wallet_name] = train_trades_df["Wallet"]
                test_wallet_df[test_wallet_name] = test_trades_df["Wallet"]
        
            train_wallet_df["Overall Train Wallet"] = train_wallet_df.iloc[:, 1::].mean(axis = 1)
            test_wallet_df["Overall Test Wallet"] = test_wallet_df.iloc[:, 1::].mean(axis = 1)
            train_wallet_dfs_by_periods["Period " + str(period_no) ] = train_wallet_df
            test_wallet_dfs_by_periods["Period " + str(period_no) ] = test_wallet_df
            
        return    train_wallet_dfs_by_periods, test_wallet_dfs_by_periods 


    def Plot_Historical_Wallets(self, train_wallet_dfs_by_periods, test_wallet_dfs_by_periods):        
        wallet_period_keys = list(train_wallet_dfs_by_periods.keys())
        for period_key_ in wallet_period_keys:
            
            train_period_wallets = train_wallet_dfs_by_periods[period_key_]
            test_period_wallets = test_wallet_dfs_by_periods[period_key_]
        
            plt.figure(figsize= (120,5))
            plt.title(period_key_) # Train Period 1-2-3
            plt.xlabel('Time')
            plt.ylabel('Wallet')
            
            x_train = list(train_period_wallets.index)
            y_benchmark_train = list(train_period_wallets["BTCUSDT"])
            plt.plot(x_train, y_benchmark_train, label='Benchmark TRAIN')
            for column_name in train_period_wallets.columns.tolist():
                y_pair = train_period_wallets[column_name]
                pair_name = column_name
                pair_name = pair_name.split("(")[-1]
                pair_name = pair_name.split(")")[0]
                plt.plot(x_train, y_pair, label= pair_name + " TRAIN" )
                
            x_test = list(test_period_wallets.index)
            y_benchmark_test = list(test_period_wallets["BTCUSDT"])
            plt.plot(x_test, y_benchmark_test, label='Benchmark TEST')
            for column_name in test_period_wallets.columns.tolist():
                y_pair = test_period_wallets[column_name]
                pair_name = column_name
                pair_name = pair_name.split("(")[-1]
                pair_name = pair_name.split(")")[0]  
                plt.plot(x_test, y_pair, label= pair_name + " TEST" )      
            plt.axvline(x=x_test[0], color='black', linestyle='--', label='Deploy to Test')
            plt.legend()
            plt.show()
    
        return None