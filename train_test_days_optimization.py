import optuna
import pandas as pd
import numpy as np
from functools import partial
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your stat_arb module
from Statistical_Arbitrage_Trading import Statistical_Arbitrage  # Replace with your actual module name


def obj_value_from_walkforward_backtesting(results_df, selected_pairs_df_by_period):

    avg_test_wallet = results_df["Test_Wallet"].mean()
    print(avg_test_wallet)
    total_ratio =  avg_test_wallet/10000
    print(total_ratio)
    test_days = selected_pairs_df_by_period["Test Days"][0]
    print(test_days)
    avg_return_by_day = total_ratio ** (1/test_days)
    print(avg_return_by_day)
    daily_return = (avg_return_by_day - 1) * 100  # Convert to percentage
    print(daily_return)
    return daily_return


def objective(trial, timeframe, start_time, end_time ):
    """
    Objective function for Optuna to optimize train_days and test_days parameters,
    with the constraint that test_days cannot exceed 50% of train_days.
    """

    price_matrix = pd.read_csv("binance_data/price_matrix_1d_close.csv", index_col='Close time')

    stat_arb = Statistical_Arbitrage(
                price_matrix=price_matrix,
                fee_rate=0.015,
                time_trend=True,
                quadratic_time_trend=False,
                max_pairs = 20, max_per_asset = 2
            )
    # First, suggest train_days
    train_days = trial.suggest_int('train_days', 30, 252)
    
    # Then, calculate the maximum allowed test_days (50% of train_days)
    max_test_days = train_days // 2  # Integer division to get the floor
    
    # Now suggest test_days with the dynamic upper bound
    # The lower bound is still 5, but the upper bound is now constrained
    test_days = trial.suggest_int('test_days', 5, min(63, max_test_days))
    
    logger.info(f"Trying parameters: train_days={train_days}, test_days={test_days}")
    
    try:
        # Rest of your function remains the same
        results_df, selected_pairs_df_by_period = stat_arb.Pair_Trading_Portfolio_Backtest_Time_Validation(
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            train_days=train_days,
            test_days=test_days,
            selected_pairs_df= []
        )

        # Extract the objective value to maximize
        objective_value = obj_value_from_walkforward_backtesting(results_df, selected_pairs_df_by_period)  
        
        logger.info(f"Obtained objective value: {objective_value}")
        return objective_value
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return float('-inf')
    




def optimize_parameters(timeframe, start_time, end_time, n_trials=100, study_name="stat_arb_optimization"):

    # Create a partial function with fixed parameters
    obj_func = partial(
        objective,
        timeframe=timeframe,
        start_time=start_time,  
        end_time=end_time
    )
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # We want to maximize the objective
    )
    
    # Run the optimization
    study.optimize(obj_func, n_trials=n_trials)
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best objective value: {best_value}")
    
    return best_params, study




def analyze_optimization_results(study):
    """
    Analyze and visualize the optimization results.
    """
    # Print optimization history
    print("\nOptimization History (top 10 trials):")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    for i, trial in enumerate(sorted_trials[:10]):
        print(f"Rank {i+1}:")
        print(f"  Trial {trial.number}")
        print(f"  Params: {trial.params}")
        print(f"  Value: {trial.value}")
    
    # Basic visualization
    try:
        import matplotlib.pyplot as plt
        
        # Extract the history data
        train_days_values = [t.params['train_days'] for t in study.trials if t.value is not None]
        test_days_values = [t.params['test_days'] for t in study.trials if t.value is not None]
        objective_values = [t.value for t in study.trials if t.value is not None]
        
        # Plot parameters vs objective
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(train_days_values, objective_values)
        plt.xlabel('train_days')
        plt.ylabel('Objective Value')
        plt.title('train_days vs. Objective')
        
        plt.subplot(1, 2, 2)
        plt.scatter(test_days_values, objective_values)
        plt.xlabel('test_days')
        plt.ylabel('Objective Value')
        plt.title('test_days vs. Objective')
        
        plt.tight_layout()
        plt.savefig('parameter_analysis.png')
        print("\nBasic visualization saved as 'parameter_analysis.png'")
        
    except ImportError:
        print("Matplotlib not installed. Skipping visualizations.")



if __name__ == "__main__":
    # Define your parameters

    price_matrix = pd.read_csv("binance_data/price_matrix_1d_close.csv", index_col='Close time')

    stat_arb = Statistical_Arbitrage(
                price_matrix=price_matrix,
                fee_rate=0.015,
                time_trend=True,
                quadratic_time_trend=False,
                max_pairs = 20, max_per_asset = 2
            )


    last_time = price_matrix.index[-1]
    day = 86400000
    TIMEFRAME = '_1D'
    START_TIME = last_time - day*300
    END_TIME = last_time - day*0

    
    # Run the optimization
    best_params, study = optimize_parameters(
        timeframe=TIMEFRAME,
        start_time=START_TIME,
        end_time=END_TIME,
        n_trials=3  # Start with fewer trials for initial testing
    )
    
    # Analyze the results
    analyze_optimization_results(study)
    
    # Use the best parameters to run your final backtest
    print("\nRunning final backtest with best parameters...")
    final_result = stat_arb.Pair_Trading_Portfolio_Backtest_Time_Validation(
        timeframe=TIMEFRAME,
        start_time=START_TIME,
        end_time=END_TIME,
        train_days=best_params['train_days'],
        test_days=best_params['test_days'],
        selected_pairs_df=[]
    )
    
    print(f"Final backtest result with optimal parameters: {final_result}")