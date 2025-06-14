from Statistical_Arbitrage_Trading import *


price_matrix = pd.read_csv("binance_data/price_matrix_1d_close.csv", index_col='Close time')

stat_arb = Statistical_Arbitrage(
            price_matrix=price_matrix,
            fee_rate=0.0,
            time_trend=False,
            quadratic_time_trend=False,
            max_pairs = 50, max_per_asset = 2
        )

"""last_time = price_matrix.index[-1]
day = 86400000
timeframe = '_1D'
start_time = last_time - day*100
end_time = last_time - day*0
train_days = 40
test_days = 10"""

"""

smooth_best_params, smooth_best_value, smooth_search_result = stat_arb.smooth_grid_search(search_result, window_size=1)
stat_arb.plot_grid_search_results(smooth_search_result)
stat_arb.Trade_Pair_Arbitrage( start_time, end_time,  "FISUSDT", "CVXUSDT", *smooth_best_params, True)"""
"""results_df, selected_pairs_df_by_period = stat_arb.Pair_Trading_Portfolio_Backtest_Time_Validation( timeframe, start_time, end_time, train_days, test_days, selected_pairs_df= [])
print(results_df, selected_pairs_df_by_period )"""
"""results_df.to_csv("results_df.csv")
selected_pairs_df_by_period.to_csv("selected_pairs_df_by_period.csv")"""
"""results_df = pd.read_csv("results_df_1H.csv")"""

selected_pairs_df_by_period = pd.read_csv("08_06_2025_runs_for_the_project/selected_pairs_by_period.csv")

train_wallet_dfs_by_periods, test_wallet_dfs_by_periods  = stat_arb.Create_Historical_Wallets( selected_pairs_df_by_period)

print(test_wallet_dfs_by_periods.keys())
for period in test_wallet_dfs_by_periods.keys():
    period_df = test_wallet_dfs_by_periods[period]
    print(period_df["Overall Test Wallet"].head(5))
    print(period_df["Overall Test Wallet"].tail(5))
    period_df.to_csv(str(period) + "_period_df.csv")
    portfolio_metrics = calculate_portfolio_metrics(period_df["Overall Test Wallet"], '_1D')
    print(period)
    print(portfolio_metrics)
