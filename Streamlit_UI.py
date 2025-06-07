import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from Statistical_Arbitrage_Trading import Statistical_Arbitrage  # Your original class
from helper_funcs import *
st.set_page_config(layout="wide")
import datetime

def load_data():
    # Add file uploader for the price matrix
    uploaded_file = st.file_uploader("Upload your price matrix CSV file", type="csv")
    if uploaded_file is not None:
        price_matrix = pd.read_csv(uploaded_file)
        price_matrix = price_matrix.set_index("Close time")
        return price_matrix
    return None

def main():
    st.title("Statistical Arbitrage Trading System")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Load data
    price_matrix = load_data()

    
    if price_matrix is not None:
        # Parameters
        fee_rate = st.sidebar.slider("Fee Rate (%)", 0.0, 1.0, 0.1, 0.01)
        time_trend = st.sidebar.checkbox("Include Time Trend")
        quadratic_time_trend = st.sidebar.checkbox("Include Quadratic Time Trend")
        timeframe = get_timeframe_of_the_data(price_matrix)

        # Initialize Statistical Arbitrage
        stat_arb = Statistical_Arbitrage(
            price_matrix=price_matrix,
            fee_rate=fee_rate,
            time_trend=time_trend,
            quadratic_time_trend=quadratic_time_trend,
            max_pairs = 5, max_per_asset = 1 
        )
        
        # Tabs for different functionalities
        tab1, tab2, tab3= st.tabs([
            "Single Pair Trading", 
            "Pair Optimization",
            "Portfolio Backtesting"
        ])
        
        with tab1:
            st.header("Single Pair Trading")
            
            col1, col2 = st.columns(2)
            with col1:
                pair_1 = st.selectbox("Select First Asset", price_matrix.columns)
            with col2:
                pair_2 = st.selectbox("Select Second Asset", price_matrix.columns)
                
            col3, col4, col5, col6, col7 = st.columns(5)
            with col3:
                z_score_thresh = st.number_input("Z-Score Threshold", 1.0, 5.0, 1.5, 0.1)
            with col4:
                lookback_window = st.number_input("Lookback Window", 10, 200, 100, 10)
            with col5:
                show_graph = st.checkbox("Show Graphs", True)
            with col6: 
                start_time = st.number_input("Taking the end of the data's end, how many days ago should the start be", 10, 5000, 200, 10)
            with col7: 
                end_time = st.number_input("Taking the end of the data's end, how many days ago should the end be", 0, 5000, 200, 10)


            if st.button("Run Single Pair Analysis"):
                try:
                    last_time = price_matrix.index[-1]
                    day = 86400000
                    trades_df = stat_arb.Trade_Pair_Arbitrage(
                        start_time = last_time - day*start_time,
                        end_time = last_time - day*end_time,
                        pair_1=pair_1,
                        pair_2=pair_2,
                        z_score_thresh=z_score_thresh,
                        lookback_window=lookback_window,
                        show_on_graph=False
                    )
                    
                    # Display results using Plotly
                    if show_graph:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Z Score"],
                                            mode='lines', name='Z Score'))
                        fig1.update_layout(title="ZScore Over Time")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Wallet"],
                                            mode='lines', name='Strategy Wallet'))
                        fig2.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Hodling Wallet"],
                                            mode='lines', name='BTC Hodl'))
                        fig2.update_layout(title="Strategy Performance vs BTC Hold")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    final_return = ((trades_df["Wallet"].iloc[-1] - trades_df["Wallet"].iloc[0]) / 
                                  trades_df["Wallet"].iloc[0] * 100)
                    hodl_return = ((trades_df["Hodling Wallet"].iloc[-1] - trades_df["Hodling Wallet"].iloc[0]) / 
                                 trades_df["Hodling Wallet"].iloc[0] * 100)
                    
                    col1.metric("Strategy Return", f"{final_return:.2f}%")
                    col2.metric("HODL Return", f"{hodl_return:.2f}%")
                    col3.metric("Outperformance", f"{final_return - hodl_return:.2f}%")

                    #Display detailed KPI metrics
                    kpi_df = calculate_portfolio_metrics(trades_df["Wallet"], timeframe, 0.04) # assumption of trades_df being daily
                    st.subheader("KPIs")
                    st.dataframe(kpi_df)
                    
                except Exception as e:
                    st.error(f"Error in analysis: {str(e)}")
        
        with tab2:
            st.header("Pair Optimization")
            
            col1, col2, col3, col4 , col5 = st.columns(5)
            with col1:
                pair_1 = st.selectbox("Select First Asset to Optimize", price_matrix.columns)
            with col2:
                pair_2 = st.selectbox("Select Second Asset to Optimize", price_matrix.columns)
                
            with col3: 
                start_time = st.number_input("Taking the end of the data's end, how many days ago should the start be?", 10, 5000, 200, 10)
            with col4: 
                end_time = st.number_input("Taking the end of the data's end, how many days ago should the end be?", 0, 5000, 200, 10)
            with col5:
                show_graph = st.checkbox("Show Optimized Wallet", True)

            if st.button("Run Pair Optimization"):
                try:
                    last_time = price_matrix.index[-1]
                    day = 86400000
                    optimization_results = stat_arb.Find_Best_Parameters_Pair_Trading(
                        timeframe, 
                        start_time = last_time - day*start_time,
                        end_time = last_time - day*end_time,
                        pair_1=pair_1,
                        pair_2=pair_2,
                        risk_free_rate=0.05
                    )
                    
                    # Display key metrics
                    col1, col2= st.columns(2)

                    best_params = optimization_results[0]
                    best_value = optimization_results[1]
                    search_results = optimization_results[2]
                    
                    st.subheader("Optimal Parameters for the Pair:")
                    col1, col2, col3= st.columns(3)
                    col1.metric("Z-Score", best_params[0])
                    col2.metric("Lookback Window", best_params[1])
                    col3.metric("Best Parameters yield a Sharpe Ratio of: ", f"{best_value:.2f}")

                    #Display detailed Iterations
                    search_results = pd.DataFrame([search_results])
                    st.subheader("Iteration results")
                    st.dataframe(search_results)

                    #best_params = list(best_params)
                    trades_df = stat_arb.Trade_Pair_Arbitrage( last_time - day*start_time, last_time - day*end_time, pair_1, pair_2, *best_params, False)
                     # Display results using Plotly
                    if show_graph:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Z Score"],
                                            mode='lines', name='Z Score'))
                        fig1.update_layout(title="ZScore Over Time")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Wallet"],
                                            mode='lines', name='Strategy Wallet'))
                        #fig2.add_trace(go.Scatter(x=pd.to_datetime(trades_df.index.astype(int), unit='ms'), y=trades_df["Hodling Wallet"],
                        #                    mode='lines', name='BTC Hodl'))
                        fig2.update_layout(title="Strategy Performance ")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    final_return = ((trades_df["Wallet"].iloc[-1] - trades_df["Wallet"].iloc[0]) / 
                                  trades_df["Wallet"].iloc[0] * 100)
                    hodl_return = ((trades_df["Hodling Wallet"].iloc[-1] - trades_df["Hodling Wallet"].iloc[0]) / 
                                 trades_df["Hodling Wallet"].iloc[0] * 100)
                    
                    col1.metric("Strategy Return", f"{final_return:.2f}%")
                    col2.metric("HODL Return", f"{hodl_return:.2f}%")
                    col3.metric("Outperformance", f"{final_return - hodl_return:.2f}%")

                    #Display detailed KPI metrics
                    kpi_df = calculate_portfolio_metrics(trades_df["Wallet"], timeframe, 0.04) # assumption of trades_df being daily
                    st.subheader("KPIs")
                    st.dataframe(kpi_df)


                    
                except Exception as e:
                    st.error(f"Error in analysis: {str(e)}")



        with tab3:
            st.header("Portfolio Backtesting")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                start_time = st.number_input("How many days ago should the start be", 10, 5000, 200, 10)
            with col2: 
                end_time = st.number_input("How many days ago should the end be", 0, 5000, 200, 10)

            with col3:
                train_days = st.number_input("Training Days", 0, 500, 300)
            with col4:
                test_days = st.number_input("Testing Days", 0, 200, 100)
            #with col3:
            #    timeframe = st.selectbox("Timeframe", ["_1D", "_4H", "_1H"])
            
            if st.button("Run Portfolio Backtesting"):
                try:
                    with st.spinner("Running portfolio backtest..."):
                        last_time = price_matrix.index[-1]
                        day = 86400000
                        results_df, pairs_df = stat_arb.Pair_Trading_Portfolio_Backtest_Time_Validation(
                            timeframe=timeframe,
                            start_time=last_time - day*start_time,
                            end_time=last_time - day*end_time,
                            train_days=train_days,
                            test_days=test_days
                        )
                        
                        st.subheader("Backtesting Results")
                        st.dataframe(results_df)
                        
                        st.subheader("Selected Pairs by Period")
                        st.dataframe(pairs_df)
                        
                        # Calculate and display performance metrics
                        success_rate = (results_df["Beat the Market"].sum() / 
                                      len(results_df["Beat the Market"]) * 100)
                        
                        st.metric("Market Outperformance Rate", f"{success_rate:.1f}%")

                        
                except Exception as e:
                    st.error(f"Error in backtesting: {str(e)}")

        
       # with tab4:
            st.header("Historical Analysis")
            uploaded_file = st.file_uploader("Upload your selected pairs df resulted from the backtest", type="csv")
            pairs_df = None
            
            if uploaded_file is not None:
                pairs_df = pd.read_csv(uploaded_file)

            if st.button("Generate Historical Analysis"):
                if pairs_df is not None:
                    try:
                        with st.spinner("Generating historical analysis..."):
                            # Load or use existing portfolio results
                                                

                            
                            train_wallets, test_wallets = stat_arb.Create_Historical_Wallets(
                                pairs_df
                             #   start_time,
                             #   train_days,
                             #   test_days
                            )
                            
                            # Display interactive plots for each period
                            for period in train_wallets.keys():
                                st.subheader(f"{period}")
                                
                                fig = go.Figure()
                                train_df = train_wallets[period]
                                test_df = test_wallets[period]
                                
                                # Add traces for training period
                                for col in train_df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=pd.to_datetime(train_df.index.astype(int), unit='ms'),
                                        y=train_df[col],
                                        name=f"{col} (Train)",
                                        line=dict(dash='solid')
                                    ))
                                
                                # Add traces for testing period
                                for col in test_df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=pd.to_datetime(test_df.index.astype(int), unit='ms'),
                                        y=test_df[col],
                                        name=f"{col} (Test)",
                                        line=dict(dash='dot')
                                    ))
                                
                                fig.update_layout(
                                    title=f"Historical Performance - {period}",
                                    xaxis_title="Date",
                                    yaxis_title="Wallet Value"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error in historical analysis: {str(e)}")

if __name__ == "__main__":
    main()

# streamlit run streamlit_ui.py