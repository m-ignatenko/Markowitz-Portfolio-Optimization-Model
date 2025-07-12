import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns, efficient_frontier, risk_models,plotting,get_latest_prices,DiscreteAllocation
import streamlit as st
import seaborn as sns

from datetime import datetime, timedelta
today = datetime.now().date()
d2 = today - timedelta(days=2)
d1 = d2 - timedelta(days=365*2) 
sns.set_theme(
        style="darkgrid",
        palette="deep",
        font_scale=1.1,
        rc={
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.edgecolor': '0.15',
            'axes.linewidth': 1.25,
        }
    )


st.set_page_config(
        page_title="Markowitz Portfolio Model",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
plt.grid()
main_col, param_col = st.columns([4, 3])
with param_col:
    linkedin_url = "https://www.linkedin.com/in/mikhail-ignatenko-b79876243/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail Ignatenko`</a>', unsafe_allow_html=True)
    tg_link = "https://t.me/mikhail_lc"
    st.markdown(f'<a href="{tg_link}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/128/2111/2111646.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail I`</a>', unsafe_allow_html=True)
    github_link = "https://github.com/m-ignatenko"
    st.markdown(f'<a href="{github_link}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/128/14063/14063266.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Mikhail I`</a>', unsafe_allow_html=True)

    st.header("Parameters")
    capital = st.number_input("Capital", value=100000.0, step=100.0)
    a = st.number_input("Maximum acceptable risk (a)", value=0.500,min_value=0.000, max_value=1.000,step=0.0001, format="%.4f")
    st.header("Portfolio Tickers")
    default_tickers = ['SPY','HOOD','TSLA','JPM','V','CSCO','GS','MA','BX','BLK','AAPL','AXP','PLTR','NVDA','ADBE']
    user_tickers = st.text_area("Enter NASDAQ tickers (comma separated)", value=", ".join(default_tickers))
    tickers = [ticker.strip().upper() for ticker in user_tickers.split(",") if ticker.strip()]
    
    
with main_col:
    try:
        df = yf.download(tickers, start=d1, end=d2)['Close']
        df = df.dropna(axis=1, how='all')
        
        if len(df.columns) == 0:
            st.error("No valid tickers entered or all tickers have no data for the selected date range.")
            st.stop()
            
        mu = expected_returns.mean_historical_return(df)
        s = risk_models.sample_cov(df)
        plt.grid()
        plt.title("Efficient Frontier")
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        ef = efficient_frontier.EfficientFrontier(mu, s)
        risk = ef.efficient_risk(a)
        x = ef.portfolio_performance(verbose=False,risk_free_rate=0.0434)
        ret = x[0]
        vol = x[1]

        ef2 = efficient_frontier.EfficientFrontier(mu, s)
        ef2.max_sharpe()
        x2 = ef2.portfolio_performance(verbose=False,risk_free_rate=0.0434)
        ret_s = x2[0]
        vol_s = x2[1]

        ef = efficient_frontier.EfficientFrontier(mu, s)
        plotting.plot_efficient_frontier(ef)
        plt.plot(vol,ret, 'ro',label='Your portfolio')
        plt.plot(vol_s,ret_s, 'go',label ='Max Sharpe ratio portfolio')
        plt.legend()
        st.pyplot(plt)
        
        st.info('For each portfolio ticker set, there is a minimum volatility below which you cannot select a portfolio.')
        st.text(f'Selected portfolio performance: \n \t Volatility: {x[1]:.4f} \n \t Return :{x[0]:.4f} \n\t Sharpe ratio: {x[2]:.2f}')
        st.text(f'Max Sharpe ratio portfolio: \n \t Volatility: {x2[1]:.4f} \n \t Return :{x2[0]:.4f} \n\t Sharpe ratio: {x2[2]:.2f}')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        st.stop()
    with param_col:
        latest_prices = get_latest_prices(df)

        portfolio = DiscreteAllocation(risk, latest_prices,total_portfolio_value=capital).lp_portfolio()
        st.text("Calculated Stock Portfolio: ")
        for i in portfolio[0]:
            st.text(f"{i}: {portfolio[0][i]}")
        st.text(f"remainder: {portfolio[1]:.2f}")