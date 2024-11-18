import scipy
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

### 데이터 다운로드 ###
def download_data(symbols, start_date, end_date):
    stocks = pd.DataFrame()
    for symbol in symbols:
        stocks[symbol] = yf.download(symbol, start=start_date, end=end_date)['Close']
    return stocks

### 로그 수익률 계산 ###
def calculate_log_returns(stocks):
    return (np.log(stocks) - np.log(stocks.shift(1))).dropna()

### 초기 변수 설정 ###
def initialize_parameters(stocks_returns):
    n = len(stocks_returns.columns)
    weights = [1 / n] * n
    weights /= np.sum(weights)
    cov_var = stocks_returns.cov()
    stock_cov = np.diag(cov_var)
    stock_sigma = np.sqrt(stock_cov)
    return weights, cov_var, stock_sigma

### 상관 관계와 난수 생성 ###
def generate_correlated_random_numbers(stocks_returns, T, days, nsimulation):
    L = spicy.linalg.cholesky(np.corrcoef(stocks_returns.T), lower=True)
    Normal = np.random.normal(size=(len(stocks_returns.columns), T * days, nsimulation))
    dz = np.einsum('ij,jkl->ikl', L, Normal)
    return dz

### 로그 수익률 계산 ###
def simulate_log_returns(dz, r, dt, stock_sigma):
    logds = []
    for i in range(len(stock_sigma)):
        logds.append(r * dt + dz[i] * stock_sigma[i])
    return logds

### 가격 경로 시뮬레이션 ###
def simulate_price_paths(logds, T, days, nsimulation, initial_prices):
    paths = []
    for i in range(len(initial_prices)):
        path = np.zeros((T * days + 1, nsimulation))
        path[0] = initial_prices[i]
        for t in range(1, T * days + 1):
            path[t] = path[t - 1] * (1 + logds[i][t - 1])
        paths.append(pd.DataFrame(path))
    return paths

### Monte Carlo VaR 계산하기 ###
def MC_VaR(sim_return, initial_investment, conf_level):
    MC_percentile = []
    for i, j in zip(sim_return.columns, range(len(sim_return.columns))):
        MC_percentile.append(np.percentile(sim_return.loc[:, i], (1 - conf_level) * 100))
        print("Based on simulation {:.2f}% of {}'s return is {:.4f}".format(conf_level * 100, i, MC_percentile[j]))
        VaR_MC = initial_investment - initial_investment * (1 + MC_percentile[j])
        print("Simulation VaR result for {} is {:.2f} ".format(i, VaR_MC))
        print('--' * 35)

### Main Function ###
def main():
    # 사용자 입력
    symbols = input("Enter stock symbols separated by space: ").split()
    start_date = input("Enter start date (YYYY-MM-DD): ")
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=365)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    initial_investment = float(input("Enter initial investment amount: "))
    conf_level = float(input("Enter confidence level (e.g., 0.95 for 95%): "))

    # 데이터 다운로드 및 로그 수익률 계산
    stocks = download_data(symbols, start_date, end_date)
    stocks_returns = calculate_log_returns(stocks)

    # 초기 변수 설정
    weights, cov_var, stock_sigma = initialize_parameters(stocks_returns)

    # 시뮬레이션 설정
    T = 1
    days = 250
    nsimulation = 300000
    r = 0.01
    dt = 1 / days

    # 상관 관계와 난수 생성
    dz = generate_correlated_random_numbers(stocks_returns, T, days, nsimulation)

    # 로그 수익률 시뮬레이션
    logds = simulate_log_returns(dz, r, dt, stock_sigma)

    # 초기 가격 설정
    initial_prices = stocks.iloc[-1].values

    # 가격 경로 시뮬레이션
    paths = simulate_price_paths(logds, T, days, nsimulation, initial_prices)

    # 로그 수익률 계산 및 결합
    sim_returns = pd.concat([
        pd.Series((np.log(path) - np.log(path.shift(1))).dropna().values.flatten()) for path in paths
    ], axis=1)
    sim_returns.columns = stocks.columns

    # Monte Carlo VaR 계산
    MC_VaR(sim_returns, initial_investment, conf_level)

if __name__ == "__main__":
    main()