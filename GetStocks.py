import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import code
import pyesg as esg
from scipy.stats import norm
import pyesg as esg
def load_data(stockSymbol='AAPL',start_date='2015-01-02',end_date='2019-12-31')->pd.DataFrame:
    def download_data()->pd.DataFrame:
        nonlocal stockSymbol,start_date,end_date
        data = yf.download(stockSymbol, start_date, end_date)
        with open('stocks.csv','a',encoding='utf-8'):
            data.to_csv('stocks.csv',mode='a')
        data_reset = data.reset_index() 
        data_reset.columns=data_reset.columns.get_level_values(0)
        data.columns=data.columns.get_level_values(0)
        print(data_reset.head())
        data_reset.to_csv(stockSymbol+'.csv', index=False)
        return data
    try:
        data = pd.read_csv(stockSymbol+'.csv', index_col='Date')
    except FileNotFoundError:
        data = download_data(stockSymbol, start_date, end_date)
    return data
data=load_data('AAPL','2015-01-01','2019-12-31')
print(data.head())