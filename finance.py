import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import code
import pyesg as esg
from scipy.stats import norm
import pyesg as esg
# Step 1: 下载数据

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
# Step 2: 计算对数收益率
def calculate_log_returns(data:pd.DataFrame,calcDataColumn='Adj Close')->pd.Series:
    '''
    计算数据指定列的对数变化率（收益率）
    '''
    data['Log_Returns'] = np.log((data[calcDataColumn]-data[calcDataColumn].shift(1)) / data[calcDataColumn].shift(1))
    log_returns = data['Log_Returns'].dropna()
    log_returns.name='log returns'
    return log_returns

# Step 3: 定义负对数似然函数
def negative_log_likelihood(params, returns):
    """
    计算对数收益率的负对数似然函数
    """
    mu, sigma = params
    n = len(returns)
    log_likelihood = -(-n / 2 * np.log(2 * np.pi)
                       - n / 2 * np.log(sigma**2)
                       - np.sum((returns - mu)**2) / (2 * sigma**2))
    return log_likelihood
# Step 4: 最大似然估计优化
def estimate_parameters(log_returns:pd.Series):
    '''
    给定series，假设其满足正态分布，极大似然估计法求均值和标准差
    '''
    initial_guess = [0.001, 0.02]  # 初始猜测年化收益率和波动率
    result = minimize(negative_log_likelihood, initial_guess, args=(log_returns,), bounds=[(-1, 1), (1e-5, 1)])
    mu_mle, sigma_mle = result.x
    mu=mu_mle
    sigma=sigma_mle
    print(f'Estimate Parameters：{mu} {sigma}')
    return mu,sigma
def estimate_parameters_direct(log_returns:pd.Series):
    '''
    给定series，假设其满足正态分布，直接根据样本均值和方差求其均值和标准差
    '''
    mu=log_returns.mean()
    sigma=np.sqrt(sum((log_returns-mu)**2)/(len(log_returns)-1))
    return mu,sigma
# Step 5: 验证拟合效果并绘制图形
def predict_stock_price(S0, mu, sigma, TimeRange:pd.Index, num_simulations=5):
    TimeRangedate=pd.to_datetime(TimeRange)
    T = (TimeRangedate[-1] - TimeRangedate[0]).days
    n = len(TimeRange)  # 交易日数量
    simulated_prices = np.zeros((n, num_simulations))
    ts=[(TimeRangedate[i]-TimeRangedate[0]).days for i in range((n))]
    dt=[ts[i]-ts[max(0,i-1)] for i in range(n)]
    for i in range(num_simulations):
        # 生成标准正态随机数（模拟Wiener过程）
        random_walk = [np.random.normal(0, 1, n)[i] * np.sqrt(dt[i]) for i in range(n)]
        Bt=np.cumsum(random_walk)
        # 计算每日股价
        sumdXt=[mu*t for t in ts]
        sumdXt=[a+sigma*b for a,b in zip(sumdXt,np.cumsum(random_walk))]
        S_path = S0 * np.exp(sumdXt)
        # 将模拟的股价路径存储
        simulated_prices[:, i] = S_path
    # 将模拟结果转换为DataFrame，TimeRange为索引，列名为 Simulation 1, Simulation 2, ...
    df_simulated_prices = pd.DataFrame(simulated_prices, index=TimeRange, columns=[f'Simulation {i+1}' for i in range(num_simulations)])
    
    return df_simulated_prices
def plotStockPrices(ind:0):    
    def plot_stock_prices(pricesWithDate:pd.DataFrame,label='Simulation'):
        #plt.figure(figsize=(6,10))
        nonlocal ind
        pricesWithDate.plot(ax=plt.gca(),label=label+str(ind))
        ind+=1
    return plot_stock_prices
def pltset(title='Simulation of Stocks-AAPL',x='Date',y='Stock Price',grid=True,size=(6,10),stock=None):
    if stock is not None:
        plt.title('Simulation of Stocks - '+stock)
    else:
        plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.grid(grid)
    plt.tight_layout()
    plt.show()

def calcS_with_logreturns(s0,logreturns:pd.Series)->pd.DataFrame:
    S=[]
    for i in range(len(logreturns)):
        ret=np.exp(logreturns[i])
        if i>0:
            S.append(S[-1]*ret)
        else:
            S.append(s0*ret)
    S=pd.DataFrame(S,index=logreturns.index,columns=['StockPrice'])
    return S
def normSeries(index:pd.Index,mu=0,sigma=1,name='normSeires',wtexist=False)->pd.Series:
    n=len(index)
    norms=[norm.rvs(mu,sigma) for _ in range(n)]
    if wtexist is True:
        wt=np.random.normal(0,1,n)
        norms=[a+b for a,b in zip(norms,wt)]
    norms=pd.Series(norms,index=index,name=name)
    return norms
def main():
    #set properties & load data 
    targetStock='MSFT'
    fittarget='Adj Close'
    data=load_data(targetStock,'2015-01-01','2019-12-31')
    print(data.head())
    dateindex=data[fittarget].index
    
    #calculate log_returns (also change rate)
    log_returns = calculate_log_returns(data)

    #estimate parameters
    mu_mle, sigma_mle = estimate_parameters(log_returns)

    #calulate S(t)
    s0=data[fittarget].values[0]
    data[fittarget].plot(label='Real Stokes')
    S=predict_stock_price(s0,mu_mle,sigma_mle/2,dateindex,5)
    S.plot(label='fit with wt',ax=plt.gca())
    pltset(stock=targetStock)
if __name__ == "__main__":
    main()
