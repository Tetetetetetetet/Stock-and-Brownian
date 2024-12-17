import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Tuple,Callable
import pyesg as esg
import code
import os
#Parameters
generateTimes=100
ticker='MSFT'
startDate='2015-01-01'
endDate='2019-12-31'
fitTarget='Adj Close'
demoTrain=False
demoResult=False
trainNum=20

class StockSimulation:
    def __init__(self, stock_symbol='AAPL', start_date='2015-01-02', end_date='2019-12-31',fittarget='Adj Close'):
        self.demoTrain=demoTrain
        self.demoResult=demoResult
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None  # 原始数据
        self.data_partition=-1
        self.train_data=None
        self.val_data=None
        self.fitted_values = None  # 模拟预测的拟合值
        self.fittarget=fittarget
        self.daterange=None
        self.mu=None
        self.sigma=None
        self.fitS=None
        self.bestseed=None
        self.predict_mse=None
        self.train_mse=None
        self.overall_mse=None
        self.check_and_create_dirs()
    def check_and_create_dirs(self):
        """检查并创建 FitStocks 和 img 文件夹"""
        folders = ['FitStocks', 'img']  # 文件夹列表
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"文件夹 '{folder}' 不存在，已创建。")
            else:
                print(f"文件夹 '{folder}' 已存在。")
    def load_data(self):
        """
        下载股票数据
        get data from file of yfinance; set train_data to data
        """
        try:
            self.data = pd.read_csv(self.stock_symbol + '.csv', index_col='Date')
        except FileNotFoundError:
            self.data = yf.download(self.stock_symbol, self.start_date, self.end_date)
            self.data.to_csv('stocks.csv','a',encoding='utf-8')
            self.data.columns=self.data.columns.get_level_values(0)
            data_reset=self.data.reset_index()
            data_reset.to_csv(self.stock_symbol+'.csv')
        print(f'get data: {self.data.head()}')
        self.train_data=self.data
        return self.data

    def calculate_log_returns(self) -> pd.Series:
        """[train_data[i]-train_data[i-1])/train_data[i-1], dropna(), merged into train_data['Log Return']"""
        calc_column=self.fittarget
        self.train_data['Log Returns'] = np.log((self.train_data[calc_column]-self.train_data[calc_column].shift(1)) / self.train_data[calc_column].shift(1))
        log_returns = self.train_data['Log Returns'].dropna()
        return log_returns

    def negative_log_likelihood(self, params, returns):
        """
        负对数似然函数,正态分布
        -Loss func of {mu, sigma} in Normal
        """
        mu, sigma = params
        n = len(returns)
        log_likelihood = -(-n / 2 * np.log(2 * np.pi)
                           - n / 2 * np.log(sigma**2)
                           - np.sum((returns - mu)**2) / (2 * sigma**2))
        return log_likelihood

    def estimate_parameters(self) -> Tuple[float, float]:
        """
        极大似然估计参数mu, sigma
        MLE estimated {mu,sigma} on train_data['Log Returns']
        """
        log_returns=self.train_data['Log Returns']
        initial_guess = [0.001, 0.02]
        result = minimize(self.negative_log_likelihood, initial_guess, args=(log_returns,), bounds=[(-1, 1), (1e-5, 1)])
        mu, sigma = result.x
        self.mu=mu
        self.sigma=sigma
        print(f'estimated mu: {mu}, sigma: {sigma}')
    def try_simultation(self,randomseed):
        '''
        找出拟合值高的随机数种子
        find highly fitted randomseed
        '''
    def MSE(self,realdata:np.array,fitdata:np.array):
        if realdata.shape != fitdata.shape:
            print(f'realdata: {realdata.shape},fitdata: {fitdata.shape}')
            raise ValueError("输入的两个 DataFrame 的形状必须相同")
        error = realdata - fitdata
        squared_error = error ** 2
        mse = squared_error.mean()
        return mse
    def split_data(self,percent=0.8):
        '''
        划分数据集->val_data开始坐标
        set train_data,val_data,(->)data_partition
        '''
        self.data_partition=int(len(self.data)*percent)
        self.train_data=self.data.iloc[:self.data_partition]
        self.val_data=self.data.iloc[self.data_partition:]
        return self.data_partition
    def predict_stock_price(self, time_range:pd.Index,num_simulations=5,setrandseed=None,setself=False):
        """
        通过模型mu,sigma在给定时间范围上模拟， 给出MSE
        simulate with {mu,sigma} on time_range ; calc MSE
        draw fitS & realS
        """
        mu=self.mu
        sigma=self.sigma
        s0=self.data[self.fittarget].values[0]
        n=len(time_range)
        time_range_date=pd.to_datetime(time_range)
        T=(time_range_date[-1]-time_range_date[0]).days
        ts=[(time_range_date[i]-time_range_date[0]).days for i in range((n))]
        dt=[ts[i]-ts[max(0,i-1)] for i in range(n)]
        geo=esg.GeometricBrownianMotion(mu,sigma)
        S_paths=np.zeros((n,num_simulations))
        if setrandseed is None:
            randseeds=np.random.randint(100,1000,num_simulations)
        else:
            randseeds=[setrandseed]
        print(f'randseeds: {randseeds}')
        mses={}
        for i in range(num_simulations):
            randseed=randseeds[i]
            newPath=geo.scenarios(s0,T/n,n_scenarios=1,n_steps=n-1,random_state=int(randseed))
            mse=self.MSE(self.data.loc[time_range,self.fittarget],newPath.T[:,0])
            S_paths[:,i]=newPath.T[:,0]
            mses[randseeds[i]]=mse
        bestseed=min(mses,key=mses.get)
        print(f'mses:{mses}\n best randseed: {bestseed}, min MSE: {mses[bestseed]}')
        S_paths=pd.DataFrame(S_paths,index=time_range,columns=[f'randseed: {randseeds[i]}' for i in range(num_simulations)])
        if setself is False:
            self.train_mse=mses[bestseed]
            if self.demoTrain is True:
                S_paths.plot(ax=plt.gca())
                self.data[self.fittarget].plot()
                self.pltset(title='Train Process')
        if setself is True:
            self.fitS=S_paths.iloc[:,0]
            self.bestseed=bestseed
            self.overall_mse=mses[bestseed]
        return bestseed,S_paths[f'randseed: {bestseed}']

    def simulate_on_data(self,times):
        #train model
        self.split_data()
        self.calculate_log_returns()
        self.estimate_parameters()
        bestseed,S=self.predict_stock_price(self.train_data[self.fittarget].index,num_simulations=times)
        #predict
        self.predict_stock_price(self.data[self.fittarget].index,num_simulations=1,setrandseed=bestseed,setself=True)
       # code.interact(local=locals())
        self.predict_mse=self.MSE(self.val_data[self.fittarget].values,self.fitS.iloc[self.data_partition:].values)
        self.fitS.name=f'Best Fit seed={self.bestseed} OM={self.overall_mse} PM={self.predict_mse} TM={self.train_mse} '
    def pltset(self,title='Simulation of Stocks-AAPL',x='Date',y='Stock Price',grid=True,size=(6,10),save=False,show=True):
        if self.stock_symbol is not None:
            plt.title('Simulation of Stocks - '+self.stock_symbol)
        else:
            plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.grid(grid)
        plt.tight_layout()
        plt.savefig(r'img\MSFT'+f'{self.bestseed}.png')
        plt.gca().relim()  # 重新计算数据范围
        plt.gca().autoscale_view()  # 自动缩放视图

        if show is True:
            plt.show()
    def run(self):
            """程序的主函数"""
            # 加载数据
            self.load_data()
            self.split_data()
            # 训练预测
            self.simulate_on_data(times=trainNum)
            if self.predict_mse<400:
                #绘图
                plt.figure(figsize=(10,6))
                self.data.index=pd.to_datetime(self.data.index)
                self.fitS.index=pd.to_datetime(self.fitS.index)
                self.data[self.fittarget].plot(ax=plt.gca())
                self.fitS.plot(ax=plt.gca())
                split_date = self.data.index[self.data_partition]  # 划分点对应的日期
                plt.axvline(x=split_date,color='r',linestyle='-',linewidth=1)
                print(f'Data Partition at {split_date}')
                result={'Random Seed':self.bestseed,'Overall MSE':self.overall_mse,'Predict MSE:':self.predict_mse,'Train MSE':self.train_mse}
                result_df=pd.DataFrame(result,index=[0])
                file_exists = os.path.isfile('record.csv')  # 检查文件是否存在
                with open('record.csv', 'a', encoding='utf-8', newline='') as f:
                    result_df.to_csv(f, index=False, header=not file_exists, mode='a')
                self.fitS.to_csv(path_or_buf=os.path.join('FitStocks',f'{self.stock_symbol}{self.bestseed}'+'.csv'))
                self.pltset(save=True,show=self.demoResult)

            
def main():
    global generateTimes,ticker,startDate,endDate,fitTarget
    for i in range(generateTimes):
        ss = StockSimulation(stock_symbol=ticker, start_date=startDate, end_date=endDate,fittarget=fitTarget)
        ss.run()


if __name__ == "__main__":
    main()