"""
Author: Giacomo Rognoni
Title: Algorithmic trading strategies comparison

Summary: 
A comparison of various trading strategies to identify the most
effective and attempt enhancement through Neural Networks. The trading
strategies considered are:
    1) Moving Average Crossover
    2) Dual Moving Average Crossover
    3) Exponential Moving Average Crossover
    4) Mean Reversion
    5) Pairs Trading
    
The Neural Network used for enhancement includes: 
    1) 1 input layer, 1 hidden layer, 1 output layer
    2) Relu activation function for the first two layers
    3) Adam optimisation function
    4) MSE error function

"""

# Start my importing all the necessary packages

import pandas as pd
import pandas_datareader as pdr
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import random
import string
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from IPython import get_ipython
from tensorflow import keras
from keras import models as mod, layers
from keras.utils.vis_utils import plot_model
import os

# Disable tensorflow information messages

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class Executor:
    
    """
    Class is used to run full end to end process:
        Creation of trading strategies
        Backtesting of trading strategies
        Enhancement of most effective trading strategy
        Backtesting of enhanced trading strategy
    """
    
    NOTHING = pd.DataFrame(columns =['dummy','columns'])
    
    def __init__(self, name,startdate,enddate):
        
        """
        Get the ticker information for the chosen stock:
            Name
            Start date
            End date
            
        Generate initial variables required for analysis
        
        Input
        
        Name: Ticker name
        Start date: Ticker start date
        End date: Ticker end date
        
        """
    
        self.name = name
        
        self.start_date = startdate
        
        self.end_date = enddate
        
        self.data = pdr.get_data_yahoo(self.name,start=self.start_date,end=self.end_date)
        
        self.strategy_names = ['Moving Average Crossover strategy','Dual Moving Average Crossover strategy',
                               'Exponential Moving Average Crossover strategy', 'Mean Reversion', 'Pairs Trading']
    
        self.years = len(np.unique(pd.date_range(self.start_date,self.end_date).year))
        
        plt.rcParams["figure.figsize"]=(10, 10)
    
    def summary_stats(self):
        
        """
        Produce summary statistics for ticker data
        """
        
        self.stats = self.data.describe()
        print(self.stats)
    
    def initial_plot(self,p):
        
        """
        Produce initial plots for ticker data
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        """
        
        self.data[p]
        
        # Initialize the plot figure
        fig = plt.figure()
        
        # Add a subplot and label for y-axis
        ax1 = fig.add_subplot(111,  ylabel='Price in $', title=str(self.name)+' Price v Time (Years)')
        
        # Plot the closing price
        
        self.data[p].plot(ax=ax1, color='red', lw=1.)
                 
        # Show the plot

        plt.show()
        
    def window_selection(self,window_list,p):
        
        """
        Analyse window selection
        
        Input
        
        window_list: list of windows which will be compared (int)
        p: Chosen ticker attribute (e.g. adjusted price)
        
        """
        
        self.running_window = pd.DataFrame(index = self.data.index)
        
        self.running_window['Actual'] = self.data[p]
        
        for i in window_list:
            
            self.running_window['window_'+str(i)] = self.data[p].rolling(window = i, center = False,min_periods = 1).mean()

        print(self.running_window)
        
        fig = plt.figure()

        ax1 = fig.add_subplot(111,  ylabel='Price in $', title=str(self.name)+' Window Size Selection')
        
        self.running_window['Actual'].plot(ax=ax1, color='red', lw=2.,label='Actual')
        
        for i in window_list:
            
            r = random.random()
            b = random.random()
            g = random.random()
            
            c = (r, g, b)
            
            self.running_window['window_'+str(i)].plot(ax=ax1, color=c, lw=2., label='window_'+str(i))
        
        ax1.legend()
        
        plt.show()
    
    def MAC_strategy(self,p,L_window,dataset = NOTHING):
        
        """
        Prepare Moving Average Crossover strategy
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: window size
        dataset (optional): dataset used to prepare the strategy
        """
        
        if dataset.empty:
            
            dataset = self.data
        
        self.MAC = pd.DataFrame(index = dataset.index)
        
        self.MAC['AVG'] = dataset[p].rolling(window = L_window,min_periods=1,center=False).mean()
        
        self.MAC['signal'] = np.where(self.MAC['AVG']<dataset[p],1,0)

        self.MAC['position'] = self.MAC['signal'].diff()
    
    def DMAC_strategy(self,p,L_window,S_window=1,dataset = NOTHING):
        
        """
        Prepare Dual Moving Average Crossover strategy
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: Long window size
        S_window: Short window size
        dataset (optional): dataset used to prepare the strategy
        """
        
        if dataset.empty:
            dataset = self.data
        
        self.DMAC = pd.DataFrame(index = dataset.index)
    
        self.DMAC['LAVG'] = dataset[p].rolling(window = L_window,min_periods=1,center=False).mean()
        
        self.DMAC['AVG'] = dataset[p].rolling(window = S_window,min_periods=1,center=False).mean()
        
        self.DMAC['signal'] = np.where(self.DMAC['LAVG']<self.DMAC['AVG'],1,0)
        
        self.DMAC['position'] = self.DMAC['signal'].diff()
        
    def EMAC_strategy(self,p,L_window,dataset = NOTHING):
        
        """
        Prepare Enhanced Moving Average Crossover strategy
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: window size
        dataset (optional): dataset used to prepare the strategy
        """
        
        if dataset.empty:
            dataset = self.data
        
        self.EMAC = pd.DataFrame(index = dataset.index)
        
        self.EMAC['AVG'] = dataset[p].ewm(span = L_window).mean()
        
        self.EMAC['signal'] = np.where(self.EMAC['AVG']<dataset[p],1,0)

        self.EMAC['position'] = self.EMAC['signal'].diff()
    
    def MR_strategy(self,p,L_window,dataset = NOTHING):
        
        """
        Prepare Mean Regression
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: window size
        dataset (optional): dataset used to prepare the strategy
        """
        
        if dataset.empty:
            dataset = self.data
        
        self.MR = pd.DataFrame(index = dataset.index)
        
        self.MR['AVG'] = dataset[p].rolling(window = L_window,min_periods=1,center=False).mean()
        
        self.MR['STDEV'] = dataset[p].rolling(window = L_window,min_periods=1,center=False).std()
        
        self.MR['Z_SCORE'] = (dataset[p] - self.MR['AVG'])/self.MR['STDEV']
        
        self.MR['signal'] = np.where(self.MR['Z_SCORE']>=1,1,0)

        self.MR['position'] = self.MR['signal'].diff()
        
    def PT_strategy(self,p,L_window,pair_stock,dataset = NOTHING):
        
        """
        Prepare Pairs Trading strategy
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: window size
        pair_stock: The pair stock chosen for pairs trading
        dataset (optional): dataset used to prepare the strategy
        """
        
        if dataset.empty:
            
            dataset = self.data
        
        else:

            self.start_date = dataset.index.min()
            
            self.end_date = dataset.index.max()
        
        self.PT = pd.DataFrame(index = dataset.index)
        
        self.coint_data = pdr.get_data_yahoo(pair_stock,start=self.start_date,end=self.end_date)
        
        _, p_value, _ = coint(dataset[p],self.coint_data[p])
        
        self.p_value = p_value

        if self.p_value <=0.05:
            
            self.PT['DIFF'] = self.coint_data[p] - dataset[p]
            
            self.PT['AVG'] = self.PT['DIFF'].rolling(window = L_window,min_periods=1,center=False).mean()
        
            self.PT['STDEV'] = self.PT['DIFF'].rolling(window = L_window,min_periods=1,center=False).std()
            
            self.PT['Z_SCORE'] = (self.PT['DIFF'] - self.PT['AVG'])/self.PT['STDEV']
            
            self.PT['signal'] = np.where(self.PT['Z_SCORE']>=1,1,0)
    
            self.PT['position'] = self.PT['signal'].diff()
            
        else:
            
            self.PT['AVG'] = 1
            
            self.PT['position'] = 1
            
            self.PT['STDEV'] = 1
            
            self.PT['signal'] = 1
            
            print("ERROR: no cointegration between chosen stocks, select different stocks!")
        
    def strategy_preparation(self,pair_stock,p,L_window,S_window=1):
        
        """
        Initialise trading strategies
        
        Input
        
        pair_stock: The pair stock chosen for pairs trading        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: Long window size
        S_window (default = 1): Short window size
        dataset (optional): dataset used to prepare the strategy
        """
        
        NOTHING = pd.DataFrame(columns =['dummy','columns'])
        
        self.MAC_strategy(p,L_window,dataset = NOTHING)
        
        self.DMAC_strategy(p,L_window,S_window,dataset = NOTHING)
        
        self.EMAC_strategy(p,L_window,dataset = NOTHING)
        
        self.MR_strategy(p,L_window,dataset = NOTHING)
        
        self.PT_strategy(p,L_window,pair_stock,dataset = NOTHING)
        
        self.dictionary = {}
        
        self.dictionary[0] = pd.DataFrame(self.MAC[['AVG','position','signal']])
        self.dictionary[1] = pd.DataFrame(self.DMAC[['AVG','LAVG','position','signal']])
        self.dictionary[2] = pd.DataFrame(self.EMAC[['AVG','position','signal']])
        self.dictionary[3] = pd.DataFrame(self.MR[['AVG','STDEV','position','signal']])
        self.dictionary[4] = pd.DataFrame(self.PT[['AVG','STDEV','position','signal']])
        
        fig = plt.figure()
        
        for i,j in enumerate(self.strategy_names):
        
            locals()["ax"+str(i+1)] = fig.add_subplot(int('32'+str(i+1)),  ylabel=str(p)+' price in $', title=j)
            
            fig.subplots_adjust(hspace=0.65,wspace = 0.4)
            
            self.data[p].plot(ax = locals()["ax"+str(i+1)],color ='red',lw=2.,label='Actual')
            
            self.dictionary[i].loc[:,(self.dictionary[i].columns != 'position') & (self.dictionary[i].columns !='signal')].plot(ax = locals()["ax"+str(i+1)],lw=2.,label=j)
            
            locals()["ax"+str(i+1)].plot(self.dictionary[i].loc[self.dictionary[i].position == 1.0].index, 
            self.dictionary[i].AVG[self.dictionary[i].position == 1.0],
            '^', markersize=10, color='m')
            
            locals()["ax"+str(i+1)].plot(self.dictionary[i].loc[self.dictionary[i].position== -1.0].index, 
            self.dictionary[i].AVG[self.dictionary[i].position == -1.0],
            'v', markersize=10, color='k')

        plt.show()
        
    def backtesting(self,capital, order,p):
        
        """
        Run backtesting on initialised strategies
        
        Input
        
        capital: Initial capital for backtesting (int)
        order: order size (int)
        p: Chosen ticker attribute (e.g. adjusted price)
        """
        
        colnames = ['Strategy','Portfolio value ($)','Max returns ($)','Min returns ($)','Sharpe ratio']
        
        fig = plt.figure()        
        
        for i,j in enumerate(self.strategy_names):
            
            positions = pd.DataFrame(index=self.data.index).fillna(0.0)
        
            positions['Order'] = order*self.dictionary[i]['signal']
            
            portfolio = positions.multiply(self.data[p],axis = 0)
            
            pos_diff = positions.diff()
            
            portfolio['holdings'] = (positions.multiply(self.data[p],axis=0)).sum(axis=1)
            
            portfolio['cash'] = capital - (pos_diff.multiply(self.data[p], axis=0)).sum(axis=1).cumsum()
            
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            
            portfolio['returns'] = portfolio['total'].pct_change()
            
            sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
                
            if i == 0:
                
                self.returns = pd.DataFrame([[j,portfolio.total.iat[-1],max(portfolio['total']),min(portfolio['total']),
                                        sharpe_ratio]],columns = colnames)
            
            else:
                
                temp_df = pd.DataFrame([[j,portfolio.total.iat[-1],max(portfolio['total']),min(portfolio['total']),
                                        sharpe_ratio]],columns = colnames)
                
                self.returns = pd.concat([self.returns, temp_df],axis=0,ignore_index=True)
            
            locals()["ax"+str(i+1)] = fig.add_subplot(int('32'+str(i+1)),  ylabel='Portfolio value ($ ''000s)', title=j)
            
            fig.subplots_adjust(hspace=0.65,wspace = 0.4)
            
            portfolio['total'].div(1000).plot(ax = locals()["ax"+str(i+1)],lw=2.,label='Portfolio value')
            
            locals()["ax"+str(i+1)].plot(portfolio.loc[self.dictionary[i].position == 1.0].index, 
            portfolio.total.div(1000)[self.dictionary[i].position == 1.0],
            '^', markersize=10, color='m')
            
            locals()["ax"+str(i+1)].plot(portfolio.loc[self.dictionary[i].position== -1.0].index, 
            portfolio.total.div(1000)[self.dictionary[i].position == -1.0],
            'v', markersize=10, color='k')
        
        plt.show()
        
        print(self.returns)
                                 
        print('The strategy with the highest portfolio value is: '
              +str(self.returns.iloc[self.returns['Portfolio value ($)'].argmax(),0]))
        
        print('The strategy with the highest Sharpe ratio is: '
              +str(self.returns.iloc[self.returns['Sharpe ratio'].argmax(),0]))
        
        self.top_strategy = self.returns.iloc[self.returns['Portfolio value ($)'].argmax(),0]
    
    def neural_enhancement(self, p,train, test, activation_func, optimiser_func, loss_func, epochs, batch):
        
        """
        Use neural network for strategy enhancement
        
        Input
        
        p: Chosen ticker attribute (e.g. adjusted price)
        train: Choose train dataset size (0-1)
        test: Choose test dataset size (0-1)
        activation_func: Choose activation function for all layers
        optimiser_fun: Choose optimiser function
        loss_func: Choose loss function
        epochs: Choose number of epochs
        batch: Choose number of batches
        """
        
        self.forecast_data = pd.DataFrame(index = self.data.index)
        
        self.forecast_data['lagged Adj Close'] = self.data['Adj Close'].shift(1)
        
        self.forecast_data['lagged Adj Close'].fillna((self.forecast_data['lagged Adj Close'].min()),inplace=True)
        
        X = self.data.iloc[:,0:4].merge(self.data['Adj Close'],left_on = 'Date', right_on = 'Date')
        
        y = self.forecast_data['lagged Adj Close']
        
        # Avoid using train test split to avoid losing index
        
        X_train = X.iloc[:int(train*len(X['Adj Close'])),:]
        
        y_train = y[:int(train*len(y))]
        
        X_test = X.iloc[int(train*len(X['Adj Close'])):int((train+test)*len(X['Adj Close'])),:]
        
        y_test = y[int(train*len(y)):int((train+test)*len(y))]
        
        X_pred = X.iloc[int((train+test)*len(X['Adj Close'])):,:]
        
        y_pred = y[int((train+test)*len(y)):]
        
        self.neural_model = mod.Sequential()

        self.neural_model.add(layers.Dense(70, activation = 'relu', input_shape = (5,)))
        
        self.neural_model.add(layers.Dense(35, activation = 'relu'))
        
        self.neural_model.add(layers.Dense(1))
        
        print(self.neural_model.summary())
        
        self.model_info = {}

        print('Training')
        
        self.neural_model.compile(optimizer='adam', loss='mse') 
    
        h = self.neural_model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch,
                  validation_data=(X_test, y_test))
    
        self.model_info[1] = {'model': self.neural_model, 'history': h.history}
        
        y_neural_pred = self.neural_model.predict(X_pred)
        
        self.FS = pd.DataFrame(index = X_pred.index)
        
        self.FS['Actual'] = y_pred
        
        self.FS[p] = y_neural_pred
        
        fig = plt.figure()
        
        ax1 = fig.add_subplot(111,  ylabel='Price in $', title = 'Predicted v Actual values for predicted period')
        
        self.FS['Actual'].plot(ax=ax1, color='r', lw=2.,label='True Values')
        
        self.FS[p].plot(ax=ax1, color = 'g', lw=2., label='Predicted Values')
        
        plt.show()
        
    def enhanced_backtesting(self,top_strategy,p,capital,order):
        
        """
        Run backtesting on enhanced strategy and most effective strategy
        
        Input
        
        top_strategy: The most effective strategy identified (automatically populated)
        p: Chosen ticker attribute (e.g. adjusted price)
        capital: Initial capital for backtesting (int)
        order: order size (int)
        """
        
        colnames = ['Strategy','Portfolio value ($)','Max returns ($)','Min returns ($)','Sharpe ratio']
        
        fig = plt.figure()        
            
        positions = pd.DataFrame(index=self.FS.index).fillna(0.0)
            
        positions['Order'] = order*top_strategy['signal']
        
        portfolio = positions.multiply(self.FS[p],axis = 0)
        
        pos_diff = positions.diff()
        
        portfolio['holdings'] = (positions.multiply(self.FS[p],axis=0)).sum(axis=1)
        
        portfolio['cash'] = capital - (pos_diff.multiply(self.FS[p], axis=0)).sum(axis=1).cumsum()
        
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        portfolio['returns'] = portfolio['total'].pct_change()
        
        sharpe_ratio = np.sqrt(252) * (portfolio['returns'].mean() / portfolio['returns'].std())
        
        positions_standard = pd.DataFrame(index=self.FS.index).fillna(0.0)
            
        positions_standard['Order'] = order*top_strategy['signal']
        
        portfolio_standard = positions_standard.multiply(self.FS[p],axis = 0)
        
        pos_diff_standard = positions_standard.diff()
        
        portfolio_standard['holdings'] = (positions_standard.multiply(self.FS['Actual'],axis=0)).sum(axis=1)
        
        portfolio_standard['cash'] = capital - (pos_diff_standard.multiply(self.FS['Actual'], axis=0)).sum(axis=1).cumsum()
        
        portfolio_standard['total'] = portfolio_standard['cash'] + portfolio_standard['holdings']
        
        portfolio_standard['returns'] = portfolio_standard['total'].pct_change()
        
        sharpe_ratio_standard = np.sqrt(252) * (portfolio_standard['returns'].mean() / portfolio_standard['returns'].std())
                
        self.enhanced_returns = pd.DataFrame([["Enhanced "+str(self.top_strategy),portfolio.total.iat[-1],max(portfolio['total']),min(portfolio['total']),
                                        sharpe_ratio]],columns = colnames)
        
        temp_df = pd.DataFrame([[str(self.top_strategy),portfolio_standard.total.iat[-1],max(portfolio_standard['total']),min(portfolio_standard['total']),
                                        sharpe_ratio_standard]],columns = colnames)
                
        self.enhanced_returns = pd.concat([self.enhanced_returns, temp_df],axis=0,ignore_index=True)

        ax1 = fig.add_subplot(211,  ylabel='Portfolio value ($)', title='Enhanced '+str(self.top_strategy)+' returns')
        
        ax2 = fig.add_subplot(212,  ylabel='Portfolio value ($)', title=str(self.top_strategy)+' returns')
        
        fig.subplots_adjust(hspace=0.5)
        
        portfolio['total'].plot(ax = ax1,lw=2.,label='Portfolio value')
            
        ax1.plot(portfolio.loc[top_strategy.position == 1.0].index, 
        portfolio.total[top_strategy.position == 1.0],
        '^', markersize=10, color='m')
        
        ax1.plot(portfolio.loc[top_strategy.position== -1.0].index, 
        portfolio.total[top_strategy.position == -1.0],
        'v', markersize=10, color='k')
        
        portfolio_standard['total'].plot(ax = ax2,lw=2.,label='Portfolio value')
            
        ax2.plot(portfolio_standard.loc[top_strategy.position == 1.0].index, 
        portfolio_standard.total[top_strategy.position == 1.0],
        '^', markersize=10, color='m')
        
        ax2.plot(portfolio_standard.loc[top_strategy.position== -1.0].index, 
        portfolio_standard.total[top_strategy.position == -1.0],
        'v', markersize=10, color='k')
        
        plt.show()
        
        if self.enhanced_returns.iloc[0]['Portfolio value ($)']<self.enhanced_returns.iloc[1]['Portfolio value ($)']:
            
            message = 'non enhanced strategy'
        
        else:
            
            message = 'enhanced strategy'
        
        print('The enhanced strategy resulted in a final portfolio value of: '
              +str(self.enhanced_returns.iloc[0]['Portfolio value ($)']))
        
        print('The enhanced strategy resulted in a final Sharpe ratio of: '
              +str(self.enhanced_returns.iloc[0]['Sharpe ratio']))
        
        print('The most effective strategy is the '+message)
        
        
    def enhanced_strategy(self,capital,order,pair_stock,p,L_window,S_window=1):
        
        """
        Initialise most effective trading strategy with neural network enhancement and run backtesting
        
        Input
        
        capital: Initial capital for backtesting (int)
        order: order size (int)
        pair_stock: The pair stock chosen for pairs trading        
        p: Chosen ticker attribute (e.g. adjusted price)
        L_window: Long window size
        S_window (default = 1): Short window size
        """
        
        if self.strategy_names.index(self.top_strategy) == 0:
            
            self.MAC_strategy(p,L_window,self.FS)
            
            self.enhanced_backtesting(self.MAC,p,capital,order)
        
        elif self.strategy_names.index(self.top_strategy) == 1:
            
            self.DMAC_strategy(p,L_window,S_window,self.FS)
            
            self.enhanced_backtesting(self.DMAC,p,capital,order)
        
        elif self.strategy_names.index(self.top_strategy) == 2:
            
            self.EMAC_strategy(p,L_window,self.FS)
            
            self.enhanced_backtesting(self.EMAC,p,capital,order)
        
        elif self.strategy_names.index(self.top_strategy) == 3:
            
            self.MR_strategy(p,L_window,self.FS)
            
            self.enhanced_backtesting(self.MR,p,capital,order)
            
        elif self.strategy_names.index(self.top_strategy) == 4:
        
            self.PT_strategy(p,L_window,pair_stock,self.FS)
            
            self.enhanced_backtesting(self.PT,p,capital,order)

    
  





