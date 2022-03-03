from model.model import Executor

if __name__ == "__main__":         
        
    MSFT = Executor('MSFT','2010-10-01','2021-01-01')
    MSFT.summary_stats()
    MSFT.initial_plot('Adj Close')
    MSFT.window_selection([11*365,5*365,2*365,180,90],'Adj Close')
    MSFT.strategy_preparation('ADBE','Adj Close',180, 90)
    MSFT.backtesting(100000,1000,'Adj Close')
    MSFT.neural_enhancement('Adj Close',0.5,0.2,'relu','adam','mse',50,32)
    MSFT.enhanced_strategy(100000,1000,'ADBE','Adj Close',365, 180)