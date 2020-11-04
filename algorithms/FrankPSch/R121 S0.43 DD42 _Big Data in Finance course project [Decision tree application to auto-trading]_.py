# Use the previous 10 bars' movements to predict the next movement.

# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import deque
import numpy as np
import pandas as pd
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar as mstar
import talib
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import SimpleMovingAverage, RSI, MovingAverageConvergenceDivergenceSignal, ExponentialWeightedMovingAverage, BollingerBands, Returns
from quantopian.pipeline.data import Fundamentals
from scipy import optimize as opt

def initialize(context):
    
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    context.recent_prices = {}
    context.window_length = 5 # Amount of prior bars to study
    context.spy = deque(maxlen=context.window_length+1)
    
    #context.classifier = RandomForestClassifier() # Use a random forest classifier
    context.classifier = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10)
    # my training data
    #context.features = ['RSI','EMA','MACD','SMA_5','SMA_10','bb_lower','bb_middle','bb_upper', 'diluted_eps','growth_score','tangible_bv']
    context.features = ['RSI','EMA','MACD','SMA_5','SMA_10','bb_lower','bb_middle','bb_upper']
    context.response = ['Class']
    context.X = pd.DataFrame(columns = context.features) # Independent, or input variables
    context.Y = pd.DataFrame(columns = context.response) # Dependent, or output variable
    
    context.prediction = {} # Stores most recent prediction
    
    context.count = 0
    context.activate = False
    context.total_buy = 0
    context.position = []
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=10))
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
    set_benchmark(symbol('QQQ'))

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def make_pipeline():
    #context.features = ['RSI', 'MACD', 'EMA','SMA_5','SMA_10','ADX']
    base_universe = Q1500US()
    sector = mstar.asset_classification.morningstar_sector_code.latest
    sectors_311 = sector.eq(311)
    returns_5 = Returns(window_length=5)
    rsi = RSI(inputs=[USEquityPricing.close])
    macd = MovingAverageConvergenceDivergenceSignal(
        mask=base_universe
    )
    ema = ExponentialWeightedMovingAverage(
        mask=base_universe,
        inputs=[USEquityPricing.close],
        window_length=30,
        decay_rate=(1 - (2.0 / (1 + 15.0)))
    )
    mean_5 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=5,
        mask=base_universe
    )
    mean_10 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=10,
        mask=base_universe
    )
    bb = BollingerBands(
        inputs=[USEquityPricing.close],
        window_length=20,
        k=2
    )
    diluted_eps = Fundamentals.diluted_eps_earnings_reports.latest
    growth_score = Fundamentals.growth_score.latest
    tangible_bv = Fundamentals.tangible_book_value.latest
    return Pipeline(
        columns={
            'returns_5': returns_5,
            'RSI': rsi,
            'MACD': macd,
            'EMA': ema,
            'SMA_5': mean_5,
            'SMA_10': mean_10,
            'bb_upper': bb.upper,
            'bb_middle': bb.middle,
            'bb_lower': bb.lower,
            'diluted_eps': diluted_eps,
            'growth_score': growth_score,
            'tangible_bv': tangible_bv
        },
        screen=(base_universe & sectors_311),
    )
    
    
def before_trading_start(context, data):
    # Store our pipeline output DataFrame in context
    context.output = pipeline_output('my_pipeline')
    context.output = context.output.rename_axis(['Name'])
    
def rebalance(context, data):
    for single_security in context.output.index.tolist():
        if not context.output.loc[single_security].isnull().values.any():
            if single_security in context.recent_prices:
                context.recent_prices[single_security] = context.recent_prices[single_security].append(context.output.loc[single_security], ignore_index=True)
            else:
                context.recent_prices[single_security] = pd.DataFrame(context.output.loc[single_security]).transpose()  
        else:
            context.output = context.output.drop([single_security])
    context.spy.append(data.current(symbol('SPY'), 'price'))
    
    if context.activate:
        context.count = context.count + 1
    
    if len(context.spy) == context.window_length+1: # If there's enough recent price data
        
        benchmark = (context.spy[-1]-context.spy[0])/context.spy[0]
        pred = {}
        # add the price 5 days ago into X and Y, do for everyday
        for single_security in context.output.index.tolist():
            
            if single_security in context.output.index and len(context.recent_prices[single_security].index) == context.window_length+2:
                
                returns_5 = context.recent_prices[single_security].iloc[-1]['returns_5']
                change = returns_5 > benchmark and returns_5 > 0
                
                row = {}
                row['RSI'] = context.recent_prices[single_security].iloc[0]['RSI']
                row['EMA'] = context.recent_prices[single_security].iloc[0]['EMA']
                row['MACD'] = context.recent_prices[single_security].iloc[0]['MACD']
                row['SMA_5'] = context.recent_prices[single_security].iloc[0]['SMA_5']
                row['SMA_10'] = context.recent_prices[single_security].iloc[0]['SMA_10']
                row['bb_lower'] = context.recent_prices[single_security].iloc[0]['bb_lower']
                row['bb_middle'] = context.recent_prices[single_security].iloc[0]['bb_middle']
                row['bb_upper'] = context.recent_prices[single_security].iloc[0]['bb_upper']
                #row['diluted_eps'] = context.recent_prices[single_security].iloc[1]['diluted_eps']
                #row['growth_score'] = context.recent_prices[single_security].iloc[1]['growth_score']
                #row['tangible_bv'] = context.recent_prices[single_security].iloc[1]['tangible_bv']
            
                res = {}
                res['Class'] = change
                
                #if len(context.Y.index) > 5000:
                    #context.X = context.X.tail(1000)
                    #context.Y = context.Y.tail(1000)
                context.X = context.X.append([row], ignore_index = True)
                context.Y = context.Y.append([res], ignore_index = True)
            
                pred_row = {}
                pred_row['RSI'] = context.recent_prices[single_security].iloc[-1]['RSI']
                pred_row['EMA'] = context.recent_prices[single_security].iloc[-1]['EMA']
                pred_row['MACD'] = context.recent_prices[single_security].iloc[-1]['MACD']
                pred_row['SMA_5'] = context.recent_prices[single_security].iloc[-1]['SMA_5']
                pred_row['SMA_10'] = context.recent_prices[single_security].iloc[-1]['SMA_10']
                pred_row['bb_lower'] = context.recent_prices[single_security].iloc[-1]['bb_lower']
                pred_row['bb_middle'] = context.recent_prices[single_security].iloc[-1]['bb_middle']
                pred_row['bb_upper'] = context.recent_prices[single_security].iloc[-1]['bb_upper']
                #pred_row['diluted_eps'] = context.recent_prices[single_security].iloc[-1]['diluted_eps']
                #pred_row['growth_score'] = context.recent_prices[single_security].iloc[-1]['growth_score']
                #pred_row['tangible_bv'] = context.recent_prices[single_security].iloc[-1]['tangible_bv']
            
                pred[single_security] = pred_row
                
                #remove first row of the recent_prices
                context.recent_prices[single_security] = context.recent_prices[single_security].iloc[1:]
            
        if len(context.Y.index) >= 500 and context.count%5 == 0: # There needs to be enough data points to make a good model and adjust positions once per 5 days
            #if context.count <= 1:
            context.activate = True
            
            context.classifier.fit(context.X[context.features].values, context.Y[context.response].values) # Generate the model
            
            context.total_buy = 0
            for single_security in context.output.index.tolist():
                if single_security in pred:
                    tempdata = pd.DataFrame(columns = context.features)
                    tempdata = tempdata.append(pred[single_security], ignore_index=True)
                    context.prediction[single_security] = context.classifier.predict(tempdata) # Predict
                    context.total_buy += context.prediction[single_security][0]
            
            # If prediction = 1, buy all shares affordable, if 0 sell all shares
            
            #if (context.total_buy != 0):
                #for single_security in context.prediction:
                    #if context.prediction[single_security] > 0:
                        #pos.append(single_security)
                
            pos = []
            #if (context.total_buy != 0):
            for single_security in context.output.index.tolist():
                if single_security in context.prediction:
                    if context.prediction[single_security] > 0:
                        pos.append(single_security)
                    else:
                        order_target_percent(single_security, 0)
            context.position = data.history(pos, 'close', 30, '1d')
            if context.total_buy != 0:
                weight = opt_portfolio(context.position)
                for single_security in weight.keys():
                    order_target_percent(single_security, weight[single_security])

def record_vars(context, data):
    #for single_security in context.security:
    record(no_of_positions=context.total_buy)

def opt_portfolio(x):
        # opt using scipy
            # mean variance opt
    size = len(x.columns)
    dat = x
    returns = dat.pct_change()
    #print(returns.head())
    mean_return = np.array(returns.mean())
    annualized_return = np.round(mean_return * 252.0,2)

    stdev_return = np.array(returns.std())
    #annualized_std = np.round(stdev_return * np.sqrt(252), 2)

    cov_matrix = np.multiply(returns.cov(),252.0)
    def portfolio_return(x):
        return np.array(np.dot(x.T, annualized_return))
    def portfolio_std(x):
        return np.array(np.sqrt(np.dot(np.dot(x.T, cov_matrix), x)))
    def portfolio_var(x):
        return np.array((np.dot(np.dot(x.T, cov_matrix), x)))
    def target(x):
        return np.array(-1*(0.1*portfolio_return(x) - portfolio_var(x)))
    def target_jac(x):
        return np.array(-1*(annualized_return - 2* np.dot(cov_matrix, np.array(x))))
    def target_constraint(x):
        return sum(x) - 1
    
        # optimize
    init = np.random.random(size)
    init = init / sum(init)
    bd = tuple([(0.05,1)]*size)
    r1 = opt.minimize(target,init, bounds=bd)
    # r1 = opt.minimize(target,init)
    r1.x = r1.x/ sum(abs(r1.x))
    weight = {}
    for i in range(size):
        weight[x.columns[i]] = r1.x[i]
    return weight