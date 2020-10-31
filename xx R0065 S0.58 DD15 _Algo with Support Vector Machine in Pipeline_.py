from quantopian.pipeline import Pipeline, CustomFilter, CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
from sklearn.preprocessing import StandardScaler
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import Q500US
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd


class Prediction(CustomFactor):
    def compute(self, today, asset_ids, out, bull_msgs, bear_msgs, close_prices, open_prices):
        predictions = []
        for i in range(close_prices.shape[1]):
            bull_msg = bull_msgs[:, i]
            bear_msg = bear_msgs[:, i]
            result = (close_prices[:, i] > open_prices[:, i]) * 2 - 1
            df = pd.DataFrame(data={'bull':bull_msg.flatten(), 'bear': bear_msg.flatten(), \
                                    'result': result.flatten()})
            
            # before shifting, we must record the last values as they will be used
            # to run the model on for the prediction.
            df['bull_ma'] = df['bull'].rolling(window=10).mean() #.shift(1)
            df['bear_ma'] = df['bear'].rolling(window=10).mean() #.shift(1)
            df['bull_rs'] = df['bull'].rolling(window=5).apply(compute_slope) #.shift(1)
            df['bear_rs'] = df['bear'].rolling(window=5).apply(compute_slope) #.shift(1)

            scaler = StandardScaler()
            try:
                features = ['bull_rs', 'bear_rs']
                X_live = df[features][-1:]
                df[features] = df[features].shift(1)
                df.dropna(inplace=True)            
                X_train = scaler.fit_transform(df[features])
                y_train = df['result']

                model = GaussianNB()
                prediction = model.fit(X_train, y_train).predict(scaler.transform(X_live))[0]

                predictions.append(prediction)
            except:
                predictions.append(1)
        
        out[:] = predictions
       
    
def custom_pipeline(context):
    sma_10 = SimpleMovingAverage(inputs = [USEquityPricing.close], window_length=10)
    sma_50 = SimpleMovingAverage(inputs = [USEquityPricing.close], window_length=50)  
    
    # for testing only
    small_universe = SidInList(sid_list = (24))
        
    #changed to be easier to read.
    my_screen = (Q500US() & \
                    (sma_10 > sma_50) & \
                    (st.bull_scored_messages.latest > 10)) 
    
    prediction = Prediction(inputs=[st.bull_scored_messages, st.bear_scored_messages, \
                                             USEquityPricing.close, USEquityPricing.open],\
                                     window_length=200, mask=small_universe)
       
    return Pipeline(
        columns={
            'sma10': sma_10,
            'close': USEquityPricing.close.latest,
            'prediction': prediction
        },
        screen=my_screen)


def initialize(context):
    
    #ADDED TO MONITOR LEVERAGE MINUTELY.
    context.minLeverage = [0]
    context.maxLeverage = [0]
    
    attach_pipeline(custom_pipeline(context), 'custom_pipeline') 
        
    schedule_function(evaluate, date_rules.every_day(), time_rules.market_open(minutes=1))
    schedule_function(sell, date_rules.every_day(), time_rules.market_open())
    schedule_function(buy, date_rules.every_day(), time_rules.market_open(minutes = 5))    

    context.longs = []
    context.shorts = []
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


def before_trading_start(context, data):
    context.results = pipeline_output('custom_pipeline')
        
        
class SidInList(CustomFilter):
    """
    Filter returns True for any SID included in parameter tuple passed at creation.
    Usage: my_filter = SidInList(sid_list=(23911, 46631))
    """    
    inputs = []
    window_length = 1
    params = ('sid_list',)

    def compute(self, today, assets, out, sid_list):
        out[:] = np.in1d(assets, sid_list)
        
        
def compute_slope(a):
    x = np.arange(0, len(a))
    y = np.array(a)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m

                
def evaluate (context, data):    
    context.longs = []
    
    for sec in context.results.index:
        if context.results.loc[sec, 'prediction'] == 1:
            print "Here"
            if sec not in context.portfolio.positions:
                context.longs.append(sec)
            
            
def sell (context,data):
    for sec in context.portfolio.positions:
        if sec not in context.longs:
            order_target_percent(sec, 0.0)
            
            
def buy (context,data):
    for sec in context.longs:
        order_target_percent(sec, 1.0 / (len(context.longs) + len(context.portfolio.positions)))