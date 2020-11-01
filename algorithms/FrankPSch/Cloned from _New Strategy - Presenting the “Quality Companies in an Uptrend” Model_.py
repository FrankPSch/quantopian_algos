import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters import Q500US

def initialize(context):
    
    set_slippage(slippage.FixedSlippage(spread = 0.0)) 
    algo.attach_pipeline(make_pipeline(), 'pipeline')    
    
    #Schedule Functions
    schedule_function(trade, date_rules.month_end() , time_rules.market_close(minutes=30))
    schedule_function(trade_bonds, date_rules.month_end(), time_rules.market_close(minutes=20))
    
    #This is for the trend following filter
    context.spy = sid(8554)
    context.TF_filter = False
    context.TF_lookback = 126
    
    #Set number of securities to buy and bonds fund (when we are out of stocks)
    context.Target_securities_to_buy = 20.0
    context.bonds = sid(23870)
    
    #Other parameters
    context.top_n_roe_to_buy = 50 #First sort by ROE
    context.relative_momentum_lookback = 126 #Momentum lookback
    context.momentum_skip_days = 10
    context.top_n_relative_momentum_to_buy = 20 #Number to buy
    
 
def make_pipeline():

    # Base universe set to the Q500US
    universe = Q500US()

    roe = Fundamentals.roe.latest

    pipe = Pipeline(columns={'roe': roe},screen=universe)
    return pipe

def before_trading_start(context, data):
    
    context.output = algo.pipeline_output('pipeline')
    context.security_list = context.output.index
        
def trade(context, data):

    ############Trend Following Regime Filter############
    TF_hist = data.history(context.spy , "close", 140, "1d")
    TF_check = TF_hist.pct_change(context.TF_lookback).iloc[-1]

    if TF_check > 0.0:
        context.TF_filter = True
    else:
        context.TF_filter = False
    ############Trend Following Regime Filter End############
    
    #DataFrame of Prices for our 500 stocks
    prices = data.history(context.security_list,"close", 180, "1d")      
    #DF here is the output of our pipeline, contains 500 rows (for 500 stocks) and one column - ROE
    df = context.output  
    
    #Grab top 50 stocks with best ROE
    top_n_roe = df['roe'].nlargest(context.top_n_roe_to_buy)
    #Calculate the momentum of our top ROE stocks   
    quality_momentum = prices[top_n_roe.index][:-context.momentum_skip_days].pct_change(context.relative_momentum_lookback).iloc[-1]
    #Grab stocks with best momentum    
    top_n_by_momentum = quality_momentum.nlargest(context.top_n_relative_momentum_to_buy)
           
    for x in context.portfolio.positions:
        if (x.sid == context.bonds):
            pass
        elif x not in top_n_by_momentum:
            order_target_percent(x, 0)
            #print('GETTING OUT OF',x)

    for x in top_n_by_momentum.index:
        if x not in context.portfolio.positions and context.TF_filter==True:
            order_target_percent(x, (1.0 / context.Target_securities_to_buy))
            #print('GETTING IN',x)


def trade_bonds(context , data):
    amount_of_current_positions=0
    if context.portfolio.positions[context.bonds].amount == 0:
        amount_of_current_positions = len(context.portfolio.positions)
    if context.portfolio.positions[context.bonds].amount > 0:
        amount_of_current_positions = len(context.portfolio.positions) - 1
    percent_bonds_to_buy = (context.Target_securities_to_buy - amount_of_current_positions) * (1.0 / context.Target_securities_to_buy)
    order_target_percent(context.bonds , percent_bonds_to_buy)