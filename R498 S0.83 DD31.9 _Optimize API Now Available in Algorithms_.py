import numpy as np
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.classifiers.morningstar import Sector
import pandas as pd
import quantopian.optimize as opt
from scipy.stats import rankdata

# Constraint Parameters
MAX_GROSS_LEVERAGE = 2.0        #was 1.5
MAX_SHORT_POSITION_SIZE = 0.03  #was 0.015
MAX_LONG_POSITION_SIZE = 0.035   #was 0.015
MAX_TURNOVER = 1.50              #was 0.75

def initialize(context):
    
    # Slippage
    #https://www.quantopian.com/help#ide-slippage
    #default: VolumeShareSlippage with volume_limit=0.025, price_impact=0.1
    #you can take up to 2.5% of a minute's trade volume, slippage is calculated by multiplying price impact with (order vol / total vol)^2
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) 
    
    # Commission
    #context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0)) # default: $0.001 per share with a $1 minimum cost per trade
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # Close approximation for IB retail customers
    
    # parameters
    # --------------------------
    context.n_stocks = 200 # universe size, top market cap
    context.N = 4 # trailing window size, days
    # --------------------------
    
    schedule_function(get_factor, date_rules.week_start(days_offset=1), time_rules.market_close(minutes=5))
    
    schedule_function(allocate, date_rules.week_start(days_offset=1), time_rules.market_close(minutes=5))
     
    attach_pipeline(make_pipeline(context), 'my_pipe')
    
def make_pipeline(context):
    
    profitable = mstar.valuation_ratios.ev_to_ebitda.latest > 0
    market_cap = mstar.valuation.market_cap.latest
    
    my_screen = market_cap.top(context.n_stocks, mask = (Q1500US() & profitable))
    
    return Pipeline(columns={
            'sector': Sector(),
        },screen = my_screen)
    
def before_trading_start(context,data):

    context.output = pipeline_output('my_pipe')
    
    context.stocks = context.output.index.tolist()
               
    record(leverage = context.account.leverage)
    
    num_secs = 0
    b_t = np.zeros(len(context.portfolio.positions.keys()))
    for i,stock in enumerate(context.portfolio.positions.keys()):
        if context.portfolio.positions[stock].amount != 0:
            num_secs += 1
        b_t[i] = context.portfolio.positions[stock].amount*data.current(stock,'price')
            
    record(num_secs = num_secs)
    
    denom = np.sum(np.absolute(b_t))
    if denom > 0:
        ls_sum = np.sum(b_t)/denom
    else:
        ls_sum = 0
        
    record(ls_sum = ls_sum)
    
def get_factor(context,data):
    
    prices = data.history(context.stocks, 'price', 390*context.N, '1m').dropna(axis=1)
    context.stocks = list(prices.columns.values)
    prices = prices.as_matrix(context.stocks)
    
    a = np.zeros(len(context.stocks))
    w = 0
        
    for n in range(1,13*context.N+1):
        (a,w) = mean_rev(context,data,prices[-n*30:,:])
        a += w*a
        w += w
    
    a = a/w
    a = rankdata(a)
    a = a - np.mean(a)
                   
    denom = np.sum(np.absolute(a))
    if denom > 0:
        a = a/denom
    
    context.weight = a
    
def allocate(context, data):
    
    port = context.stocks + list(set(context.portfolio.positions.keys()) - set(context.stocks))
    w = np.zeros(len(port))
    for i,stock in enumerate(port):
        w[i] = context.portfolio.positions[stock].amount*data.current(stock,'price')
    
    denom = np.sum(np.absolute(w))
    if denom > 0:
        w = w/denom
    
    current_portfolio = pd.Series(w,index=port)
    
    pipeline_data = pd.DataFrame({'alpha': context.weight},index = context.stocks)
    df_pipeline = context.output.ix[context.stocks]
    pipeline_data = pipeline_data.join(df_pipeline,how='inner')
    pipeline_data = pipeline_data.loc[data.can_trade(context.stocks)]
    
    objective = opt.MaximizeAlpha(pipeline_data.alpha)
 
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    market_neutral = opt.DollarNeutral()
    
    sector_neutral = opt.NetGroupExposure.with_equal_bounds(
        labels=pipeline_data.sector,
        min=-0.0001,
        max=0.0001,
    )
     
    constrain_turnover = opt.MaxTurnover(MAX_TURNOVER)
    
    constraints=[
                constrain_gross_leverage,
                constrain_pos_size,
                market_neutral,
                sector_neutral,
                constrain_turnover
                ]
    
    try:
        
        weights = opt.calculate_optimal_portfolio(objective, constraints, current_portfolio)
        
    except:
        print "oops"
        return
    
    for (stock,weight) in weights.iteritems():
        if data.can_trade(stock):
            order_target_percent(stock,weight*3.0)
    
def mean_rev(context,data,prices):
    
    m = len(context.stocks)
    d = np.ones(m)
    
    x_tilde = np.mean(prices,axis=0)/prices[-1,:]
    y_tilde = 1.0/x_tilde
    
    d[x_tilde < 1] = -1
    d[x_tilde < 0.9] = 1
    d[x_tilde > 1.1] = -1
    
    x_tilde[x_tilde < 1] = 0
    y_tilde[x_tilde != 0] = 0
    
    x_tilde = x_tilde + y_tilde

    return (d*x_tilde, np.sum(x_tilde)/m)