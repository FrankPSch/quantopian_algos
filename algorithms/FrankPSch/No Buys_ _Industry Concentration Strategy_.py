"""

Trading strategy related to "Are US Industries Becoming More Concentrated" (Grullon, Larkin, Michaely)
    At the end of June each year, get trailing 12 month sales for each stock
    Compute the Herfindahl-Hirschman Index (HHI) for each industry (using NAICS industry classification)
    Sort industries by change in HHI from previous year, and buy top 10 industries and short bottom 10
    Form equally weighted portfolio of industries, and equally weight stocks within each industry
    Rebalance once/year

"""

import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.filters import Q500US, Q1500US
# These last two imports are for using Quantopian's Optimize API
from quantopian.algorithm import order_optimal_portfolio
import quantopian.optimize as opt

    
def initialize(context):
    
    # Set benchmark to short-term Treasury note ETF (SHY) since strategy is dollar neutral
    set_benchmark(sid(23911))
    
    # We run rebalance every month, but only rebalance in June
    schedule_function(my_rebalance, date_rules.month_end(), time_rules.market_open())

    # Record variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    # Set commissions and slippage to 0 to determine pure alpha
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


    # context.conc is a DataFrame, indexed by industry, of the old HHI, the new HHI, and 
    #   number of stocks in each industry
    context.conc=pd.DataFrame(columns=['old','new','count'])
    context.longs=[]
    context.shorts=[]

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
            
        
class Sales(CustomFactor):   
    inputs = [morningstar.income_statement.operating_revenue]  
    window_length = 189
    
    def compute(self, today, assets, out, sales):       
        out[:] = sales[0]+sales[-1]+sales[-64]+sales[-127] 
        
class Industry(CustomFactor):
    inputs=[morningstar.asset_classification.naics]
    window_length=1
    
    def compute(self, today, assets, out, industry):
        out[:] = industry[-1]   
        
        
def make_pipeline():
    """
    Create our pipeline.
    """
    
    # To make the program more flexible, we allow for the possibility of having two universes,
    #   one for estimating HHI and one for trading stocks after sorting by changes in HHI
    estimation_universe=Q500US()
    trading_universe=Q500US()
    
    pricing=USEquityPricing.close.latest
    # Note that if we don't filter for primary share class, we double count sales for dual class stocks
    primary_share = IsPrimaryShare(mask=estimation_universe)
    sales=Sales(mask=estimation_universe)
    industry=Industry(mask=estimation_universe)
    
    universe = (
          primary_share
          & (pricing > 5)
    )


    return Pipeline(
        screen=universe,
        columns= {
        'sales':sales,
        'industry':industry,
        'in_trading_univ':trading_universe
    }       
    )

def before_trading_start(context, data):
    # Gets our pipeline output every day.
    context.output = pipeline_output('my_pipeline')
   

    
def my_rebalance(context, data):

    # We only want to rebalance once/year, at the end of June, so we check if the month is June
    backtest_month=get_datetime().month
    if backtest_month != 6:
        return
    
    context.output=context.output.dropna()
    # There are a few stocks that have an industry code of -1, which we'll drop
    context.output=context.output[context.output['industry']!=-1]
    # We divide by 1000 to convert from 6 digit NAICS to 3 digit NAICS industry classification
    context.output['industry']=context.output['industry'] // 1000
    log.info('Number of unique industries: %d' %(context.output['industry'].nunique()))


    # 'share' is the  market share of total sales for each company
    context.output['share']=context.output.groupby('industry')['sales'].transform(lambda x: (x/x.sum()))
    # We square market share for HHI calculation below
    context.output['share']=context.output['share']**2

    print(context.output.head(30))
    
    # 'new' is the current HHI for each industry, computed as the sum of squared market shares
    context.conc['new']=context.output.groupby('industry')['share'].sum()
    
    # Check whether it's first time running by looking at whether the 'old' HHI hasn't been created yet
    if pd.isnull(context.conc['old']).all():
       context.conc['old']=context.conc['new']
       return
    
    # 'count' is the numer of stocks in each industry (that are in the trading universe)
    context.conc['count']=context.output[context.output['in_trading_univ']].groupby('industry')['sales'].count()
    
    # If the trading unviverse is smaller than the estimation universe, there may be some industries with
    #    no stocks to trade.  We eliminate those industries.
    context.conc=context.conc[context.conc['count'] != 0]
    context.conc=context.conc.dropna()
    
    # Compute change in HHI
    context.conc['change']=context.conc['new']/context.conc['old']
    print(context.conc.head(30))
    
    # Sort industries by change in HHI and go long top 10 industries and short bottom 10
    long_ind=context.conc['change'].nlargest(10).index.tolist()
    short_ind=context.conc['change'].nsmallest(10).index.tolist()
    
    context.output=context.output[context.output['in_trading_univ']]

    # Equally weight 10 industries, and equally weight stocks within each industry
    context.output['weights']=context.output.groupby('industry')['sales'].transform(lambda x: (.5/10)/x.count())
    context.longs=context.output[context.output['industry'].isin(long_ind)].index.tolist()
    context.longs_weights=context.output.weights[context.output['industry'].isin(long_ind)].values.tolist()
    longs_weights_dict=dict(zip(context.longs,context.longs_weights))
    context.shorts=context.output[context.output['industry'].isin(short_ind)].index.tolist()
    context.shorts_weights=context.output.weights[context.output['industry'].isin(short_ind)].values.tolist()
    shorts_weights_dict=dict(zip(context.shorts,context.shorts_weights))

    # Copy the 'new' HHI into 'old' HHI for comparison next year
    context.conc['old']=context.conc['new']
    
    
    
    ### Optimize code. ###
    # Makes the weights negative for our shorts
    shorts_weights_dict = {k: -v for k, v in shorts_weights_dict.items()}
    
    # Combine the weights for our longs and shorts in one dictionary
    target_weights = longs_weights_dict.copy()
    target_weights.update(shorts_weights_dict)
    
    # Place orders according to weights.
    # Note that with the Optimize API, you don't have to explicity set the weight to zero for an unwind.
    #  If you have an existing position and no weight is given, it assumes the weight is zero.
    order_optimal_portfolio(objective=opt.TargetPortfolioWeights(target_weights),
                           constraints=[],
                           universe=target_weights.keys())

    
    # For comparison, this is the same ordering code, but not using the Optimize API
    # for security in context.portfolio.positions:
    #     # Unwinds
    #     if (security not in context.longs) and (security not in context.shorts) and data.can_trade(security): 
    #         order_target_percent(security, 0)
 
    # for security in context.longs:
    #     # New longs
    #     if data.can_trade(security):
    #         order_target_percent(security, longs_weights_dict[security])

    # for security in context.shorts:
    #     # New shorts
    #     if data.can_trade(security):
    #         order_target_percent(security, -shorts_weights_dict[security])
            

def my_record_vars(context, data):
    """
    Record variables at the end of each day.
    """
    if len(context.longs)==0:
        return
    else:
        longs = shorts = 0
        for position in context.portfolio.positions.itervalues():
            if position.amount > 0:
                longs += 1
            elif position.amount < 0:
                shorts += 1
        # Record our variables.
        record(leverage=context.account.leverage, long_count=longs, short_count=shorts)
    
        # log.info("Today's shorts: "  +", ".join([short_.symbol for short_ in context.shorts]))
        # log.info("Today's longs: "  +", ".join([long_.symbol for long_ in context.longs]))