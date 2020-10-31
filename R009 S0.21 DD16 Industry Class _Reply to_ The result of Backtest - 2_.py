#加一個rebalance
import numpy as np
import pandas as pd
from itertools import repeat, chain
from quantopian.pipeline import CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.data import morningstar as ms
from quantopian.pipeline.filters.morningstar import IsPrimaryShare
from quantopian.pipeline.filters.morningstar import Q500US,Q1500US
# These last two imports are for using Quantopian's Optimize API
from quantopian.algorithm import order_optimal_portfolio
import quantopian.optimize as opt
#from quantopian.algorithm import calendars

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    #Set benchmark to short-term Treasury note ETF (SHY) since strategy is dollar neutral
    set_benchmark(sid(23911))
    
    schedule_function(my_assign_weights, date_rules.month_start(), time_rules.market_open())
    
    schedule_function(close_orders, date_rules.month_end(), time_rules.market_close(minutes=30))
    
    # Set commissions and slippage to 0 to determine pure alpha
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    # context.conc is a DataFrame, indexed by industry, of the old HHI, the new HHI, and 
    #   number of stocks in each industry
    context.conc=pd.DataFrame(columns=['old','new','count'])

    context.long=[]
    context.short=[]
    

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')

class Market_cap(CustomFactor):   
    inputs = [ms.valuation.market_cap]  
    window_length = 1000
    
    def compute(self, today, assets, out, market_cap):       
        out[:] = market_cap[-1]
        
class Sales(CustomFactor):   
    inputs = [ms.income_statement.operating_revenue]  
    window_length = 189
    
    def compute(self, today, assets, out, sales):       
        out[:] = sales[0]+sales[-1]+sales[-64]+sales[-127]         
        
class Industry(CustomFactor):
    inputs=[ms.asset_classification.morningstar_industry_code]
    window_length=100
    
    def compute(self, today, assets, out, industry):
        out[:] = industry[-1] 
        
        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    # Base universe set to the Q500US
    base_universe = Q1500US()
    trading_universe = Q1500US()

    # Factor of yesterday's close price.
    pricing = USEquityPricing.close.latest
    
    primary_share = IsPrimaryShare(mask=base_universe)
    market_cap = Market_cap(mask=base_universe)
    sales = Sales(mask=base_universe)
    industry=Industry(mask=base_universe)
        
    universe = (
          primary_share & (pricing > 5)
    )

    return Pipeline(
#        screen = base_universe,
        screen = universe,
        columns = {          
            'market_cap':market_cap,
            'sales':sales,
            'industry':industry,           
            'in_trading_univ':trading_universe,
                  }
    )
   
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')

     
def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    context.output=context.output.dropna()#丟掉有Nan值的
    context.output=context.output[context.output['industry']!=-1]#丟掉產業編號-1的
    context.output['industry']=context.output['industry'] // 1000
    #把原本六位數的NAICS變成三位數的NAICS
    
    log.info('Number of unique industries: %d' %(context.output['industry'].nunique()))#回傳不同的產業代e碼有幾個

    # 'share' is the  market share of total sales for ach company
    context.output['share']=context.output.groupby('industry')['sales'].transform(lambda x: (x/x.sum()))
    
    # We square market share for HHI calculation below
    context.output['share']=context.output['share']**2

    # 'new' 是每個產業現在的HHI，= sum of squared market shares 
    context.conc['count']=context.output.groupby('industry')['share'].count()
    print context.conc['count']
    context.conc['new']=context.output.groupby('industry')['share'].sum()
    context.conc=context.conc[context.conc['count'] >= 6]
    
    context.conc['new']=context.conc['new']*10000
    context.conc=context.conc.sort_values(['new'],ascending=True)
    hhi_industry = context.conc['new'].index.tolist()

    number = int(len(hhi_industry)*0.2)
    hhi_high_industry = hhi_industry[-number : ]
    hhi_low_industry = hhi_industry[ : number]

       
#    hhi_high_industry = hhi_industry[int(len(hhi_industry) * .80) : int(len(hhi_industry))]
#    hhi_low_industry = hhi_industry[0 : int(len(hhi_industry)* .20)]
#    print(hhi_high_industry)
#    print(hhi_low_industry)
    
    all_stock = context.output.index.tolist()
    all_industry = context.output['industry'].tolist()
    table = { 'all_stock': all_stock,
              'all_industry': all_industry
             }

    df = pd.DataFrame.from_dict(table)
    df = df.sort_values(['all_industry'],ascending=True)
    df = df[df['all_industry'].isin( hhi_high_industry)]
    long_stock = df['all_stock'].tolist()
    df = df.groupby('all_industry')['all_stock'].count()
    long_industry_count = df.tolist()

    df2 = pd.DataFrame.from_dict(table)
    df2 = df2.sort_values(['all_industry'],ascending=True)
    df2 = df2[df2['all_industry'].isin( hhi_low_industry)]
    short_stock = df2['all_stock'].tolist()
    df2 = df2.groupby('all_industry')['all_stock'].count()
    short_industry_count = df2.tolist()

# Equally weight industries, and equally weight stocks within each industry    
    total_industry = len(long_industry_count)+len(short_industry_count)
      
    weights = [1.0/total_industry/x for x in long_industry_count]
    weights2 = [1.0/total_industry/x for x in short_industry_count]
    #得到不同產業的每家公司的比重ex:[0.016, 0.014, 0.016, 0.016, 0.016]

    longs_weights = list(chain.from_iterable(repeat(i, j) for i, j in zip(weights, long_industry_count)))
    shorts_weights = list(chain.from_iterable(repeat(i, j) for i, j in zip(weights2, short_industry_count)))
    
    longs_weights_dict=dict(zip(long_stock,longs_weights))
    shorts_weights_dict=dict(zip(short_stock,shorts_weights))
    
    
    #context.conc['order_share']=
    #df4 = pd.DataFrame(longs_weights_dict,columns=['long_stock', 'longs_weights'])
    #dk = pd.DataFrame([longs_weights_dict])
    #context.long['stock']=longs_weights_dict[long_stock]
    
    
    context.long=long_stock
    context.short=short_stock
    #print(context.long)

    for security in long_stock:
    #     # New longs
      if data.can_trade(security):
            order_target_percent(security, longs_weights_dict[security])

    for security in short_stock:
    #     # New shorts
      if data.can_trade(security):
            order_target_percent(security, -shorts_weights_dict[security])
            
def close_orders(context,data):
    
    for security in context.long:
            order_target(security, 0)  
    for security in context.short:    
            order_target(security, 0)  
     

    
#def my_record_vars(context, data):
#    """
#    Record variables at the end of each day.
#    """
#    pass
#    if len(long_merge_list)==0:
#        return
#    else:
#        longs = shorts = 0
#        for position in context.portfolio.positions.itervalues():
#            if position.amount > 0:
#                longs += 1
#            elif position.amount < 0:
#                shorts += 1
        # Record our variables.
#        record(leverage=context.account.leverage, long_count=longs, short_count=shorts)
    
#    log.info("Today's shorts: "  +", ".join([short_.symbol for short_ in context.shorts]))
#    log.info("Today's longs: "  +", ".join([long_.symbol for long_ in context.longs]))