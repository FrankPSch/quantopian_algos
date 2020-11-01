from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.algorithm import order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume, Latest
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters.fundamentals import Q1500US
import quantopian.optimize as opt
from operator import truediv

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    set_long_only()
    schedule_function(my_rebalance,    
                      date_rules.month_start(),                               time_rules.market_open())
            
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector
    my_pipe =make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
         
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    #Chose only "good stocks", i.e. they do no trade OTC, are not depositary receips, liquid enough etc.
    base_universe = Q1500US()
    
    #Chose low cap and high b/m stocks
    cap = Latest(inputs=[Fundamentals.market_cap],       
                 mask=base_universe)
    
    bm = Latest(inputs=[Fundamentals.book_value_yield], mask=base_universe)
    
    hbm = bm.percentile_between(80,100)
    lcap = cap.percentile_between(0,20)
    
    #Filtering out our pipeline
    ts = (lcap & hbm)
        
    return Pipeline(
      columns={
          'cap' :cap
       },
    screen=ts,
    )
 

     
def compute_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    
    weights = {}
   
    #Value weighted portfolio      
    for sec in context['sec_list']:
        if data.can_trade(sec): 
            weights[sec]=context['caps'][sec] 
        else:
            weights[sec]=0
    #deviding each security's cap by sum of all caps    
    cap_sum = sum(weights.values()) 
    
    for sec in weights.keys():
        weights[sec]=weights[sec]/cap_sum
    
    
    return weights


       
  
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    pipeline_results = pipeline_output('my_pipeline')
   
    #time series inexed by secutities; holds market cap
    context['caps']=pipeline_results['cap']
    
    # These are the securities that we are interested in trading each day.
    context['sec_list'] = []
    for i in pipeline_results.index.values:
        context['sec_list'].append(i)
    
    
    
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    
    
    """    
   
    target_weights = compute_weights(context, data)
    
    today = get_datetime()  
    #testing
    context['a'] = target_weights[context['sec_list'][1]]
    log.info('weight of first secutity: ' +str(context['a']))
    
    #Rebalancing in June
    if today.month == 6 and target_weights:
        order_optimal_portfolio(
        objective=opt.TargetWeights(target_weights),
        constraints=[],)
    
    log.info('month: ' + str(today.month))
        
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    maxw = max(compute_weights(context, data).values())
    
    record(number_of_stocks=len(compute_weights(context, data).keys()), biggest_weight=maxw)
    
    """
    Called every minute.
    """