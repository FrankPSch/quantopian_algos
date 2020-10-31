from pytz import timezone
import numpy as np

def initialize(context):  
    
    set_symbol_lookup_date('2015-01-01')
    ########################################
    context.secs = symbols('SPY', # S&P 500
                           'TLT', # T Bonds
                           'GLD', # Gold
                           'QQQ', # Nasdaq 100
                           'EEM', # MSCI EM
                           'VGK') # Europe
    #######################################                       
    set_benchmark(sid(22739))
    context.day_count = -1
    set_commission(commission.PerTrade(cost=1))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))
    
    context.trading_freq = 30
    context.n = 150 # sample size
    
    #trade at the beginning of each month at market open
    schedule_function(trade, date_rules.month_start(), time_rules.market_open())
 
def handle_data(context, data):  
    
    lev = context.account.leverage
    record(lev = lev)
    
def trade(context,data):    
    ### calculating the zscores #############
    stock = context.secs 
    p = history(bar_count=context.n, frequency='1d', field='price')
    p10 = p.iloc[-10:] # previous 10 days
    mu = np.mean(p[stock]) # n day mavg
    mu10 = np.mean(p10[stock]) # 10 day mavg
    sigma = np.std(p[stock])
    z = (mu10-mu)/sigma 
    z = z.order(ascending=True)
    zmean = float(sum(z)/len(z))
    ##########################################
    
    # begin trade logic
    if zmean < -1:  
        for stock in z.index:
            # first check if there are open orders
            if get_open_orders(stock): 
                continue 
            order_target_percent(stock, 0)  
    else:  
        counter = 1.0  
        fraction = 0.96 / ((len(z) * (len(z)+1.0)) / 2.0)  
        for stock in z.index:  
            # first check if there are open orders
            if get_open_orders(stock): 
                continue 
            order_target_percent(stock, counter * fraction)  
            counter += 1.0
            
    record(zmean = zmean)