import math
from pytz import timezone

trading_freq = 1 # of days
n = 150 # sample size

def initialize(context): 

    context.secs = sid(8554)
       
    context.day_count = -1
    
    set_commission(commission.PerShare(cost=.005, min_trade_cost=1))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))

def handle_data(context, data):

    lev = context.account.leverage   
            
    # Trade according to freq rule
    loc_dt = get_datetime().astimezone(timezone('US/Eastern'))
    if loc_dt.hour == 11 and loc_dt.minute ==0:
        context.day_count += 1
        pass
    else:
        return

    if context.day_count % trading_freq != 0.0:
        return         

    ####################################
    #mean = data[context.secs].mavg(n)
    price_hist = data.history(context.secs, 'price', n, '1d')
    mean = price_hist.mean()

    #sigma = data[context.secs].stddev(n)
    stddev_hist = data.history(context.secs, 'price', n, '1d')[:-1]
    sigma = stddev_hist.std()

    #price = data[context.secs].price
    price = data.current(context.secs, 'price')
    
    z = (price - mean) / sigma
    
    #target = math.sin(z)+1 # adding 1 to the sin wave to prevent shorting
    #target = 0.5*(math.sin(z)+1) # Long without leverage
    
    #target = 1.0/(1+math.exp(-1.2*z)) # Pure momentum
    target = 2.0/(1+math.exp(-1.2*z)) # Pure momentum with leverage
    
    target = round(target*10.0,0)/10.0
    
    # condition required to enter a trade.
    if -4 <= z <= 4 and lev <=2.5:
        order_target_percent(context.secs, target) 
        order_target_percent(symbol('TLT'), 2-target)    
    else:
       target = 0
       order_target_percent(symbol('TLT'), 1)  
       order_target_percent(context.secs, 0)                       
   
    record(Treasury = (1-target)*100)
    record(SPY = target*100)
    #record(target = target)