# Short Bear
# https://www.quantopian.com/posts/problems-observed-in-algo-trading-equities-on-futures-data
    
def initialize(context):
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes = 65))  

def trade(context, data):
    # ------------------------------------------------------
    bear = symbol('VXX')
    future_vx = continuous_future('VX')
    LB = 0.950
    # ------------------------------------------------------
    vx_chain = data.current_chain(future_vx)
    front_contract = vx_chain[0]
    secondary_contract = vx_chain[1]

    ratio = data.current(front_contract, 'price')/data.current(secondary_contract, 'price')    
    wt_bear = -1.0  if ratio < LB else 0.0
    
    if get_open_orders(): return   
    order_target_percent(bear, wt_bear)
        
    record(ratio = ratio, LB = LB)