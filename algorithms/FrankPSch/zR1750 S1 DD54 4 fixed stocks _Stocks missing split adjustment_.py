def initialize(context):
    context.my_equities = [
        sid(26401),
        sid(5213),
        sid(23709),
        sid(21382),
    ]
    
    schedule_function(
        buy_and_hold,
        date_rules.every_day(),
        time_rules.market_open()
    )
    
    context.bought = False
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    
def buy_and_hold(context, data):
    if not context.bought:
        for equity in context.my_equities:
            order_target_percent(equity, 1.0/len(context.my_equities))
            
        context.bought = True
        
    else:
        pass