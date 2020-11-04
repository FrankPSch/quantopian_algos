import talib

def initialize(context):    
    
    schedule_function(setAlerts, date_rules.every_day(), time_rules.market_close())
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())
    set_commission(commission.PerTrade(cost=5.00))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    set_benchmark(symbol('SPY'))
    
    context.asset = symbol('SPY')
    
    context.rsi_period = 2
    context.OB = 50
    context.OS = 30
    context.pct_alloc = 1.00
    context.leverage = 1.00
    
    #Alert to buy or sell next day.
    context.buyAssetAlert = False
    context.sellAssetAlert = False
    
def setAlerts(context, data):    
    
    asset_price = data.history(context.asset, 'price', 3, '1d')
    
    rsi = talib.RSI(asset_price, context.rsi_period)

    if rsi[-1] < context.OS and data.can_trade(context.asset):        
        #order_target_percent(asset, context.pct_alloc * leverage)
        context.buyAssetAlert = True
    elif rsi[-1] > context.OB and data.can_trade(context.asset):
        #order_target_percent(asset, 0.00 * leverage)
        context.sellAssetAlert = True

    record(leverage = context.account.leverage)
    
def rebalance(context, data):
    
    if context.buyAssetAlert and data.can_trade(context.asset) and context.portfolio.positions[context.asset].amount == 0:  
        order_target_percent(context.asset, context.pct_alloc * context.leverage)
        context.buyAssetAlert = False
    if context.sellAssetAlert and data.can_trade(context.asset):
        order_target_percent(context.asset, 0.00 * context.leverage)
        context.sellAssetAlert = False