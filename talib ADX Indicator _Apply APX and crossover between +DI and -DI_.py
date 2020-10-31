# talib ADX Indicator

import talib

def initialize(context):
    schedule_function(record_ADX, date_rules.every_day(), time_rules.market_close(hours = 1)) 
    context.aapl = sid(24)
    
def record_ADX(context, data):   

    period = 14
    
    H = data.history(context.aapl,'high', 2*period, '1d').dropna()
    L = data.history(context.aapl,'low', 2*period, '1d').dropna()   
    C = data.history(context.aapl,'price', 2*period, '1d').dropna()
    
    
    ta_ADX = talib.ADX(H, L, C, period)
    ta_nDI = talib.MINUS_DI(H, L, C, period)
    ta_pDI = talib.PLUS_DI(H, L, C, period)
    
    ADX = ta_ADX[-1]
    nDI = ta_nDI[-1]
    pDI = ta_pDI[-1]   

    record( ADX = ADX, nDI = nDI, pDI = pDI) 
    record(close_price = data.current(context.aapl,'close'))
    
    
    if nDI < pDI and ADX > 20 and data.can_trade(context.aapl):
        order_target_percent(context.aapl, 1)
    elif nDI > pDI and ADX > 20 and data.can_trade(context.aapl):
        order_target_percent(context.aapl, -1)

    
        
    

    print context.portfolio.positions[context.aapl].amount