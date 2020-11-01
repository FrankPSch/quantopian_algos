def initialize(context):
    context.symbols = {symbol("TQQQ"):0.50, symbol("TMF"):0.25, symbol("EDZ"):0.25}
    #TQQQ - 3x performance of the Nasdaq-100 Index®
    #TMF  - 3x performance of the ICE U.S. Treasury 20+ Year Bond Index
    #EDZ  - 3x inverse performance of the MSCI Emerging Markets Index
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes = 33))
    #0.5, 0.25, 0.25: r3049 a0.29 b0.58 s1.55 dd-26.49

def trade(context, data):  
    wt = {}
    for tick, weigh in context.symbols.items():
        prices = data.history(tick,'open', 21,'1d')
        pricesl = data.history(tick,'open', 252,'1d')
        wt_sh = 1.0/(prices.pct_change()[-14:-1].std()**2)
        if (tick == symbol("TQQQ") or tick == symbol("TMF")):
            wt[tick] = max(0, weigh*(pricesl.max()/prices[-1])*wt_sh) # bull
        else:
            wt[tick] = max(0, weigh*(prices[-1]/pricesl.min())*wt_sh) # bear
    total_wt = sum([wt[x] for x in wt])
    
    round_pct = 5.0 # rounding to reduce slippage and comission
    for tick, weigh in wt.items():
        target_pct = 0.0
        try:
            target_pct = round(wt[tick]/total_wt/round_pct, 2)*round_pct
        except:
            pass
        order_target_percent(tick, target_pct)