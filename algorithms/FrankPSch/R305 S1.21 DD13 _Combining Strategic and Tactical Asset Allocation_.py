# Core-satellite strategy by Vladimir Yevtushenko
# This is a combination of static portfolio and tactical portfolio with a different set of assets that trade on a moving average crossover
# https://www.quantopian.com/posts/combining-strategic-and-tactical-asset-allocation

# ---------------------------------------------------------------------------------
# Static portfolio
core_etf = symbols('QQQ','XLP','TLT','IEF')

# Weight of the static portfolio
proportion = [0.25, 0.25, 0.25, 0.25]

# Tactical portfolio
tact_etf = symbols('XLV', 'XLY', 'TLO', 'GLD')

# Moving averages for tactical portfolio
ma_s, ma_f = 200, 20

# Leverage
lev = 1.0

# Share of the static portfolio
wt_core = 0.25
# ---------------------------------------------------------------------------------


def initialize(context):    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    # Schedule static portfolio
    #schedule_function(trade_core, date_rules.month_start(0), time_rules.market_open(minutes=65))      
    schedule_function(trade_core, date_rules.week_start(2), time_rules.market_open(minutes=65))      

    # Schedule tactical portfolio
    #schedule_function(trade_tact, date_rules.every_day(), time_rules.market_open( minutes=65))  
    schedule_function(trade_tact, date_rules.week_start(2), time_rules.market_open( minutes=65))  

    
def trade_core(context,data):    
    # Trade static portfolio
    
    # Loop through core assets and order target percentages
    for i in range(len(core_etf)):
        if data.can_trade(core_etf[i]):
            order_target_percent(core_etf[i],  lev*wt_core * proportion[i])

            
def trade_tact(context, data):    
    # Trade tactical portfolio

    # Calculate moving averages and fast/slow ratio
    ma_fast = data.history(tact_etf, 'price', ma_f, '1d').mean()
    ma_slow = data.history(tact_etf, 'price', ma_s, '1d').mean()
    ratio   = ma_fast/ma_slow 
    
    # Calculate the targeted position momentum and the weight
    pos_mom = ratio[ratio >= 1.0]   

    # Calculate the targeted weight
    wt      = lev*(1.0 - wt_core) / len(pos_mom) if len(pos_mom) !=0 else 0

    # Loop through tactical assets and order target percentages
    for etf in tact_etf:
        if data.can_trade(etf): 
            if etf in pos_mom.index: 
                order_target_percent(etf, wt)
            else:
                order_target(etf, 0)                

                
def before_trading_start(context,data):    
    # Record data
    record(leverage = context.account.leverage)