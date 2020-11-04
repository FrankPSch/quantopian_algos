#Published By: Quant Prophet, LLC 
#Author: Kory Hoang
#Description: this is a simple mean-reversion strategy that uses the RSI2 technical indicator which was originally developed and popularized by Cesar Alvarez. Most people are familiar with the RSI14 but the RSI2 is a much more powerful tool for short-term market timing that works well in both momentum and mean-reversion strtegies. The algorithm monitors the 2-day RSI of SPY (S&P500) and TLT (Long-Term US Treasury), both of which are well-known as mean-reverting assets. When an asset's RSI2 dips below 30 ("Oversold"), go long with 50% of the portfolio. When its RSI2 crosses above 70 ("Overbought"), sell the position and go to cash. The vanilla version of this strategy generates good risk-adjusted performance but still underperforms the S&P 500, for the most part. However, using leverage can increase the risk/reward profile in order to outperform the S&P 500 while still maintaining lower risk. I'm using 2x leverge for this strategy.  

import talib

def initialize(context):    
    
    schedule_function(setAlerts, date_rules.every_day(), time_rules.market_close())
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    set_benchmark(symbol('SPY'))
    
    context.stock = symbol('SPY') #Equity asset
    context.bond = symbol('TLT') #Bond asset 
    context.rsi_period = 2
    context.OB1 = 80 #Overbought threshold 1
    context.OS1 = 40 #Oversold threshold 1
    context.OB2 = 60 #Overbought threshold 2
    context.OS2 = 30 #Oversold threshold 2
    context.pct_alloc1 = 0.50
    context.pct_alloc2 = 0.50
    context.leverage = 1.00
    
    #Alert to buy or sell next day.
    context.buyStockAlert = False
    context.sellStockAlert = False
    context.buyBondAlert = False
    context.sellBondAlert = False
    
def setAlerts(context, data):    
    
    stock_price = data.history(context.stock, 'price', 3, '1d')
    bond_price = data.history(context.bond, 'price', 3, '1d')
    
    rsi1 = talib.RSI(stock_price, context.rsi_period)
    rsi2 = talib.RSI(bond_price, context.rsi_period)

    if rsi1[-1] < context.OS1 and data.can_trade(context.stock):        
        #order_target_percent(stock, context.pct_alloc1 * leverage)
        context.buyStockAlert = True
    elif rsi1[-1] > context.OB1 and data.can_trade(context.stock):
        #order_target_percent(stock, 0.00 * leverage)
        context.sellStockAlert = True
  
    if rsi2[-1] < context.OS2 and data.can_trade(context.bond):        
        #order_target_percent(bond, context.pct_alloc2 * leverage)
        context.buyBondAlert = True
    elif rsi2[-1] > context.OB2 and data.can_trade(context.bond):
        #order_target_percent(bond, 0.00 * leverage)
        context.sellBondAlert = True

    record(leverage = context.account.leverage)
    
def rebalance(context, data):
    
    if context.buyStockAlert and data.can_trade(context.stock) and context.portfolio.positions[context.stock].amount == 0:        
        order_target_percent(context.stock, context.pct_alloc1 * context.leverage)
        context.buyStockAlert = False
    if context.sellStockAlert and data.can_trade(context.stock):
        order_target_percent(context.stock, 0.00 * context.leverage)
        context.sellStockAlert = False
  
    if context.buyBondAlert and data.can_trade(context.bond) and context.portfolio.positions[context.bond].amount == 0:        
        order_target_percent(context.bond, context.pct_alloc2 * context.leverage)
        context.buyBondAlert = False
    if context.sellBondAlert and data.can_trade(context.bond):
        order_target_percent(context.bond, 0.00 * context.leverage)
        context.sellBondAlert = False