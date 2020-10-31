import math
import talib

#no slippage / unfilled orders make testing annoying / volume of some ETFs not high enough?

set_slippage(slippage.FixedSlippage(spread=0.00))

'''
TODO v3
- Experiment use APR instead of ATR to prioritize stocks
- Apply a filter, either using max/min or atr over a longer period
- Bring the turtle back to life
  - Widen the stop
  - Scale out, not in
  - Reduce sector risk
  - Add markets (prioritize by volatility, manage drag-down)
  - Add filter
- Factor in broker costs(?)

ROADMAP
* Adapt algo for futures trading with margin and leverage
* Apply genetic algo or some other machine learning technique to tweak APR trendwatching params, params: limit, atr days
* Implement s2 as well, or some trailing algo that expands on bigger trends (KISS?)
'''
def initialize(context):
    
    context.securities = [
        #BOND
        sid(23921), #20 year treasury
        sid(23870), #7-10 year treasury    
            
        #CURRENCY
        sid(32304), #canada 
        sid(33334), #japan
        sid(27894), #euro
        sid(32307), #aussie    
      
        #NY COFFEE COKE SUGAR
        sid(36468), #coffee
        sid(41318), #cocoa
        sid(36462), #sugar
        sid(36466), #cotton
            
        #COMEX
        sid(26807), #gold
        sid(28368), #silver
        sid(34926), #copper
            
        #NYME
        sid(28320), #us oil
        sid(33697), #us nat gas
        ] 
    
    schedule_function(rebalance, date_rules.every_day())

    context.atr_padding = 14
    
    # ratio
    divider = 2.0
    
    # entry / exit
    context.days_go_long = 120
    context.days_sell_short = 90
    
    # piramiding / profit
    context.days_exit_short = int(context.days_go_long / divider)
    context.days_exit_long = int(context.days_sell_short / divider)

    # risk exposure
    context.s1_unit_limit = 50
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # FSC for IB

def get_account_size(context):
    return context.portfolio.cash + context.portfolio.positions_value
    

def rebalance(context, data):
    hist_with_padding = data.history(context.securities, ['high', 'low', 'close'], context.days_go_long + context.atr_padding, '1d')
    
    # adjust for calculating with padding
    hist_s1 = hist_with_padding[-context.days_go_long:]
    
    # TODO sort security by volatility - also test results
    # sorted(list_of_objects, key=lambda obj: obj.method_name())
    
    # for each security in the list
    for security in context.securities:
        account_size = get_account_size(context)
        price = data.current(security, 'price')
        position = context.portfolio.positions[security].amount
        
        # x day ATR
        atr_s1 = talib.ATR(hist_s1['high'][security],
                        hist_s1['low'][security],
                        hist_s1['close'][security],
                        timeperiod=context.days_go_long)[-1]
        
        # get x day ATR
        N = abs(atr_s1)
        
        # buy and stop units
        one_percent = account_size * .01
        trade_amount = math.floor(one_percent/(N * price)) # unit
        
        # entry
        # go long s1 (because Donchian breakout)
        high_x_s1 = max(hist_s1['high'][security][0:-1]) # the last x-1 days, where x = days
        if price > high_x_s1 and position is 0: # and filter_small_atr > filter_large_atr:
            try:
                order(security, trade_amount)
                print("GO LONG %s / price %s / past high %s  / n %s / orders %s"  % (security, price, high_x_s1, N, trade_amount))
            except ValueError:
                # there are days when the adj close is NaN, this buggers up N
                print "error: trade_amount = math.floor(%s / (%s * %s))" % (one_percent, N, price)

        # total complete exit
        # sell short s1
        low_x_s1 = min(hist_s1['low'][security][0:-1])
        if price < low_x_s1 and position > 0: # and filter_small_atr < filter_large_atr:
            try:
                order(security, -position)
                print("SELL SHORT %s / price %s / past low %s  / n %s / amount %s"  % (security, price, low_x_s1, N, -trade_amount))
            except ValueError:
                print "error: trade_amount = math.floor(%s / (%s * %s))" % (one_percent, N, price)
        
        # exit long when price drops below a y day low, where x > y (pocket profit)
        low_y_s1 = min(hist_s1['low'][security][-context.days_exit_long:])
        if price < low_y_s1 and position > 0:
            try:
                order(security, -trade_amount)
                print("EXIT LONG %s / price %s / past low %s  / n %s / amount %s"  % (security, price, low_y_s1, N, -trade_amount))
            except ValueError:
                print "error: trade_amount = math.floor(%s / (%s * %s))" % (one_percent, N, price)

        # exit short when price exceeds a y day high, where x > y (buy)
        high_y_s1 = max(hist_s1['high'][security][-context.days_exit_short:])
        limit = context.s1_unit_limit * trade_amount
        if price < high_y_s1 and position > 0 and trade_amount and position < limit:
            try:
                order(security, trade_amount)
                print("EXIT LONG %s / price %s / past low %s  / n %s / amount %s"  % (security, price, low_y_s1, N, -trade_amount))
            except ValueError:
                print "error: trade_amount = math.floor(%s / (%s * %s))" % (one_percent, N, price)