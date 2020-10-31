"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.filters.morningstar import Q500US
import talib
import numpy as np
import pandas as pd
from quantopian.pipeline.data.quandl import cboe_vix, cboe_vxv, cboe_vxd, cboe_vvix
#from quantopian.pipeline.data.quandl import yahoo_index_vix
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.factors import CustomFactor, Latest

class GetVIX(CustomFactor):
    window_length = 1
    def compute(self, today, assets, out, vix):
        out[:] = vix[-1]
 
class GetVIXma(CustomFactor):
    def compute(self, today, assets, out, vix):
        out[:] = vix.mean()
        
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.stocks = [sid(8554), sid(38054), sid(40516)] # spy, vxx, xiv
    context.spy = sid(8554)
    context.xiv = sid(40516)
    context.vxx = sid(38054)
    context.sell_price     = 0
    context.vix_last_price = 0
    
    set_benchmark(sid(40516))
    
    context.allocation = [0,0,0]
    
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(minutes=1))
    #schedule_function(check_xiv, date_rules.every_day(), time_rules.market_close(minutes=60))     
    # Record tracking variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
    
    #Set slippage and commission
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    #set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))

    # fetch VIX data
    pipe = Pipeline()
    attach_pipeline(pipe, 'my_pipeline')
    
    #get VIX at market open
    pipe.add(GetVIX(inputs=[cboe_vix.vix_close]), 'VixOpen')
    pipe.add(GetVIX(inputs=[cboe_vvix.vvix]), 'Vvix')
    pipe.add(GetVIXma(inputs=[cboe_vvix.vvix], window_length = 5), 'Vvix_ma5')
    #get VIX average in the last 2 days
    pipe.add(SimpleMovingAverage(inputs=[cboe_vix.vix_close], window_length=2), 'vix_mean')  

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    
def before_trading_start(context, data):
    """
    Called every day before market open. Variable definition
    """
    
    output = pipeline_output('my_pipeline')
    context.vix = output["VixOpen"].iloc[0]
    context.vvix = output["Vvix"].iloc[0] 
    context.vvix_ma5 = output["Vvix_ma5"].iloc[0] 
    context.vix_mean = output["vix_mean"].iloc[0] # last 2 days average of VIX
    # We get VIX's current price. 
    current_price = context.vix  # vix current value at market open
    record(VIX=current_price)
    #vix_mean = context.vix_mean  # vix mean value of last 2 days
    if (context.vix_last_price == 0):
        context.vix_last_price = current_price # set our last trading VIX price

    prices = data.history(context.spy, ['price','low'], 100, '1d')
    #xivpri = data.history(context.xiv, ['price','low'], 100, '1d')
    #vxxpri = data.history(context.vxx, ['price','low'], 100, '1d')
    
    xiv_current = data.current(context.xiv, 'price') 
    vxx_current = data.current(context.vxx, 'price')
    record(XIV=xiv_current)
    record(VXX=vxx_current)
    record(Vvix=context.vvix)
    record(Vvix=context.vvix_ma5)   

    previous=30
    after=0
    #aftertop=1
    
    
        # Loop through our list of stocks
    for stock in context.stocks:
        # Get the StochRSI of this stock.
        rsik = talib.STOCHRSI(prices['price'], timeperiod=14, fastk_period=3, fastd_period=3)[0]
        rsid = talib.STOCHRSI(prices['price'], timeperiod=14, fastk_period=3, fastd_period=3)[1]
     
    WVF = (prices['price'].rolling(28).max() - prices['low'])/prices['price'].rolling(28).max() * 100
    
    shortma = pd.rolling_mean(WVF, 12)
    longma  = pd.rolling_mean(WVF, 26)
    macd    = shortma - longma
    signal  = pd.rolling_mean(macd, 9)
    trigger = macd - signal
    
    # Get the Moving average of WVF.
    #Moving_WVF = WVF.rolling(3).mean()
    
    # Get the bollinger bands of WVF.
    WVFupper, WVFmiddle, WVFlower = talib.BBANDS(
        np.array(WVF), 
        timeperiod=28,
        # number of non-biased standard deviations from the mean
        nbdevup=2,
        nbdevdn=2,
        matype=0)

    # ROC on XIV to trigger sell
    #ROC = xivpri.apply(talib.ROC, timeperiod=3).iloc[-1]
    #log.info("xivpri ROC %s" %ROC["price"])
    
    if prices['price'][-after-1] < min(prices['price'][-previous-after-1:-after-1]):
        
        if after > 1 and prices['price'][-after-1] < min(prices['price'][-after:-1]):
            context.allocation[2] = 1 #xiv / is it getting here?
            context.allocation[0] = 0.0 #spy 
            context.allocation[1] = 0.0 #vxx
            
    if prices['price'][-after-1] < min(prices['price'][-previous-after-1:-after-1]):
        
        if after > 1 and prices['price'][-after-1] < min(prices['price'][-after:-1]):
            context.allocation[2] = 1 #xiv / is it getting here?
            context.allocation[0] = 0.0
            context.allocation[1] = 0.0
                      
        elif after == 1 and prices['price'][-after-1] < prices['price'][-1]:
            context.allocation[2] = 1 #xiv / is it getting here?
            context.allocation[0] = 0.0
            context.allocation[1] = 0.0
            
        elif after == 0:
            context.allocation[2] = 0.5
            context.allocation[0] = 0.0
            context.allocation[1] = 0.0
            
    else:
        # When WVF goes below the lower band, time to buy the stock.
        if (WVF[-2] > WVFlower[-2] and WVF[-1] < WVFlower[-1]):
            context.allocation[2] = 0.5 #not much difference 0.5 or 1
            context.allocation[0] = 0
            context.allocation[1] = 0

        if (rsik[-2]>rsid[-2] and rsik[-1]<=rsid[-1]):
            context.allocation[2] = 0.5 #relevant
            context.allocation[0] = 0.5
            context.allocation[1] = 0

        if (trigger[-2]>0 and trigger[-1]<0): # MACD Signal for the WVF.
            context.allocation[2] = 0.5 #relevant
            context.allocation[0] = 0.5
            context.allocation[1] = 0.0
            
    if prices['price'][-after-1] > max(prices['price'][-previous-after-1:-after-1]):
        
        if after == 0:
            context.allocation[2] = 0 #not getting here?
            context.allocation[0] = 0
            context.allocation[1] = 0.9 #0.4  low impact
          
    else:            
    # When WVF goes above the upper band, time to buy.    
        if WVF[-2] < WVFupper[-2] and WVF[-1] > WVFupper[-1]:
            context.allocation[2] = 0.0
            context.allocation[0] = 0.0
            context.allocation[1] = 0.9 #1 high impact
            
        if (trigger[-2]<0 and trigger[-1]>0): # MACD Signal for the WVF.
            context.allocation[2] = 1 #0
            context.allocation[0] = 0 #
            context.allocation[1] = 0.0
            
        if (WVF[-1]>10):
            context.allocation[2] = 0
            context.allocation[0] = 1
            context.allocation[1] = 0
     
        #if (ROC[-1] < -20):
            #context.allocation[2] = 1.0
            #context.allocation[0] = 0.0
            #context.allocation[1] = 0.0
        
        #if (current_price > 49):
        #    context.allocation[2] = 0.0 #not getting here
        #   context.allocation[0] = 0.0
        #   context.allocation[1] = 0.0
        
        if (context.vvix > 120) and (context.vvix_ma5 < context.vvix):
            context.allocation[2] = 0.5
            context.allocation[0] = 0.5
            context.allocation[1] = 0.0
            
        if (context.vvix > 105) and (context.vvix_ma5 > context.vvix):
            context.allocation[2] = 0.5
            context.allocation[0] = 0.5
            context.allocation[1] = 0.0

def check_xiv(context, data):
    xivpri = data.history(context.xiv, ['price','low'], 100, '1d')
    xiv_current = data.current(context.xiv, 'price')     
    MA = xivpri.apply(talib.MA, timeperiod=5).iloc[-1]
    change = (xiv_current - MA["price"])/MA["price"]*100
    log.info("change %f" %change)
    if change < -8:
        log.info("xiv down triggered..............")
        context.allocation[2] = 0.0
        context.allocation[0] = 0.0
        context.allocation[1] = 0.0
        my_rebalance(context,data)

def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing. 
    """    
    #allocation = np.array(context.allocation)/sum(context.allocation)
    allocation = context.allocation
    for i,stock in enumerate(context.stocks):
        order_target_percent(stock, allocation[i])
    log.info (", ".join(["%s %0.3f" % (stock.symbol, allocation[i]) for i,stock in enumerate(context.stocks)]))

    
def my_record_vars(context, data):
 # Check how many long and short positions we have.
    """
    This function is called at the end of each day and plots our leverage as well
    as the number of long and short positions we are holding.
    """

    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    # Record our variables.
    #record(leverage=context.account.leverage, long_count=longs, short_count=shorts)

def pvr(context, data):  
    ''' Custom chart and/or logging of profit_vs_risk returns and related information  
    '''  
    # # # # # # # # # #  Options  # # # # # # # # # #  
    logging         = 1            # Info to logging window with some new maximums. 

    record_pvr      = 1            # Profit vs Risk returns (percentage)  
    record_pvrp     = 0            # PvR (p)roportional neg cash vs portfolio value  
    record_cash     = 0            # Cash available  
    record_max_lvrg = 0            # Maximum leverage encountered  
    record_risk_hi  = 0            # Highest risk overall  
    record_shorting = 0            # Total value of any shorts  
    record_max_shrt = 0            # Max value of shorting total  
    record_cash_low = 0            # Any new lowest cash level  
    record_q_return = 0            # Quantopian returns (percentage)  
    record_pnl      = 0            # Profit-n-Loss  
    record_risk     = 0            # Risked, max cash spent or shorts beyond longs+cash  
    record_leverage = 0            # Leverage (context.account.leverage)  
    record_overshrt = 0            # Shorts beyond longs+cash  
    if record_pvrp: record_pvr = 0 # if pvrp is active, straight pvr is off

    import time  
    from datetime import datetime  
    from pytz import timezone      # Python will only do once, makes this portable.  
                                   #   Move to top of algo for better efficiency.  
    c = context  # Brevity is the soul of wit -- Shakespeare [for readability]  
    if 'pvr' not in c:  
        date_strt = get_environment('start').date()  
        date_end  = get_environment('end').date()  
        cash_low  = c.portfolio.starting_cash  
        c.cagr    = 0.0  
        c.pvr     = {  
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent  
            'max_lvrg'   : 0,  
            'max_shrt'   : 0,  
            'risk_hi'    : 0,  
            'days'       : 0.0,  
            'date_prv'   : '',  
            'date_end'   : date_end,  
            'cash_low'   : cash_low,  
            'cash'       : cash_low,  
            'start'      : cash_low,  
            'begin'      : time.time(),  # For run time  
            'log_summary': 126,          # Summary every x days  
            'run_str'    : '{} to {}  ${}  {} US/Eastern'.format(date_strt, date_end, int(cash_low), datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M"))  
        }  
        log.info(c.pvr['run_str'])

    def _pvr(c):  
        c.cagr = ((c.portfolio.portfolio_value / c.pvr['start']) ** (1 / (c.pvr['days'] / 252.))) - 1  
        ptype = 'PvR' if record_pvr else 'PvRp'  
        log.info('{} {} %/day   cagr {}   Portfolio value {}'.format(ptype, '%.4f' % (c.pvr['pvr'] / c.pvr['days']), '%.1f' % c.cagr, '%.0f' % c.portfolio.portfolio_value))  
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % (c.portfolio.portfolio_value - c.pvr['start']), '%.0f' % c.pvr['risk_hi'], '%.1f' % c.pvr['pvr']))  
        log.info('  QRet {} PvR {} CshLw {} MxLv {} RskHi {} MxShrt {}'.format('%.2f' % q_rtrn, '%.2f' % c.pvr['pvr'], '%.0f' % c.pvr['cash_low'], '%.2f' % c.pvr['max_lvrg'], '%.0f' % c.pvr['risk_hi'], '%.0f' % c.pvr['max_shrt']))

    def _minut():  
        dt = get_datetime().astimezone(timezone('US/Eastern'))  
        return str((dt.hour * 60) + dt.minute - 570).rjust(3)  # (-570 = 9:31a)

    date = get_datetime().date()  
    if c.pvr['date_prv'] != date:  
        c.pvr['date_prv'] = date  
        c.pvr['days'] += 1.0  
    do_summary = 0  
    if c.pvr['log_summary'] and c.pvr['days'] % c.pvr['log_summary'] == 0 and _minut() == '100':  
        do_summary = 1              # Log summary every x days  
    if do_summary or date == c.pvr['date_end']:  
        c.pvr['cash'] = c.portfolio.cash  
    elif c.pvr['cash'] == c.portfolio.cash and not logging: return  # for speed

    longs  = sum([p.amount * p.last_sale_price for s, p in c.portfolio.positions.items() if p.amount > 0])  
    shorts = sum([p.amount * p.last_sale_price for s, p in c.portfolio.positions.items() if p.amount < 0])  
    q_rtrn        = 100 * (c.portfolio.portfolio_value - c.pvr['start']) / c.pvr['start']  
    cash          = c.portfolio.cash  
    cash_dip      = int(max(0, c.pvr['start'] - cash))  
    risk          = int(max(cash_dip, shorts))  
    new_risk_hi   = 0  
    new_max_lv    = 0  
    new_max_shrt  = 0  
    new_cash_low  = 0               # To trigger logging in cash_low case  
    overshorts    = 0               # Shorts value beyond longs plus cash  
    if record_pvrp and cash < 0:    # Let negative cash ding less when portfolio is up.  
        cash_dip = int(max(0, c.pvr['start'] - cash * c.pvr['start'] / c.portfolio.portfolio_value))  
        # Imagine: Start with 10, grows to 1000, goes negative to -10, shud not be 200% risk.

    if int(cash) < c.pvr['cash_low']:             # New cash low  
        new_cash_low = 1  
        c.pvr['cash_low']  = int(cash)            # Lowest cash level hit  
        if record_cash_low: record(CashLow = c.pvr['cash_low'])

    if c.account.leverage > c.pvr['max_lvrg']:  
        new_max_lv = 1  
        c.pvr['max_lvrg'] = c.account.leverage    # Maximum intraday leverage  
        if record_max_lvrg: record(MaxLv   = c.pvr['max_lvrg'])

    if shorts < c.pvr['max_shrt']:  
        new_max_shrt = 1  
        c.pvr['max_shrt'] = shorts                # Maximum shorts value  
        if record_max_shrt: record(MxShrts = c.pvr['max_shrt'])

    if risk > c.pvr['risk_hi']:  
        new_risk_hi = 1  
        c.pvr['risk_hi'] = risk                   # Highest risk overall  
        if record_risk_hi:  record(RiskHi  = c.pvr['risk_hi'])

    # Profit_vs_Risk returns based on max amount actually spent (risk high)  
    if c.pvr['risk_hi'] != 0: # Avoid zero-divide  
        c.pvr['pvr'] = 100 * (c.portfolio.portfolio_value - c.pvr['start']) / c.pvr['risk_hi']  
        ptype = 'PvRp' if record_pvrp else 'PvR'  
        if record_pvr or record_pvrp: record(**{ptype: c.pvr['pvr']})

    if shorts > longs + cash: overshorts = shorts             # Shorts when too high  
    if record_shorting: record(Shorts    = shorts)            # Shorts value as a positve  
    if record_overshrt: record(OvrShrt   = overshorts)        # Shorts beyond payable  
    if record_cash:     record(Cash = cash)                   # Cash  
    if record_leverage: record(Lvrg = c.account.leverage)     # Leverage  
    if record_risk:     record(Risk = risk)   # Amount in play, maximum of shorts or cash used  
    if record_q_return: record(QRet = q_rtrn) # Quantopian returns to compare to pvr returns curve  
    if record_pnl:      record(PnL  = c.portfolio.portfolio_value - c.pvr['start']) # Profit|Loss

    if logging and (new_risk_hi or new_cash_low or new_max_lv or new_max_shrt):  
        csh     = ' Cash '   + '%.0f' % cash  
        risk    = ' Risk '   + '%.0f' % risk  
        qret    = ' QRet '   + '%.1f' % q_rtrn  
        shrt    = ' Shrt '   + '%.0f' % shorts  
        ovrshrt = ' oShrt '  + '%.0f' % overshorts  
        lv      = ' Lv '     + '%.1f' % c.account.leverage  
        pvr     = ' PvR '    + '%.1f' % c.pvr['pvr']  
        rsk_hi  = ' RskHi '  + '%.0f' % c.pvr['risk_hi']  
        csh_lw  = ' CshLw '  + '%.0f' % c.pvr['cash_low']  
        mxlv    = ' MxLv '   + '%.2f' % c.pvr['max_lvrg']  
        mxshrt  = ' MxShrt ' + '%.0f' % c.pvr['max_shrt']  
        pnl     = ' PnL '    + '%.0f' % (c.portfolio.portfolio_value - c.pvr['start'])  
        log.info('{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(), lv, mxlv, qret, pvr, pnl, csh, csh_lw, shrt, mxshrt, ovrshrt, risk, rsk_hi))  
    if do_summary: _pvr(c)  
    if get_datetime() == get_environment('end'):    # Summary at end of run  
        if 'pvr_summary_done' not in c: c.pvr_summary_done = 0  
        if not c.pvr_summary_done:  
            _pvr(c)  
            elapsed = (time.time() - c.pvr['begin']) / 60  # minutes  
            log.info( '{}\nRuntime {} hr {} min'.format(c.pvr['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))  
            c.pvr_summary_done = 1

def handle_data(context, data):  
    pvr(context, data)
    pass