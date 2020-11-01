"""
Blue's version.  This algorithm enhances a simple five-day mean reversion strategy by:
1. Skipping the last day's return
2. Sorting stocks based on the volatility of the five day return, to get steady moves vs jumpy ones
I also commented out two other filters that I looked at:
1. Six month volatility
2. Liquidity (volume/(shares outstanding))

"""
import numpy as np
import pandas as pd
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters import Q500US

def initialize(context):

    # Set benchmark to short-term Treasury note ETF (SHY) since strategy is dollar neutral
    #set_benchmark(sid(23911))
    set_benchmark(sid(8554)) # SPY

    # Get intraday prices today before the close if you are not skipping the most recent data
    schedule_function(get_prices,date_rules.every_day(), time_rules.market_open(minutes=5))

    # Schedule rebalance function to run at the end of each day.
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=10))

    # Record variables at the end of each day.
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

    # Set commissions and slippage to 0 to determine pure alpha
    #set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    #set_slippage(slippage.FixedSlippage(spread=0))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    # Number of quantiles for sorting returns for mean reversion
    context.nq=5

    # Number of quantiles for sorting volatility over five-day mean reversion period
    context.nq_vol=3

    # Create pipeline and attach it to algorithm.
    pipe = make_pipeline()
    attach_pipeline(pipe, 'pipeline')

    context.waits = {}
    context.waits_max = 3    # trading days

    return

    context.pnl_sids_exclude  = [ sid(2561) ]
    context.pnl_sids  = [  ]
    context.day_count = 0
    #schedule_function(record_pnl, date_rules.every_day(), time_rules.market_close())

    # Take profit several times a day
    context.profit_threshold = .05
    context.profit_logging   = 1
    for i in range(20, 390, 60):    # (low, high, every i minutes)
        #continue    # uncomment to deactivate
        schedule_function(take_profit, date_rules.every_day(), time_rules.market_open(minutes=i))

def wait(c, sec=None, action=None):
    if sec and action:
        if action == 1:
            c.waits[sec] = 1    # start wait
        elif action == 0:
            del c.waits[sec]    # end wait
    else:
        for sec in c.waits.copy():
            if c.waits[sec] > c.waits_max:
                del c.waits[sec]
            else:
                c.waits[sec] += 1   # increment

def take_profit(context, data):    # Close some positions to take profit
    pos     = context.portfolio.positions
    history = data.history(pos.keys(), 'close', 10, '1m').bfill().ffill()
    for s in pos:
        if not data.can_trade(s):      continue
        if slope(history[s])      > 0: continue
        if slope(history[s][-5:]) > 0: continue
        if history[s][-1] > history[s][-2]: continue
        prc = data.current(s, 'price')
        amt = pos[s].amount
        if (amt / abs(amt)) * ((prc / pos[s].cost_basis) - 1) > context.profit_threshold:
            order_target(s, 0)
            wait(context, s, 1)    # start wait
            if not context.profit_logging: continue
            pnl = (amt * (prc - pos[s].cost_basis))
            if pnl < 3000: continue
            log.info('close {} {}  cb {}  now {}  pnl {}'.format(
                amt, s.symbol, '%.2f' % pos[s].cost_basis, prc, '%.0f' % pnl))

import statsmodels.api as sm
def slope(in_list):     # Return slope of regression line. [Make sure this list contains no nans]
    return sm.OLS(in_list, sm.add_constant(range(-len(in_list) + 1, 1))).fit().params[-1]  # slope

def record_pnl(context, data):
    def _pnl_value(sec, context, data):
        pos = context.portfolio.positions[sec]
        return pos.amount * (data.current(sec, 'price') - pos.cost_basis)

    context.day_count += 1

    for s in context.portfolio.positions:
        if not data.can_trade(s): continue
        if s in context.pnl_sids_exclude: continue

        # periodically log all
        if context.day_count % 126 == 0:
            log.info('{} {}'.format(s.symbol, int(_pnl_value(s, context, data))))

        # add up to 5 securities for record
        if len(context.pnl_sids) < 5 and s not in context.pnl_sids:
            context.pnl_sids.append(s)
        if s not in context.pnl_sids: continue     # limit to only them

        # record their profit and loss
        who  = s.symbol
        what = _pnl_value(s, context, data)
        record( **{ who: what } )

class Volatility(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length=132

    def compute(self, today, assets, out, close):
        # I compute 6-month volatility, starting before the five-day mean reversion period
        daily_returns = np.log(close[1:-6]) - np.log(close[0:-7])
        out[:] = daily_returns.std(axis = 0)

class Liquidity(CustomFactor):
    inputs = [USEquityPricing.volume, morningstar.valuation.shares_outstanding]
    window_length = 1

    def compute(self, today, assets, out, volume, shares):
        out[:] = volume[-1]/shares[-1]

class Sector(CustomFactor):
    inputs=[morningstar.asset_classification.morningstar_sector_code]
    window_length=1

    def compute(self, today, assets, out, sector):
        out[:] = sector[-1]

def make_pipeline():
    """
    Create pipeline.
    """

    pricing=USEquityPricing.close.latest

    # Volatility filter (I made it sector neutral to replicate what UBS did).  Uncomment and
    # change the percentile bounds as you would like before adding to 'universe'
    # vol=Volatility(mask=Q500US())
    # sector=morningstar.asset_classification.morningstar_sector_code.latest
    # vol=vol.zscore(groupby=sector)
    # vol_filter=vol.percentile_between(0,100)

    # Liquidity filter (Uncomment and change the percentile bounds as you would like before
    # adding to 'universe'
    # liquidity=Liquidity(mask=Q500US())
    # I included NaN in liquidity filter because of the large amount of missing data for shares out
    # liquidity_filter=liquidity.percentile_between(0,75) | liquidity.isnan()

    profitable = morningstar.valuation_ratios.ev_to_ebitda.latest > 0
    universe = (  
        Q500US()  
#        & (pricing > 5)
        & (pricing > 5) & profitable  
        # & liquidity_filter  
        # & volatility_filter  
    )  

    return Pipeline(
        screen  = universe
    )

def before_trading_start(context, data):
    # Gets pipeline output every day.
    context.output = pipeline_output('pipeline')

    wait(context)    # Increment any that are present

def get_prices(context, data):
    # Get the last 6 days of prices for every stock in universe
    Universe500=context.output.index.tolist()
    prices = data.history(Universe500,'price',6,'1d')
    daily_rets = np.log(prices/prices.shift(1))

    rets=(prices.iloc[-2] - prices.iloc[0]) / prices.iloc[0]
    # I used data.history instead of Pipeline to get historical prices so you can have the
    # option of using the intraday price just before the close to get the most recent return.
    # In my post, I argue that you generally get better results when you skip that return.
    # If you don't want to skip the most recent return, however, use .iloc[-1] instead of .iloc[-2]:
    # rets=(prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    stdevs=daily_rets.std(axis=0)

    rets_df=pd.DataFrame(rets,columns=['five_day_ret'])
    stdevs_df=pd.DataFrame(stdevs,columns=['stdev_ret'])

    context.output=context.output.join(rets_df,how='outer')
    context.output=context.output.join(stdevs_df,how='outer')

    context.output['ret_quantile']=pd.qcut(context.output['five_day_ret'],context.nq,labels=False)+1
    context.output['stdev_quantile']=pd.qcut(context.output['stdev_ret'],3,labels=False)+1

    context.longs=context.output[(context.output['ret_quantile']==1) &
                                (context.output['stdev_quantile']<context.nq_vol)].index.tolist()
    context.shorts=context.output[(context.output['ret_quantile']==context.nq) &
                                 (context.output['stdev_quantile']<context.nq_vol)].index.tolist()


def rebalance(context, data):
    """
    Rebalance daily.
    """
    Universe500=context.output.index.tolist()


    existing_longs=0
    existing_shorts=0
    for security in context.portfolio.positions:
        # Unwind stocks that have moved out of Q500US
        if security not in Universe500 and data.can_trade(security):
            order_target(security, 0)
        else:
            if data.can_trade(security):
                current_quantile=context.output['ret_quantile'].loc[security]
                if context.portfolio.positions[security].amount>0:
                    if (current_quantile==1) and (security not in context.longs):
                        existing_longs += 1
                    elif (current_quantile>1) and (security not in context.shorts):
                        order_target(security, 0)
                elif context.portfolio.positions[security].amount<0:
                    if (current_quantile==context.nq) and (security not in context.shorts):
                        existing_shorts += 1
                    elif (current_quantile<context.nq) and (security not in context.longs):
                        order_target(security, 0)

    order_sids = get_open_orders().keys()                        
    for security in context.longs:
        if security in context.waits: continue
        if security in order_sids:    continue
        if data.can_trade(security):
            order_value(security, context.portfolio.cash / len(context.longs) )

    for security in context.shorts:
        if security in context.waits: continue
        if security in order_sids:    continue
        if data.can_trade(security):
            order_target_percent(security, -.5/(len(context.shorts)+existing_shorts))


def record_vars(context, data):
    """
    Record variables at the end of each day.
    
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1
    """
    
    record(
        leverage=context.account.leverage, 
        long_count  = len(context.longs), 
        short_count = len(context.shorts),
     )

    #log.info("Today's shorts: "  +", ".join([short_.symbol for short_ in context.shorts]))
    #log.info("Today's longs: "  +", ".join([long_.symbol for long_ in context.longs]))

def pvr(context, data):
    ''' Custom chart and/or logging of profit_vs_risk returns and related information
    '''
    # # # # # # # # # #  Options  # # # # # # # # # #
    logging         = 0            # Info to logging window with some new maximums

    record_pvr      = 1            # Profit vs Risk returns (percentage)
    record_pvrp     = 0            # PvR (p)roportional neg cash vs portfolio value
    record_cash     = 1            # Cash available
    record_max_lvrg = 1            # Maximum leverage encountered
    record_risk_hi  = 0            # Highest risk overall
    record_shorting = 0            # Total value of any shorts
    record_max_shrt = 1            # Max value of shorting total
    record_cash_low = 1            # Any new lowest cash level
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
            'pstart'     : c.portfolio.portfolio_value, # Used if restart
            'begin'      : time.time(),                 # For run time
            'log_summary': 126,                         # Summary every x days
            'run_str'    : '{} to {}  ${}  {} US/Eastern'.format(date_strt, date_end, int(cash_low), datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M"))
        }
        log.info(c.pvr['run_str'])

    def _pvr(c):
        c.cagr = ((c.portfolio.portfolio_value / c.pvr['start']) ** (1 / (c.pvr['days'] / 252.))) - 1
        ptype = 'PvR' if record_pvr else 'PvRp'
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (c.pvr['pvr'] / c.pvr['days']), '%.1f' % c.cagr, '%.0f' % c.portfolio.portfolio_value, '%.0f' % (c.portfolio.portfolio_value - c.pvr['start'])))
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
    q_rtrn       = 100 * (c.portfolio.portfolio_value - c.pvr['start']) / c.pvr['start']
    cash         = c.portfolio.cash
    new_risk_hi  = 0
    new_max_lv   = 0
    new_max_shrt = 0
    new_cash_low = 0               # To trigger logging in cash_low case
    overshorts   = 0               # Shorts value beyond longs plus cash
    cash_dip     = int(max(0, c.pvr['pstart'] - cash))
    risk         = int(max(cash_dip, -shorts))

    if record_pvrp and cash < 0:   # Let negative cash ding less when portfolio is up.
        cash_dip = int(max(0, c.pvr['start'] - cash * c.pvr['start'] / c.portfolio.portfolio_value))
        # Imagine: Start with 10, grows to 1000, goes negative to -10, should not be 200% risk.

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
        if record_max_shrt: record(MxShrt  = c.pvr['max_shrt'])

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
    if record_leverage: record(Lvrg = c.account.leverage)     # Leverage
    if record_cash:     record(Cash = cash)                   # Cash
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
        _pvr(c)
        elapsed = (time.time() - c.pvr['begin']) / 60  # minutes
        log.info( '{}\nRuntime {} hr {} min'.format(c.pvr['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))

#def handle_data(context, data):
#    pvr(context, data)