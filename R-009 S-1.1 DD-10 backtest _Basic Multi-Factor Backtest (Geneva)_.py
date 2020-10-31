'''
Multi Factor Backtest - Final
Sal Geneva
10/22/2017

modified
'''

from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data.morningstar import operation_ratios
from quantopian.pipeline.data.morningstar import cash_flow_statement
from quantopian.pipeline.data.morningstar import earnings_ratios
from quantopian.pipeline.data.morningstar import valuation_ratios
from quantopian.pipeline.data.morningstar import balance_sheet
from quantopian.pipeline.data.morningstar import asset_classification
from quantopian.pipeline.data.zacks import EarningsSurprises
from quantopian.pipeline.factors import Returns

def initialize(context):
    set_benchmark(symbol('SHY'))
    set_commission(commission.PerTrade(cost=0.001)) #comment out to use default
    
    attach_pipeline(make_pipeline(), 'pipeline')

    schedule_function(close, date_rules.every_day(), time_rules.market_open())
    schedule_function(shrts, date_rules.every_day(), time_rules.market_open(minutes=30))
    schedule_function(longs, date_rules.every_day(), time_rules.market_open(hours=1))
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

    # Take profit several times a day  
    context.profit_threshold = .05  
    context.profit_logging   = 1  
    for i in range(20, 390, 60):    # (low, high, every i minutes)  
        #break    # uncomment to deactivate  
        schedule_function(take_profit, date_rules.every_day(), time_rules.market_open(minutes=i))

    for i in range(1, 391):
        schedule_function(pvr, date_rules.every_day(), time_rules.market_open(minutes=i))

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def take_profit(context, data):    # Close some positions to take profit  
    pos = context.portfolio.positions  
    for s in pos:  
        if not data.can_trade(s): continue  
        prc = data.current(s, 'price')  
        amt = pos[s].amount  
        if (amt / abs(amt)) * ((prc / pos[s].cost_basis) - 1) > context.profit_threshold:  
            order_target(s, 0)  
            if not context.profit_logging: continue  
            log.info('take profit {} {}  cb {}  now {}'.format(  
                amt, s.symbol, '%.2f' % pos[s].cost_basis, prc)) 

def make_pipeline():
    testing_factor1 = operation_ratios.operation_revenue_growth3_month_avg.latest #operation ratios
    testing_factor2 = sentiment.sentiment_signal.latest #sentiment
    testing_factor3 = valuation_ratios.dividend_yield.latest #valuation
    testing_factor4 = cash_flow_statement.free_cash_flow.latest #cashflow

    universe = (Q1500US() &
               testing_factor1.notnull() &
               testing_factor2.notnull() &
               testing_factor3.notnull() &
               testing_factor4.notnull()
               )

    testing_factor1 = testing_factor1.rank(mask=universe, method='average')
    testing_factor2 = testing_factor2.rank(mask=universe, method='average')
    testing_factor3 = testing_factor3.rank(mask=universe, method='average')
    testing_factor4 = testing_factor4.rank(mask=universe, method='average')
    testing_factor  = testing_factor2 + testing_factor3 + testing_factor4 #testing_factor1  +

    testing_quantiles = testing_factor.quantiles(2)

    pipe = Pipeline(columns={
            'testing_factor': testing_factor,
            'shorts': testing_quantiles.eq(0),
            'longs' : testing_quantiles.eq(1)},
                   screen=universe)
    return pipe

def before_trading_start(context, data):
    context.output        = pipeline_output('pipeline')
    context.long_secs     = context.output[context.output['longs']].index
    context.short_secs    = context.output[context.output['shorts']].index
    context.security_list = context.output.index.tolist()

def close(context,data):    # close positions no longer in pipeline
    for security in context.portfolio.positions:
        if not data.can_trade(security): continue
        if security not in context.long_secs and security not in context.short_secs:
            order_target_percent(security, 0)

def shrts(context,data):
    short_weight = -0.5 / len(context.short_secs)

    for security in context.short_secs:
        if not data.can_trade(security): continue
        order_target_percent(security, short_weight)

def longs(context,data):
    long_weight = 0.5 / len(context.long_secs)
    
    for security in context.long_secs:
        if not data.can_trade(security): continue
        order_target_percent(security, long_weight)

def record_vars(context, data):
    '''    Plot variables at the end of each day.
    '''
    long_count  = 0
    short_count = 0

    # count long & short positions
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0: long_count  += 1
        if position.amount < 0: short_count += 1

    # plot counts & leverage
    record(num_long=long_count, num_short=short_count, leverage=context.account.leverage)



def pvr(context, data):
    ''' Custom chart and/or logging of profit_vs_risk returns and related information
        http://quantopian.com/posts/pvr
    '''
    import time
    from datetime import datetime as _dt
    from pytz import timezone      # Python will only do once, makes this portable.
                                   #   Move to top of algo for better efficiency.
    c = context  # Brevity is the soul of wit -- Shakespeare [for readability]
    if 'pvr' not in c:

        # For real money, you can modify this to total cash input minus any withdrawals
        manual_cash = c.portfolio.starting_cash
        time_zone   = 'US/Pacific'   # Optionally change to your own time zone for wall clock time

        c.pvr = {
            'options': {
                # # # # # # # # # #  Options  # # # # # # # # # #
                'logging'         : 0,    # Info to logging window with some new maximums
                'log_summary'     : 126,  # Summary every x days. 252/yr

                'record_pvr'      : 1,    # Profit vs Risk returns (percentage)
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value
                'record_cash'     : 0,    # Cash available
                'record_max_lvrg' : 1,    # Maximum leverage encountered
                'record_max_risk' : 0,    # Highest risk overall
                'record_shorting' : 0,    # Total value of any shorts
                'record_max_shrt' : 0,    # Max value of shorting total
                'record_cash_low' : 0,    # Any new lowest cash level
                'record_q_return' : 0,    # Quantopian returns (percentage)
                'record_pnl'      : 0,    # Profit-n-Loss
                'record_risk'     : 0,    # Risked, max cash spent or shorts beyond longs+cash
                'record_leverage' : 0,    # End of day leverage (context.account.leverage)
                # All records are end-of-day or the last data sent to chart during any day.
                # The way the chart operates, only the last value of the day will be seen.
                # # # # # # # # #  End options  # # # # # # # # #
            },
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent
            'cagr'       : 0,
            'max_lvrg'   : 0,
            'max_shrt'   : 0,
            'max_risk'   : 0,
            'days'       : 0.0,
            'date_prv'   : '',
            'date_end'   : get_environment('end').date(),
            'cash_low'   : manual_cash,
            'cash'       : manual_cash,
            'start'      : manual_cash,
            'tz'         : time_zone,
            'begin'      : time.time(),  # For run time
            'run_str'    : '{} to {}  ${}  {} {}'.format(get_environment('start').date(), get_environment('end').date(), int(manual_cash), _dt.now(timezone(time_zone)).strftime("%Y-%m-%d %H:%M"), time_zone)
        }
        if c.pvr['options']['record_pvrp']: c.pvr['options']['record_pvr'] = 0 # if pvrp is active, straight pvr is off
        if get_environment('arena') not in ['backtest', 'live']: c.pvr['options']['log_summary'] = 1 # Every day when real money
        log.info(c.pvr['run_str'])
    p = c.pvr ; o = c.pvr['options'] ; pf = c.portfolio ; pnl = pf.portfolio_value - p['start']
    def _pvr(c):
        p['cagr'] = ((pf.portfolio_value / p['start']) ** (1 / (p['days'] / 252.))) - 1
        ptype = 'PvR' if o['record_pvr'] else 'PvRp'
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (p['pvr'] / p['days']), '%.3f' % p['cagr'], '%.0f' % pf.portfolio_value, '%.0f' % pnl))
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % pnl, '%.0f' % p['max_risk'], '%.1f' % p['pvr']))
        log.info('  QRet {} PvR {} CshLw {} MxLv {} MxRisk {} MxShrt {}'.format('%.2f' % (100 * pf.returns), '%.2f' % p['pvr'], '%.0f' % p['cash_low'], '%.2f' % p['max_lvrg'], '%.0f' % p['max_risk'], '%.0f' % p['max_shrt']))
    def _minut():
        dt = get_datetime().astimezone(timezone(p['tz']))
        return str((dt.hour * 60) + dt.minute - 570).rjust(3)  # (-570 = 9:31a)
    date = get_datetime().date()
    if p['date_prv'] != date:
        p['date_prv'] = date
        p['days'] += 1.0
    do_summary = 0
    if o['log_summary'] and p['days'] % o['log_summary'] == 0 and _minut() == '100':
        do_summary = 1              # Log summary every x days
    if do_summary or date == p['date_end']:
        p['cash'] = pf.cash
    elif p['cash'] == pf.cash and not o['logging']: return  # for speed

    shorts = sum([z.amount * z.last_sale_price for s, z in pf.positions.items() if z.amount < 0])
    new_key_hi = 0                  # To trigger logging if on.
    cash       = pf.cash
    cash_dip   = int(max(0, p['start'] - cash))
    risk       = int(max(cash_dip, -shorts))

    if o['record_pvrp'] and cash < 0:   # Let negative cash ding less when portfolio is up.
        cash_dip = int(max(0, cash_dip * p['start'] / pf.portfolio_value))
        # Imagine: Start with 10, grows to 1000, goes negative to -10, should not be 200% risk.

    if int(cash) < p['cash_low']:             # New cash low
        new_key_hi = 1
        p['cash_low'] = int(cash)             # Lowest cash level hit
        if o['record_cash_low']: record(CashLow = p['cash_low'])

    if c.account.leverage > p['max_lvrg']:
        new_key_hi = 1
        p['max_lvrg'] = c.account.leverage    # Maximum intraday leverage
        if o['record_max_lvrg']: record(MxLv    = p['max_lvrg'])

    if shorts < p['max_shrt']:
        new_key_hi = 1
        p['max_shrt'] = shorts                # Maximum shorts value
        if o['record_max_shrt']: record(MxShrt  = p['max_shrt'])

    if risk > p['max_risk']:
        new_key_hi = 1
        p['max_risk'] = risk                  # Highest risk overall
        if o['record_max_risk']:  record(MxRisk = p['max_risk'])

    # Profit_vs_Risk returns based on max amount actually invested, long or short
    if p['max_risk'] != 0: # Avoid zero-divide
        p['pvr'] = 100 * pnl / p['max_risk']
        ptype = 'PvRp' if o['record_pvrp'] else 'PvR'
        if o['record_pvr'] or o['record_pvrp']: record(**{ptype: p['pvr']})

    if o['record_shorting']: record(Shorts = shorts)             # Shorts value as a positve
    if o['record_leverage']: record(Lv     = c.account.leverage) # Leverage
    if o['record_cash']    : record(Cash   = cash)               # Cash
    if o['record_risk']    : record(Risk   = risk)  # Amount in play, maximum of shorts or cash used
    if o['record_q_return']: record(QRet   = 100 * pf.returns)
    if o['record_pnl']     : record(PnL    = pnl)                # Profit|Loss

    if o['logging'] and new_key_hi:
        log.info('{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(),
            ' Lv '     + '%.1f' % c.account.leverage,
            ' MxLv '   + '%.2f' % p['max_lvrg'],
            ' QRet '   + '%.1f' % (100 * pf.returns),
            ' PvR '    + '%.1f' % p['pvr'],
            ' PnL '    + '%.0f' % pnl,
            ' Cash '   + '%.0f' % cash,
            ' CshLw '  + '%.0f' % p['cash_low'],
            ' Shrt '   + '%.0f' % shorts,
            ' MxShrt ' + '%.0f' % p['max_shrt'],
            ' Risk '   + '%.0f' % risk,
            ' MxRisk ' + '%.0f' % p['max_risk']
        ))
    if do_summary: _pvr(c)
    if get_datetime() == get_environment('end'):   # Summary at end of run
        _pvr(c) ; elapsed = (time.time() - p['begin']) / 60  # minutes
        log.info( '{}\nRuntime {} hr {} min'.format(p['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))