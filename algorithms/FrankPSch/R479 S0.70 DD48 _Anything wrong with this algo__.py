'''
'''
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.data.eventvestor import EarningsCalendar
from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings
)
from pytz import timezone as tz

def make_pipeline():
    minprice = USEquityPricing.close.latest > 5
    pipe = Pipeline(screen=Q1500US() & minprice)
    pipe.add(BusinessDaysSincePreviousEarnings(), 'prev_earn')
    return pipe

def initialize(context):
    context.day_done = 0    # When all close are done, in trade_stocks() set to 1.
    #set_commission(commission.PerShare(cost=0.005, min_trade_cost=1))
    #set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))  # used in the fund
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=.1))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    context.stocks = None
    context.tradeables = {}
    context.days = 0
    context.portfolio_vals = []
    context.tax_adjusted_vals = []
    context.starting_bal = context.portfolio.portfolio_value

    schedule_function(close_all,        date_rules.every_day(), time_rules.market_open(minutes=25))
    #schedule_function(trade_stocks,     date_rules.every_day(), time_rules.market_open(minutes=35))
    #schedule_function(log_num_positions,date_rules.every_day(), time_rules.market_open(minutes=60))
    schedule_function(calc_after_tax_pf,date_rules.every_day(), time_rules.market_close(minutes=1))
    schedule_function(graph_returns,    date_rules.every_day(), time_rules.market_close(minutes=1))

    attach_pipeline(make_pipeline(), "Q1500")

    # Maybe not the most efficient, looking every minute after a point, for all close to be done, just one quick way.
    for i in range(35, 391):
        schedule_function(trade_stocks, date_rules.every_day(), time_rules.market_open(minutes=i))

    for i in range(1, 391):
        schedule_function(pvr, date_rules.every_day(), time_rules.market_open(minutes=i))

def graph_returns(context, data):
    if context.days <= 0:
        return
    year_growth = (context.portfolio.portfolio_value/float(context.portfolio_vals[-1])) - 1
    current_bal = (year_growth*context.tax_adjusted_vals[-1]) + context.tax_adjusted_vals[-1]
    running_growth = ((current_bal/float(context.starting_bal)) - 1)*100
    record(tax_adj_returns=running_growth)

def calc_after_tax_pf(context, data):
    if context.days % 252 == 0:
        context.portfolio_vals.append(context.portfolio.portfolio_value)
        if len(context.portfolio_vals) > 1:
            this_year_grow_rate = context.portfolio_vals[-1]/float(context.portfolio_vals[-2]) - 1
            if this_year_grow_rate > 0:
                this_year_grow_rate = 0.6*this_year_grow_rate
            returns = context.tax_adjusted_vals[-1]*this_year_grow_rate
            end_year_balance = context.tax_adjusted_vals[-1]+returns
            context.tax_adjusted_vals.append(end_year_balance)
        else:
            context.tax_adjusted_vals.append(context.portfolio.portfolio_value)
    context.days += 1

def close_all(context, data):
    os = get_open_orders()

    for ol in os.values():
        for o in ol:
            cancel_order(o)

    for sid in context.portfolio.positions:
        order_target(sid, 0)

def before_trading_start(context, data):
    context.day_done = 0    # reset for today
    context.tradeables = {}
    context.screener = pipeline_output("Q1500")
    context.stocks = context.screener[context.screener['prev_earn'] == 0].index

def trade_stocks(context, data):
    if len(context.stocks) == 0:
        return

    if context.day_done:
        return

    oos = get_open_orders()
    if oos:
        for s in oos:
            for o in oos[s]:
                log.info('{}  {} of {} unfilled'.format(s.symbol, o.amount - o.filled, o.amount))
        return
    else:
        # Chart minute of the day trading
        record(minute = minut())

        context.day_done = 1

    closes = data.history(context.stocks, 'close', 2, '1d')
    prices = data.current(context.stocks, 'price')
    #positive_gaps = 0
    for stock in context.stocks:
        night_gap = prices[stock]/closes[stock][-2] - 1
        if night_gap > .04:
            context.tradeables[stock] = night_gap
    num_stocks = len(context.tradeables.keys())
    if num_stocks == 0:
        return
    weight = 1./num_stocks
    for key in context.tradeables.keys():
        order_target_percent(key, weight)

def log_num_positions(context, data):
    num_pos = 0;
    for position in context.portfolio.positions.itervalues():
        if position.amount != 0:
            num_pos += 1
    if num_pos > 0:
        log.info('num_pos=[%s]' % str(num_pos))

def minut():           # Minute of the trading day
    dt_ = get_datetime().astimezone(tz('US/Eastern'))
    return (dt_.hour * 60) + dt_.minute - 570  # (-570 = 9:31a)

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

                'record_pvr'      : 0,    # Profit vs Risk returns (percentage)
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value
                'record_cash'     : 0,    # Cash available
                'record_max_lvrg' : 1,    # Maximum leverage encountered
                'record_max_risk' : 0,    # Highest risk overall
                'record_shorting' : 0,    # Total value of any shorts
                'record_max_shrt' : 1,    # Max value of shorting total
                'record_cash_low' : 1,    # Any new lowest cash level
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