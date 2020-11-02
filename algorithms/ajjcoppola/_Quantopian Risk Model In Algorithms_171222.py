import pandas as pd

import quantopian.algorithm as algo
import quantopian.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import builtin, Fundamentals, psychsignal
from quantopian.pipeline.factors import SimpleBeta, AverageDollarVolume, RollingLinearRegressionOfReturns
from quantopian.pipeline.factors.fundamentals import MarketCap
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.experimental import QTradableStocksUS, risk_loading_pipeline
import datetime
import time
# Algorithm Parameters
# --------------------
UNIVERSE_SIZE = 1000
LIQUIDITY_LOOKBACK_LENGTH = 100

MINUTES_AFTER_OPEN_TO_TRADE = 68

MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.01  # 1%
MAX_LONG_POSITION_SIZE = 0.01  # 1%
BETA_LIM = 0.05


def initialize(context):
    schedule_function(func=pvr,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(hours=0,minutes=1))

    # Universe Selection
    # ------------------
    base_universe = QTradableStocksUS()

    # From what remains, each month, take the top UNIVERSE_SIZE stocks by average dollar
    # volume traded.
    monthly_top_volume = (
        AverageDollarVolume(window_length=LIQUIDITY_LOOKBACK_LENGTH)
        .top(UNIVERSE_SIZE, mask=base_universe)
        .downsample('week_start')
    )
    # The final universe is the monthly top volume &-ed with the original base universe.
    # &-ing these is necessary because the top volume universe is calculated at the start 
    # of each month, and an asset might fall out of the base universe during that month.
    universe = monthly_top_volume & base_universe

    # Alpha Generation
    # ----------------
    # Compute Z-scores of free cash flow yield and earnings yield. 
    # Both of these are fundamental value measures.
    fcf_zscore = Fundamentals.fcf_yield.latest.zscore(mask=universe)
    yield_zscore = Fundamentals.earning_yield.latest.zscore(mask=universe)
    sentiment_zscore = psychsignal.stocktwits.bull_minus_bear.latest.zscore(mask=universe)
    
    # Alpha Combination
    # -----------------
    # Assign every asset a combined rank and center the values at 0.
    # For UNIVERSE_SIZE=500, the range of values should be roughly -250 to 250.
    #combined_alpha = (1.2*fcf_zscore + yield_zscore).rank().demean()
    combined_alpha = (5.1*fcf_zscore + yield_zscore).zscore()
    
    rho =0.7
    beta = rho*SimpleBeta(target=sid(8554),regression_length=260,) + (1.0-rho)*1.0

    # beta = 0.66*RollingLinearRegressionOfReturns(
    #                 target=sid(8554),
    #                 returns_length=5,
    #                 regression_length=260,
    #                 mask=combined_alpha.notnull() & Sector().notnull()
    #                 ).beta + 0.33*1.0

    # Schedule Tasks
    # --------------
    # Create and register a pipeline computing our combined alpha and a sector
    # code for every stock in our universe. We'll use these values in our 
    # optimization below.
    pipe = Pipeline(
        columns={
            'alpha': combined_alpha,
            'sector': Sector(),
            'sentiment': sentiment_zscore,
            'beta': beta,
        },
        # combined_alpha will be NaN for all stocks not in our universe,
        # but we also want to make sure that we have a sector code for everything
        # we trade.
        screen=combined_alpha.notnull() & Sector().notnull() & beta.notnull(),
    )
    
    # Multiple pipelines can be used in a single algorithm.
    algo.attach_pipeline(pipe, 'pipe')
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    

    # Schedule a function, 'do_portfolio_construction', to run twice a week
    # ten minutes after market open.
    algo.schedule_function(
        do_portfolio_construction,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open(minutes=MINUTES_AFTER_OPEN_TO_TRADE),
        half_days=False,
    )


def before_trading_start(context, data):
    # Call pipeline_output in before_trading_start so that pipeline
    # computations happen in the 5 minute timeout of BTS instead of the 1
    # minute timeout of handle_data/scheduled functions.
    context.pipeline_data = algo.pipeline_output('pipe')
    context.risk_loading_pipeline = algo.pipeline_output('risk_loading_pipeline')


# Portfolio Construction
# ----------------------
def do_portfolio_construction(context, data):
    pipeline_data = context.pipeline_data

    # Objective
    # ---------
    # For our objective, we simply use our naive ranks as an alpha coefficient
    # and try to maximize that alpha.
    # 
    # This is a **very** naive model. Since our alphas are so widely spread out,
    # we should expect to always allocate the maximum amount of long/short
    # capital to assets with high/low ranks.
    #
    # A more sophisticated model would apply some re-scaling here to try to generate
    # more meaningful predictions of future returns.
    objective = opt.MaximizeAlpha(pipeline_data.alpha)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    # Constrain individual position size to no more than a fixed percentage 
    # of our portfolio. Because our alphas are so widely distributed, we 
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to 
    # long and short positions.
    market_neutral = opt.DollarNeutral()
    
    # Constrain beta-to-SPY to remain under the contest criteria.
    beta_neutral = opt.FactorExposure(
        pipeline_data[['beta']],
        min_exposures={'beta': -BETA_LIM}, #-0.05},
        max_exposures={'beta': BETA_LIM }, #0.05},
    )

    # Constrain exposure to common sector and style risk 
    # factors, using the latest default values. At the time
    # of writing, those are +-0.18 for sector and +-0.36 for 
    # style.
    constrain_sector_style_risk = opt.experimental.RiskModelExposure(
        context.risk_loading_pipeline,
        version=opt.Newest,
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            constrain_sector_style_risk,
            beta_neutral,
        ],
    )
    
# PvR Code: Author:Blue -------------------------------------------------------------
# https://www.quantopian.com/posts/pvr-profit-vs-risk#569784bda73e9bf2b7000180
# Version 05/05/2017
def pvr(context, data):  
    ''' Custom chart and/or logging of profit_vs_risk returns and related information  
    '''  
    import time  
    from datetime import datetime  
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
                'logging'         : 1,    # Info to logging window with some new maximums  
                'log_summary'     : 22, #Q #H 126,  # Summary every x days. 252/yr

                'record_pvr'      : 1,    # Profit vs Risk returns (percentage)  
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value  
                'record_pnl'      : 0,    # PnL  
                'record_cash'     : 0,    # Cash available  
                'record_max_lvrg' : 0,    # Maximum leverage encountered  
                'record_risk_hi'  : 0,    # Highest risk overall  
                'record_shorting' : 0,    # Total value of any shorts  
                'record_max_shrt' : 0,    # Max value of shorting total  
                'record_cash_low' : 0,    # Any new lowest cash level  
                'record_q_return' : 1,    # Profit-n-Loss  
                'record_risk'     : 0,    # Risked, max cash spent or shorts beyond longs+cash  
                'record_leverage' : 1,    # End of day leverage (context.account.leverage) 
                'record_numshorts': 1,   # num short assets  
                'record_numlongs' : 1,   # num long assets  

                # All records are end-of-day or the last data sent to chart during any day.  
                # The way the chart operates, only the last value of the day will be seen.  
                # # # # # # # # #  End options  # # # # # # # # #  
            },  
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent  
            'cagr'       : 0,  
            'max_lvrg'   : 0,  
            'max_shrt'   : 0,  
            'risk_hi'    : 0,  
            'days'       : 0.0,  
            'date_prv'   : '',  
            'date_end'   : get_environment('end').date(),  
            'cash_low'   : manual_cash,  
            'cash'       : manual_cash,  
            'start'      : manual_cash,  
            'tz'         : time_zone,  
            'begin'      : time.time(),  # For run time  
            'run_str'    : '{} to {}  ${}  {} {}'.format(get_environment('start').date(), get_environment('end').date(), int(manual_cash), datetime.now(timezone(time_zone)).strftime("%Y-%m-%d %H:%M"), time_zone)  
        }  
        if c.pvr['options']['record_pvrp']: c.pvr['options']['record_pvr'] = 0 # if pvrp is active, straight pvr is off  
        if get_environment('arena') not in ['backtest', 'live']: c.pvr['options']['log_summary'] = 1 # Every day when real money  
        log.info(c.pvr['run_str'])  
    p = c.pvr ; o = c.pvr['options'] ; pf = c.portfolio ; pnl = pf.portfolio_value - p['start']  
    def _pvr(c):  
        p['cagr'] = ((pf.portfolio_value / p['start']) ** (1 / (p['days'] / 252.))) - 1  
        ptype = 'PvR' if o['record_pvr'] else 'PvRp'  
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (p['pvr'] / p['days']), '%.3f' % p['cagr'], '%.0f' % pf.portfolio_value, '%.0f' % pnl))  
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % pnl, '%.0f' % p['risk_hi'], '%.1f' % p['pvr']))  
        log.info('  QRet {} PvR {} CshLw {} MxLv {} RskHi {} MxShrt {}'.format('%.2f' % (100 * pf.returns), '%.2f' % p['pvr'], '%.0f' % p['cash_low'], '%.2f' % p['max_lvrg'], '%.0f' % p['risk_hi'], '%.0f' % p['max_shrt']))  
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
    num_longs  = len([s for s, z in pf.positions.items() if z.amount > 0])      
    num_shorts = len([s for s, z in pf.positions.items() if z.amount < 0])   

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
        if o['record_max_lvrg']: record(MaxLv   = p['max_lvrg'])

    if shorts < p['max_shrt']:  
        new_key_hi = 1  
        p['max_shrt'] = shorts                # Maximum shorts value  
        if o['record_max_shrt']: record(MxShrt  = p['max_shrt'])

    if risk > p['risk_hi']:  
        new_key_hi = 1  
        p['risk_hi'] = risk                   # Highest risk overall  
        if o['record_risk_hi']:  record(RiskHi  = p['risk_hi'])

    # Profit_vs_Risk returns based on max amount actually invested, long or short  
    if p['risk_hi'] != 0: # Avoid zero-divide  
        p['pvr'] = 100 * pnl / p['risk_hi']  
        ptype = 'PvRp' if o['record_pvrp'] else 'PvR'  
        if o['record_pvr'] or o['record_pvrp']: record(**{ptype: p['pvr']})

    if o['record_shorting']: record(Shorts = shorts)             # Shorts value as a positve  
    if o['record_leverage']: record(Lvrg   = c.account.leverage) # Leverage  
    if o['record_cash']    : record(Cash   = cash)               # Cash  
    if o['record_risk']    : record(Risk   = risk)  # Amount in play, maximum of shorts or cash used  
    if o['record_q_return']: record(QRet   = 100 * pf.returns)  
    if o['record_pnl']     : record(PnL    = pnl)                # Profit|Loss
    if o['record_numshorts']: record(numShorts    = num_shorts) # Num short assets
    if o['record_numlongs']    : record(numLongs     = num_longs)  # Num long assets


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
            ' RskHi '  + '%.0f' % p['risk_hi']  
        ))  
    if do_summary: _pvr(c)  
    if get_datetime() == get_environment('end'):   # Summary at end of run  
        _pvr(c) ; elapsed = (time.time() - p['begin']) / 60  # minutes  
        log.info( '{}\nRuntime {} hr {} min'.format(p['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))