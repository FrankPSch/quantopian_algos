'''
    Zero-Beta targeting example.
    Automatically adjust proportions of spy and tlt to hold Beta to around 0.0 or beta_target.
    c.beta_limit is one strictness adjustment, there are others.
    In terms of *effect* on Beta generally:
      - Longs (many) tend to be like SPY (increase)
      - Short often acts similar to TLT here (decrease)
'''
import pandas as pd

def initialize(context):
    c = context
    c.spy          = sid(8554)
    c.tlt          = sid(23921)
    c.beta         = 1.0    # Assumed starting beta
    c.beta_target  = 0.0    # Target any Beta you wish
    c.beta_limit   =  .01   # Pos/neg threshold, balance only outside of this either side of target
    c.spy_limit_hi =  .95   # Max ratio of spy to portfolio
    c.spy_limit_lo = 1.0 - c.spy_limit_hi
    c.beta_df      = pd.DataFrame([], columns=['pf', 'spy'])
    schedule_function(balance, date_rules.week_start(), time_rules.market_open())

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def balance(context, data):
    c = context
    if not c.portfolio.positions:   # Initial positions to start, reusing spy_limit
        order_target_percent(c.spy, c.spy_limit_hi)
        order_target_percent(c.tlt, c.spy_limit_lo)
        return

    beta = calc_beta(c)
    bzat = beta - c.beta_target     # bzat is beta-zero adjusted for target
    if -c.beta_limit < bzat < c.beta_limit:     # Skip if inside boundaries
        return

    # -------- Adjust positions to move toward target Beta --------
    pos_val   = c.portfolio.positions_value
    spy_val   = c.portfolio.positions[c.spy].amount * data.current(c.spy, 'price')
    spy_ratio = spy_val / pos_val

    # Reduce spy & increase tlt or visa-versa
    # The further away from target Beta, the stronger the adjustment.
    # https://www.quantopian.com/posts/scaling for explanation of next line ...
    temperance = scale(abs(bzat), 0, .30, .35, .80) # Not straight Beta, a portion of it.
    adjust     = max(c.spy_limit_lo, spy_ratio - (bzat * temperance))
    adjust     = min(c.spy_limit_hi, adjust)  # spy ratio no higher than spy_limit_hi
    log.info('b{} spy {} to {}'.format('%.2f' % beta, '%.2f' % spy_ratio, '%.2f' % adjust))
    order_target_percent(c.spy, adjust)
    order_target_percent(c.tlt, 1.0 - adjust) # Remainder for tlt

def before_trading_start(context, data):
    c = context
    c.beta_df = c.beta_df.append({    # Beta calc prep
            'pf' : c.portfolio.portfolio_value,
            'spy': data.current(c.spy, 'price')}, ignore_index=True)
    c.beta_df['spy_chg'] = c.beta_df.spy.pct_change()
    c.beta_df[ 'pf_chg'] = c.beta_df.pf .pct_change()
    c.beta_df            = c.beta_df.ix[-252:]    # trim to one year

def calc_beta(c):   # Calculate current Beta value
    if len(c.beta_df.spy.values) < 3: return c.beta
    beta = c.beta_df.pf_chg.cov(c.beta_df.spy_chg) / c.beta_df.spy_chg.var()
    record(beta_calculated = beta)
    return beta

def scale(wild, a_lo, a_hi, b_lo, b_hi):
    ''' Based on wild value relative to a_lo_hi range,
          return its analog within b_hi_lo, with min b_lo and max b_hi
    '''
    return min(b_hi, max(b_lo, (b_hi * (wild - a_lo)) / (a_hi - a_lo)))

def handle_data(context, data):
    return
    pvr(context, data)

def pvr(context, data):
    ''' Custom chart and/or log of profit_vs_risk returns and related information
    '''
    # # # # # # # # # #  Options  # # # # # # # # # #
    record_pvr      = 0            # Profit vs Risk returns (percentage)
    record_pvrp     = 1            # PvR (p)roportional neg cash vs portfolio value
    record_cash     = 1            # Cash available
    record_max_lvrg = 1            # Maximum leverage encountered
    record_risk_hi  = 0            # Highest risk overall
    record_shorting = 1            # Total value of any shorts
    record_cash_low = 0            # Any new lowest cash level
    record_q_return = 0            # Quantopian returns (percentage)
    record_pnl      = 0            # Profit-n-Loss
    record_risk     = 0            # Risked, max cash spent or shorts beyond longs+cash
    record_leverage = 0            # Leverage (context.account.leverage)
    record_overshrt = 0            # Shorts beyond longs+cash
    logging         = 0            # Also to logging window conditionally (1) or not (0)
    if record_pvrp: record_pvr = 0 # if pvrp is active, straight pvr is off

    import time
    from datetime import datetime
    from pytz import timezone   # Python will only do once, makes this portable.
                                #   Move to top of algo for better efficiency.
    c = context  # Brevity is the soul of wit -- Shakespeare [for efficiency, readability]
    if 'pvr' not in c:
        date_strt = get_environment('start').date()
        date_end  = get_environment('end').date()
        cash_low  = c.portfolio.starting_cash
        c.pvr = {
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent
            'max_lvrg'   : 0,
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

    def _pvr_(c):
        ptype = 'PvR' if record_pvr else 'PvRp'
        log.info('{} {} %/day     {}'.format(ptype, '%.4f' % (c.pvr['pvr'] / c.pvr['days']), c.pvr['run_str']))
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % (c.portfolio.portfolio_value - c.pvr['start']), '%.0f' % c.pvr['risk_hi'], '%.1f' % c.pvr['pvr']))
        log.info('  QRet {} PvR {} CshLw {} MxLv {} RskHi {} Shrts {}'.format('%.2f' % q_rtrn, '%.2f' % c.pvr['pvr'], '%.0f' % c.pvr['cash_low'], '%.2f' % c.pvr['max_lvrg'], '%.0f' % c.pvr['risk_hi'], '%.0f' % shorts))

    def _minut():   # To preface each line with the minute of the day.
        dt = get_datetime().astimezone(timezone('US/Eastern'))
        minute = (dt.hour * 60) + dt.minute - 570  # (-570 = 9:31a)
        return str(minute).rjust(3)

    date = get_datetime().date()
    if c.pvr['date_prv'] != date: c.pvr['days'] += 1.0
    do_summary = 0
    if c.pvr['log_summary'] and c.pvr['days'] % c.pvr['log_summary'] == 0 and _minut() == '100':
        do_summary = 1 # Log summary every x days
    c.pvr['date_prv'] = date    # next line for speed
    if c.pvr['cash'] == c.portfolio.cash and not do_summary and date != c.pvr['date_end']: return
    c.pvr['cash'] = c.portfolio.cash

    longs         = 0                      # Longs  value
    shorts        = 0                      # Shorts value
    overshorts    = 0                      # Shorts value beyond longs plus cash
    new_cash_low  = 0                      # To trigger logging in cash_low case
    new_risk_hi   = 0
    q_rtrn        = 100 * (c.portfolio.portfolio_value - c.pvr['start']) / c.pvr['start']
    cash          = c.portfolio.cash
    cash_dip      = int(max(0, c.pvr['start'] - cash))
    if record_pvrp and cash < 0:    # Let negative cash ding less when portfolio is up.
        cash_dip = int(max(0, c.pvr['start'] - cash * c.pvr['start'] / c.portfolio.portfolio_value))
        # Imagine: Start with 10, grows to 1000, goes negative to -10, shud not be 200% risk.

    if int(cash) < c.pvr['cash_low']:                # New cash low
        new_cash_low = 1
        c.pvr['cash_low']   = int(cash)
        if record_cash_low:
            record(CashLow = int(c.pvr['cash_low'])) # Lowest cash level hit

    if record_max_lvrg:
        if c.account.leverage > c.pvr['max_lvrg']:
            c.pvr['max_lvrg'] = c.account.leverage
            record(MaxLv = c.pvr['max_lvrg'])        # Maximum leverage
            #log.info('Max Lvrg {}'.format('%.2f' % c.pvr['max_lvrg']))

    for p in c.portfolio.positions:
        if not data.can_trade(p): continue
        shrs = c.portfolio.positions[p].amount
        if   shrs < 0: shorts += int(abs(shrs * data.current(p, 'price')))
        elif shrs > 0: longs  += int(    shrs * data.current(p, 'price'))

    if shorts > longs + cash: overshorts = shorts             # Shorts when too high
    if record_overshrt: record(OvrShrt = overshorts)          # Shorts beyond payable
    if record_shorting: record(Shorts  = shorts)              # Shorts value as a positve
    if record_leverage: record(Lvrg = c.account.leverage)     # Leverage
    if record_cash:     record(Cash = int(cash))              # Cash

    risk = int(max(cash_dip,   shorts))
    if record_risk: record(Risk = risk)       # Amount in play, maximum of shorts or cash used

    if risk > c.pvr['risk_hi']:
        c.pvr['risk_hi'] = risk
        new_risk_hi = 1
        if record_risk_hi:
            record(RiskHi = c.pvr['risk_hi']) # Highest risk overall

    if record_pnl:                            # "Profit and Loss" in dollars
        record(PnL = min(0, c.pvr['cash_low']) + context.portfolio.pnl )

    if record_pvr or record_pvrp: # Profit_vs_Risk returns based on max amount actually spent (risk high)
        if c.pvr['risk_hi'] != 0: # Avoid zero-divide
            c.pvr['pvr'] = 100 * (c.portfolio.portfolio_value - c.pvr['start']) / c.pvr['risk_hi']
            ptype = 'PvRp' if record_pvrp else 'PvR'
            record(**{ptype: c.pvr['pvr']})

    if record_q_return:
        record(QRet = q_rtrn)                 # Quantopian returns to compare to pvr returns curve

    if logging:
        if new_risk_hi or new_cash_low:
            qret    = ' QRet '   + '%.1f' % q_rtrn
            lv      = ' Lv '     + '%.1f' % c.account.leverage if record_leverage else ''
            pvr     = ' PvR '    + '%.1f' % c.pvr['pvr']       if record_pvr      else ''
            pnl     = ' PnL '    + '%.0f' % c.portfolio.pnl    if record_pnl      else ''
            csh     = ' Cash '   + '%.0f' % cash               if record_cash     else ''
            shrt    = ' Shrt '   + '%.0f' % shorts             if record_shorting else ''
            ovrshrt = ' Shrt '   + '%.0f' % overshorts         if record_overshrt else ''
            risk    = ' Risk '   + '%.0f' % risk               if record_risk     else ''
            mxlv    = ' MaxLv '  + '%.2f' % c.pvr['max_lvrg']  if record_max_lvrg else ''
            csh_lw  = ' CshLw '  + '%.0f' % c.pvr['cash_low']  if record_cash_low else ''
            rsk_hi  = ' RskHi '  + '%.0f' % c.pvr['risk_hi']   if record_risk_hi  else ''
            log.info('{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(), lv, mxlv, qret, pvr, pnl, csh, csh_lw, shrt, ovrshrt, risk, rsk_hi))
    if do_summary: _pvr_(c)
    if date == c.pvr['date_end']:        # Summary on last day once.
        if 'pvr_summary_done' not in c: c.pvr_summary_done = 0
        if not c.pvr_summary_done:
            _pvr_(c)
            elapsed = (time.time() - c.pvr['begin']) / 60  # minutes
            log.info( '\nRuntime {} hr {} min    End: {} {}'.format(
                int(elapsed / 60), '%.1f' % (elapsed % 60),
                datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M"), 'US/Eastern'))
            c.pvr_summary_done = 1