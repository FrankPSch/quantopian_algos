'''
    Zero-Beta targeting example.
    Automatically adjust proportions of spy and tlt to hold Beta to around 0.0 or beta_target.
    c.beta_limit is one strictness adjustment, there are others.
    In terms of *effect* on Beta generally:
      - Longs (many) tend to be like SPY (increase)
      - Short often acts similar to TLT here (decrease)
      
      Standalone version - Just copy the schedule_function at your desired interval and the beta_balance function into any algorithm and specify the % of your portfolio that the beta hedge will occupy. You will need to lower your algorithm's leverage accordingly. Not yet tested with algorithms that use SPY and TLT already!
'''

def initialize(context):
    #Backtester settings
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    #Balance Beta
    schedule_function(beta_balance, date_rules.week_start(), time_rules.market_open())
    
def handle_data(context, data):
    pass

def before_trading_start(context, data):
    record(lev = context.account.leverage)
    pass



def beta_balance(context, data):
    ########### Variables ###################################
    spy              = sid(8554)
    tlt              = sid(23921)
    beta_lev         = 1.0    # Percentage of portfolio you wish to act as a beta hedge
    beta             = 1.0    # Assumed starting beta
    beta_target      = 0.0    # Target any Beta you wish
    beta_limit       =  .01   # Pos/neg threshold, balance only outside of this either side of target
    record_beta      = True   # Graph beta over time
    record_spy_ratio = True   # Graph the percentage of beta adjust "spy"
    log_info         = True   #log of adjustments to beta
    ##########################################################
    import pandas as panda_var # Python will only do once, makes this portable. Move to top of algo for better efficiency.
    import numpy as numpy_var  # Python will only do once, makes this portable. Move to top of algo for better efficiency.
    
    try: context.beta_df
    except: context.beta_df = panda_var.DataFrame([], columns=['pf', 'spy'])
    
    if get_open_orders(spy) or not data.can_trade(spy) or get_open_orders(tlt) or not data.can_trade(tlt): 
        log.error("Beta Stocks Not Tradeable")
        return
    
    spy_price = data.current(spy, 'price')
    spy_amt = context.portfolio.positions[spy].amount
    tlt_price = data.current(tlt, 'price')
    tlt_amt = context.portfolio.positions[tlt].amount
    port_val   = context.portfolio.portfolio_value
    spy_val   = spy_amt * spy_price
    spy_ratio = spy_val / port_val
    
    context.beta_df = context.beta_df.append({    # Beta calc prep
            'pf' : port_val,
            'spy': spy_price}, ignore_index=True)
    context.beta_df['spy_chg'] = context.beta_df.spy.pct_change()
    context.beta_df[ 'pf_chg'] = context.beta_df.pf.pct_change()
    context.beta_df            = context.beta_df.ix[-252:]    # trim to one year
    
    

    if not context.portfolio.positions:   # Initial positions to start, only if none exist
        spy_goal_val = (port_val * beta_lev)
        spy_goal_shares = numpy_var.floor(spy_goal_val / spy_price)
        spy_shares = (spy_goal_shares - spy_amt)
        order(spy,  spy_shares)
        return

    if len(context.beta_df.spy.values) < 3: return
    
    beta = context.beta_df.pf_chg.cov(context.beta_df.spy_chg) / context.beta_df.spy_chg.var()
    
    if record_beta:
        record(beta_calculated = beta)
        
    bzat = beta - beta_target     # bzat is beta-zero adjusted for target
    if -beta_limit < bzat < beta_limit:     # Skip if inside boundaries
        return

    # -------- Adjust positions to move toward target Beta --------
    # Reduce spy & increase tlt or visa-versa
    # The further away from target Beta, the stronger the adjustment.
    # https://www.quantopian.com/posts/scaling for explanation of next line ...
    def scale(wild, a_lo, a_hi, b_lo, b_hi):
        ''' Based on wild value relative to a_lo_hi range,
            return its analog within b_hi_lo, with min b_lo and max b_hi
        '''
        return min(b_hi, max(b_lo, (b_hi * (wild - a_lo)) / (a_hi - a_lo)))  
    
    temperance = scale(abs(bzat), 0, .30, .35, .80) # Not straight Beta, a portion of it.
    adjust     = min(beta_lev,max(0.0, spy_ratio - (bzat * temperance))) # spy ratio no higher than spy_limit_hi or lower than 0
    
    if log_info:
        log.info('beta {} spy {} to {}'.format('%.2f' % beta, '%.2f' % spy_ratio, '%.2f' % adjust))
    
    if record_spy_ratio:
        record(Beta_spy_ratio = adjust/beta_lev)
            
    spy_goal_val = (port_val * adjust)
    spy_goal_shares = numpy_var.floor(spy_goal_val / spy_price)
    spy_shares = (spy_goal_shares - spy_amt)
    
    tlt_goal_val = (port_val * (beta_lev - adjust))
    tlt_goal_shares = numpy_var.floor(tlt_goal_val / tlt_price)
    tlt_shares = (tlt_goal_shares - tlt_amt)
                
    order(spy,  spy_shares)
    order(tlt,  tlt_shares)