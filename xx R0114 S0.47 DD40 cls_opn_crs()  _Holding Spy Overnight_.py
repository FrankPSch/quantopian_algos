# https://www.quantopian.com/posts/holding-spy-overnight
spy = sid(8554)
 
def initialize(context):
    # Fill orders with no volume check? Are we 100% sure?
    #set_slippage(slippage.FixedSlippage(spread=0.0))
    # Fill orders with no volume limit, set outrageously high for testing only. 
    # With starting capital turned down, nothing for this to chew on
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=   1e6   ))
    
    minut = 120
    schedule_function(close_pos, date_rules.every_day(), time_rules.market_open (minutes=minut))
    schedule_function(open_pos,  date_rules.every_day(), time_rules.market_close(minutes=minut))
    
    # end is non-inclusive, at 390 will not run on 390 the last minute of the day.
    #   Set to 391 to run 390 and see incomplete orders. How is that possible?
    for i in range(1, 390, 1):  # start, end, every i
        
      break    # comment out or remove to activate
    
      schedule_function(profit_or_stop, date_rules.every_day(), time_rules.market_open(minutes=i))

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def profit_or_stop(context, data):
    for s in context.portfolio.positions:
        cb  = context.portfolio.positions[s].cost_basis  
        prc = data.current(s, 'price')
        if   prc > cb * 1.002: 
            if orders_just_so(context, s):
                continue
            order_target(s, 0)
            #log.info('  profit prc {}  cb {}  {}'.format(prc, cb, '%.3f' % (prc / cb)))
        elif prc < cb *  .95: 
            if orders_just_so(context, s):
                continue
            order_target(s, 0)
            log.info('stoploss prc {}  cb {}  {}'.format(prc, cb, '%.3f' % (prc / cb)))
    
def open_pos(context, data):    # pos stands for position
    oo  = get_open_orders(spy)
    for o in oo:
        cancel_order(o.id)
    order_target_percent(spy, 1.0)
    #log.info('open')
    
def close_pos(context, data):
    oo  = get_open_orders(spy)
    for o in oo:
        cancel_order(o.id)
    order_target(spy, 0)
    #log.info('close')
    
def orders_just_so(context, s):
    oo  = get_open_orders(s)
    for o in oo:                # returns are assuming only one order
        if cls_opn_crs(context, s, o) in [1, 3]:    # Opening
            cancel_order(o.id)          # Cancel opening order
            return 0                    # To allow the close
        else:
            return 1    # Already closing, signal to skip adding another close order
    
def cls_opn_crs(c, s, o):
    # https://www.quantopian.com/posts/order-state-on-partial-fills-close-open-or-crossover
    if o.stop or o.limit: return 0    # ... assuming you're using stop, limit only to close
    if c.portfolio.positions[s].amount * o.amount < 0:   # close or x
        if abs(c.portfolio.positions[s].amount) < abs(o.amount - o.filled):
            if abs(c.portfolio.positions[s].amount) - abs(o.filled) < 0:
                  return 3  # crossed now opening
            else: return 2  # cross closing
        else:     return 0  # close
    else:         return 1  # open