# https://www.quantopian.com/posts/google-search-terms-predict-market-movements
import numpy as np
import datetime
from collections import deque as window

delta_t = 6

def initialize(context):
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    context.query = 'terrorism'
                fetch_csv('https://gist.githubusercontent.com/gordondavidf/ee6bd0998ac727b1dbe2/raw/1f564dcb60200cb16860ad18e9698006c5b0dd8f/Terrorism',
              date_format='%d/%m/%Y',
              symbol='terrorism',
                 )
    context.order_size = 50
    context.security = symbol('GLD') # Gold Stocks
    context.window = window(maxlen=delta_t)
    set_slippage(slippage.FixedSlippage(spread=0.0))

def handle_data(context, data):
    c = context
      
   
    if (data[c.security].datetime.weekday()) == 0 and (len(c.window) == delta_t): # Monday and window full
        # Compute average over weeks in range [t-delta_t-1, t[
        mean_indicator = np.mean(c.window)
        if mean_indicator is None:
            return

        # Exit previous positions
        order_target(c.security, 0)

        # Long or short depending on whether debt search 
        # frequency went down or up, respectively.
        if data[c.query][c.query] > mean_indicator:
           order(c.security, -c.order_size*(c.query in data[c.query]))
        else:                
           order(c.security, c.order_size)

    if c.query in data[c.query]:
        c.window.append(data[c.query][c.query])
        record(terrorism_avg=np.mean(c.window))