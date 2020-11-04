# This algorithm recreates the algorithm presented in
# "Quantifying Trading Behavior in Financial Markets Using Google Trends"
# Preis, Moat & Stanley (2013), Scientific Reports
# (c) 2014 Thomas Wiecki, Quantopian Inc.

import numpy as np
import datetime
from collections import deque as window
# Average over 5 weeks, free parameter.
delta_t = 5

def initialize(context):
    # This is the search query we are using, this is tied to the csv file.
    context.query = 'debt'
    # User fetcher to get data. I uploaded this csv file manually, feel free to use.
    # Note that this data is already weekly averages.
    #'https://gist.github.com/twiecki/5629198/raw/6247da04bacebcd6334a4b91ed21f14483c6d4d0/debt_google_trend'
    fetch_csv('https://gist.githubusercontent.com/anonymous/9778928/raw/92b9a8eafce50a2d184d72d7601c67a0223f77b8/preis_data.csv',
              date_format='%Y-%m-%d',
              symbol='debt',
    )
    context.order_size = 10000
    context.security = symbol('SPY') # S&P5000
    context.window = window(maxlen=delta_t)
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def handle_data(context, data):
    c = context
      
    # Buy and hold strategy that enters on the first day of the 
    # week and exits after one week.
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
            order(c.security, -c.order_size)
        else:
            order(c.security, c.order_size)

    if c.query in data[c.query]:
        c.window.append(data[c.query][c.query])
        record(debt_avg=np.mean(c.window))