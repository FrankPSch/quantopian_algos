# This algorithm recreates the algorithm presented in
# "Quantifying Trading Behavior in Financial Markets Using Google Trends"
# Preis, Moat & Stanley (2013), Scientific Reports
# (c) 2013 Thomas Wiecki, Quantopian Inc.

import numpy as np
import datetime
# Average over 5 weeks, free parameter.
delta_t = 5

def initialize(context):
    # This is the search query we are using, this is tied to the csv file.
    context.query = 'flu'
    # User fetcher to get data. I uploaded this csv file manually, feel free to use.
    # Note that this data is already weekly averages.
    fetch_csv('https://raw.githubusercontent.com/timgood12/flu-data/master/data',
              date_format='%Y-%m-%d',
              symbol='flu',
    )
    context.order_size = 10000
    context.sec_id = 5923
    context.security = sid(5923) #merck

def handle_data(context, data):
    c = context
  
    if c.query not in data[c.query]:
        return
   
    # Extract weekly average of search query.
    indicator = data[c.query][c.query]
    
    # Buy and hold strategy that enters on the first day of the week
    # and exits after one week.
    if data[c.security].dt.weekday() == 0: # Monday
        # Compute average over weeks in range [t-delta_t-1, t[
        mean_indicator = mean_past_queries(data, c.query)
        if mean_indicator is None:
            return

        # Exit positions
        amount = c.portfolio['positions'][c.sec_id].amount
        order(c.security, -amount)

        # Long or short depending on whether debt search frequency
        # went down or up, respectively.
        if indicator > mean_indicator:
            order(c.security, c.order_size)
        else:
            order(c.security, -c.order_size)
        
# If we want the average over 5 weeks, we'll have to use a 6
# week window as the newest element will be the current event.
@batch_transform(window_length=delta_t+1, refresh_period=0)
def mean_past_queries(data, query):
    # Compute mean over all events except most current one.
    return data[query][query][:-1].mean()


