import pandas as pd
import numpy as np
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

# time frame on which we want to compute Fama-French
normal_days = 31
# approximate the number of trading days in that period
# this is the number of trading days we'll look back on,
# on every trading day.
business_days = int(0.69 * normal_days)

class Returns(CustomFactor):
    """
    this factor outputs the returns over the period defined by 
    business_days, ending on the previous trading day, for every security.
    """
    window_length = business_days
    inputs = [USEquityPricing.close]
    def compute(self,today,assets,out,price):
        out[:] = (price[-1] - price[0]) / price[0] * 100

class MarketEquity(CustomFactor):
    """
    this factor outputs the market cap of every security on the day.
    """
    window_length = business_days
    inputs = [morningstar.valuation.market_cap]
    def compute(self,today,assets,out,mcap):
        out[:] = mcap[0]

class BookEquity(CustomFactor):
    """
    this factor outputs the book value of every security on the day.
    """
    window_length = business_days
    inputs = [morningstar.balance_sheet.tangible_book_value]
    def compute(self,today,assets,out,book):
        out[:] = book[0]
                                        
class CommonStock(CustomFactor):
    """
    this factor outputs 1.0 for all securities that are either common stock or SPY,
    and outputs 0.0 for all other securities. This is to filter out ETFs and other
    types of share that we do not wish to consider.
    """
    window_length = business_days
    inputs = [morningstar.share_class_reference.is_primary_share]
    def compute(self,today,assets,out, share_class):
        out[:] = ((share_class[-1].astype(bool)) | (assets == 8554)).astype(float)                                     
        
def initialize(context):
    """
    use our factors to add our pipes and screens.
    """
    pipe = Pipeline()
    attach_pipeline(pipe, 'ff_example')
    
    common_stock = CommonStock()
    # filter down to securities that are either common stock or SPY
    pipe.set_screen(common_stock.eq(1))
    mkt_cap = MarketEquity()
    pipe.add(mkt_cap,'market_cap')
    
    book_equity = BookEquity()
    # book equity over market equity
    be_me = book_equity/mkt_cap
    pipe.add(be_me,'be_me')

    returns = Returns()
    pipe.add(returns,'returns')
    schedule_function(func=print_fama_french, date_rule=date_rules.every_day())

def print_fama_french(context, data):
    # print the Fama-French factors for the period defined by business_days
    # ending on the previous trading day.
    print(context.rm_rf, context.smb, context.hml)
    
def before_trading_start(context,data):
    """
    every trading day, we use our pipes to construct the Fama-French
    portfolios, and then calculate the Fama-French factors appropriately.
    """
    spy = sid(8554)
    
    factors = pipeline_output('ff_example')
    
    # get the data we're going to use
    returns = factors['returns']
    mkt_cap = factors.sort(['market_cap'], ascending=True)
    be_me = factors.sort(['be_me'], ascending=True)
    
    # to compose the six portfolios, split our universe into portions
    half = int(len(mkt_cap)*0.5)
    small_caps = mkt_cap[:half]
    big_caps = mkt_cap[half:]
    
    thirty = int(len(be_me)*0.3)
    seventy = int(len(be_me)*0.7)
    growth = be_me[:thirty]
    neutral = be_me[thirty:seventy]
    value = be_me[seventy:]
    
    # now use the portions to construct the portfolios.
    # note: these portfolios are just lists (indices) of equities
    small_value = small_caps.index.intersection(value.index)
    small_neutral = small_caps.index.intersection(neutral.index)
    small_growth = small_caps.index.intersection(growth.index)
    
    big_value = big_caps.index.intersection(value.index)
    big_neutral = big_caps.index.intersection(neutral.index)
    big_growth = big_caps.index.intersection(growth.index)
    
    # take the mean to get the portfolio return, assuming uniform
    # allocation to its constituent equities.
    sv = returns[small_value].mean()
    sn = returns[small_neutral].mean()
    sg = returns[small_growth].mean()
    
    bv = returns[big_value].mean()
    bn = returns[big_neutral].mean()
    bg = returns[big_growth].mean()
    
    # computing Rm-Rf (Market Returns - Risk-Free Returns). we take the 
    # rate of risk-free returns to be zero, so this is simply SPY's returns.
    # have to set an initial dummy value
    context.rm_rf = float('nan')
    if spy in returns.index:
        context.rm_rf = returns.loc[spy]
    
    # computing SMB
    context.smb = (sv + sn + sg)/3 - (bv + bn + bg)/3
    
    # computing HML
    context.hml = (sv + bv)/2 - (sg + bg)/2