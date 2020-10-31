"""
Simple equity momentum model by Andreas Clenow

Purpose: 
To capture momentum effect in US stock markets.
Implements two methods of reducing downside.
Index level trend filter, and minimum momentum.

Momentum analytic:
Uses annualized exponential regression slope, multiplied by R2, to adjust for fit.
Possibility to use average of two windows.

Settings:
* Investment universe
* Momentum windows (x2)
* Index trend filter on/off
* Index trend filter window
* Minimum required slope
* Exclude recent x days
* Trading frequency
* Cash management via bond etf, on off
* Bond ETF selection
* Sizing method, inverse vola or equal.


Suggested areas of research and improvements:
* Use pipeline to limit number of processed stocks, by putting the regression logic in there.
* Adding fundamental factors.
* Portfolio sizes
* Market cap weighting / inverse market cap weighting. (you may want to z-score and winsorize)
* Momentum analytic: There are many possible ways. Try simplifications and variations.


"""


from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, CustomFactor, SimpleMovingAverage, Latest
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US
from quantopian.pipeline.data import morningstar

import numpy as np  # we're using this for various math operations
from scipy import stats  # using this for the reg slope
import pandas as pd


def slope(ts):
    """
    Input: Price time series.
    Output: Annualized exponential regression slope, multipl
    """
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100
    return annualized_slope * (r_value ** 2)

def inv_vola_calc(ts):
    """
    Input: Price time series.
    Output: Inverse exponential moving average standard deviation. 
    Purpose: Provides inverse vola for use in vola parity position sizing.
    """
    returns = np.log(ts).diff()
    stddev = returns.ewm(halflife=20, ignore_na=True, min_periods=0,
                         adjust=True).std(bias=False).dropna()
    return 1 / stddev.iloc[-1]


def initialize(context):
    """
    Setting our variables. Modify here to test different
    iterations of the model.
    """
    
    # Investment Universe 1 = Q500US, 2 = Q1500US
    context.investment_set = 1
    
    
    # This version uses the average of two momentum slopes.
    # Want just one? Set them both to the same number.
    context.momentum_window = 60  # first momentum window.
    context.momentum_window2 = 90  # second momentum window
    
    # Limit minimum slope. Keep in mind that shorter momentum windows
    # yield more extreme slope numbers. Adjust one, and you may want
    # to adjust the other.
    context.minimum_momentum = 60  # momentum score cap
    
    # Fixed number of stocks in the portfolio. How diversified
    # do you want to be?
    context.number_of_stocks = 25  # portfolio size
    context.index_id = sid(8554) # identifier for the SPY. used for trend filter.
    context.index_average_window = 100  # moving average periods for index filter
    
    # enable/disable trend filter.
    context.index_trend_filter = True  
    
    # Most momentum research excludes most recent data.
    context.exclude_days = 5  # excludes most recent days from momentum calculation
    
    # Set trading frequency here.
    context.trading_frequency = date_rules.month_start()
    
    # identifier for the cash management etf, if used.
    context.use_bond_etf = True
    context.bond_etf = sid(23870) 
    
    # 1 = inv.vola. 2 = equal size. Suggest to implement 
    # market cap and inverse market cap as well. There's 
    # lots of room for development here.
    context.size_method = 2 
    
    

    # Schedule rebalance
    schedule_function(
        my_rebalance,
        context.trading_frequency,
        time_rules.market_open(
            hours=1))

    # Schedule daily recording of number of positions - For display in back
    # test results only.
    schedule_function(
        my_record_vars,
        date_rules.every_day(),
        time_rules.market_close())

    # Create our dynamic stock selector - getting the top 500 most liquid US
    # stocks.
    
    if(context.investment_set == 1):
        inv_set = Q500US()
    elif(context.investment_set == 2):
        inv_set = Q1500US()
    
    attach_pipeline(make_pipeline(inv_set), 'investment_universe')
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB




def make_pipeline(investment_set): 
    """
    This will return the selected stocks by market cap, dynamically updated.
    """
    # Base universe 
    base_universe = investment_set 
    yesterday_close = USEquityPricing.close.latest

    pipe = Pipeline(
        screen=base_universe,
        columns={
            'close': yesterday_close,
        }
    )
    return pipe


def my_rebalance(context, data):
    """
    Our monthly rebalancing routine
    """

    # First update the stock universe. 
    context.output = pipeline_output('investment_universe')
    context.security_list = context.output.index

    # Get data
    hist_window = max(context.momentum_window,
                      context.momentum_window2) + context.exclude_days

    hist = data.history(context.security_list, "close", hist_window, "1d")

    data_end = -1 * context.exclude_days # exclude most recent data

    momentum1_start = -1 * (context.momentum_window + context.exclude_days)
    momentum_hist1 = hist[momentum1_start:data_end]

    momentum2_start = -1 * (context.momentum_window2 + context.exclude_days)
    momentum_hist2 = hist[momentum2_start:data_end]

    # Calculate momentum scores for all stocks.
    momentum_list = momentum_hist1.apply(slope)  # Mom Window 1
    momentum_list2 = momentum_hist2.apply(slope)  # Mom Window 2

    # Combine the lists and make average
    momentum_concat = pd.concat((momentum_list, momentum_list2))
    mom_by_row = momentum_concat.groupby(momentum_concat.index)
    mom_means = mom_by_row.mean()

    # Sort the momentum list, and we've got ourselves a ranking table.
    ranking_table = mom_means.sort_values(ascending=False)

    # Get the top X stocks, based on the setting above. Slice the dictionary.
    # These are the stocks we want to buy.
    buy_list = ranking_table[:context.number_of_stocks]
    final_buy_list = buy_list[buy_list > context.minimum_momentum] # those who passed minimum slope requirement

    # Calculate inverse volatility, for position size.
    inv_vola_table = hist[buy_list.index].apply(inv_vola_calc)
    # sum inv.vola for all selected stocks.
    sum_inv_vola = np.sum(inv_vola_table)

    # Check trend filter if enabled.
    if (context.index_trend_filter):
        index_history = data.history(
            context.index_id,
            "close",
            context.index_average_window,
            "1d")  # Gets index history
        index_sma = index_history.mean()  # Average of index history
        current_index = index_history[-1]  # get last element
        # declare bull if index is over average
        bull_market = current_index > index_sma

    # if trend filter is used, only buy in bull markets
    # else always buy
    if context.index_trend_filter:
        can_buy = bull_market
    else:
        can_buy = True


    equity_weight = 0.0 # for keeping track of exposure to stocks
    
    # Sell positions no longer wanted.
    for security in context.portfolio.positions:
        if (security not in final_buy_list):
            if (security.sid != context.bond_etf):
                # print 'selling %s' % security
                order_target(security, 0.0)
                
    vola_target_weights = inv_vola_table / sum_inv_vola
    
    for security in final_buy_list.index:
        # allow rebalancing of existing, and new buys if can_buy, i.e. passed trend filter.
        if (security in context.portfolio.positions) or (can_buy): 
            if (context.size_method == 1):
                weight = vola_target_weights[security]
            elif (context.size_method == 2):
                weight = (1.0 / context.number_of_stocks)
                print context.number_of_stocks
            order_target_percent(security, weight)
            equity_weight += weight
    
       

    # Fill remaining portfolio with bond ETF
    etf_weight = max(1 - equity_weight, 0.0)

    print 'equity exposure should be %s ' % equity_weight

    if (context.use_bond_etf):
        order_target_percent(context.bond_etf, etf_weight)

def my_record_vars(context, data):
    """
    This routine just records number of open positions and exposure level
    for display purposes.
    """
    etf_exp = 0.0
    pos_count = 0
    eq_exp = 0.0
    for position in context.portfolio.positions.itervalues():
        pos_count += 1
        if (position.sid == context.bond_etf):
            etf_exp += (position.amount * position.last_sale_price) / \
                context.portfolio.portfolio_value
        else:
            eq_exp += (position.amount * position.last_sale_price) / \
                context.portfolio.portfolio_value

    record(equity_exposure=eq_exp)
    record(bond_exposure=etf_exp)
    record(tot_exposure=context.account.leverage)
    record(positions=pos_count)