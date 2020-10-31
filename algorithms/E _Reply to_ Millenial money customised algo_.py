"""
    Patrick O'Shaughnessy - Millennial Money
    
    1. Stakeholder yield < 5%. Stakeholder yield = Cash from financing 12m / Market Cap Q1
    2. ROIC > 25%
       ROIC = operating_income /  (invested_capital - cash)
    3. CFO > Net Income (Earnings Quality)
    4. EV/FCF < 15 (Value)
    5. 6M Relative Strength top three-quarters of the market. 
       6M Relative Strength = 6M Stock Total Return / 6M Total Return S&P500 (Momentum)
    6. Positions are kept at least for 2 quarters
    7. implemented 15% stop loss; if the stock hits stop loss it cannot be repurchased for a             half year
    8. market cap > 30m
       
    The positions are updated quarterly.
"""

import pandas as pd
import numpy as np

def initialize(context):
    context.account.leverage = 0 
    context.bought_for={}
    context.all_current_stocks = []
    context.prices = {}
    context.max_num_stocks = 50
    context.days = 64
    context.quarter_days = 65
    context.quarters_alive = {}
    context.relative_strength_6m = {}
    context.banned_days = {}
    schedule_function(func=compute_strength_and_rebalance, date_rule=date_rules.every_day())
          
def quarter_passed(context): 
    """
    Screener results quarterly updated
    """
    return context.days % context.quarter_days == 0

def before_trading_start(context, data): 
    context.days += 1
    
    if not quarter_passed(context):
        return
    
    do_screening(context)
    context.security_list = list(context.fundamental_df.columns.values)
    
def do_screening(context):
    fundamental_df = get_fundamentals(
        query(
            fundamentals.asset_classification.morningstar_sector_code,
            fundamentals.company_reference.country_id,
            fundamentals.company_reference.primary_exchange_id,
            fundamentals.share_class_reference.is_depositary_receipt,
            fundamentals.share_class_reference.is_primary_share,
            fundamentals.cash_flow_statement.financing_cash_flow,
            fundamentals.valuation.market_cap,
            fundamentals.income_statement.operating_income,
            fundamentals.balance_sheet.invested_capital,
            fundamentals.balance_sheet.cash_and_cash_equivalents,
            fundamentals.cash_flow_statement.operating_cash_flow,
            fundamentals.income_statement.net_income,
            fundamentals.valuation.enterprise_value,
            fundamentals.cash_flow_statement.free_cash_flow
        )

        # No Financials (103) and Real Estate (104) Stocks, no ADR or PINK, only USA
        .filter(fundamentals.asset_classification.morningstar_sector_code != 103)
        .filter(fundamentals.company_reference.country_id == "USA")
        .filter(fundamentals.asset_classification.morningstar_sector_code != 104)
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False)
        .filter(fundamentals.share_class_reference.is_primary_share == True)
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK")
        
        # Check for data sanity (i,e. avoid division by zero)
        .filter(fundamentals.valuation.market_cap > 30000000)
        .filter(fundamentals.valuation.shares_outstanding > 0)
        .filter(fundamentals.cash_flow_statement.free_cash_flow > 0)
        .filter(fundamentals.balance_sheet.invested_capital > 0)
        .filter(fundamentals.balance_sheet.cash_and_cash_equivalents > 0)
        .filter(fundamentals.balance_sheet.invested_capital != fundamentals.balance_sheet.cash_and_cash_equivalents)
        
        .filter((fundamentals.cash_flow_statement.financing_cash_flow / fundamentals.valuation.market_cap) < 0.05)
        .filter((fundamentals.income_statement.operating_income / (fundamentals.balance_sheet.invested_capital - fundamentals.balance_sheet.cash_and_cash_equivalents)) > 0.25)
       
        .filter(fundamentals.cash_flow_statement.operating_cash_flow > fundamentals.income_statement.net_income)
        .filter((fundamentals.valuation.enterprise_value / fundamentals.cash_flow_statement.free_cash_flow) < 12)
        
        .limit(context.max_num_stocks)
    )
   
     # Update context
    context.stocks = [stock for stock in fundamental_df]
    context.all_current_stocks = [stock for stock in fundamental_df]
    
    context.fundamental_df = fundamental_df
    
    
def rebalance(context, data):
    stock_already_bought_not_matching_latest_search = []
    # Exit positions before starting new ones
    for stock in context.portfolio.positions:
        exceded_required_quarters = (stock not in context.quarters_alive or context.quarters_alive[stock] >= 2)
        if stock not in context.fundamental_df and exceded_required_quarters:
            if data.can_trade(stock):
                order_target_percent(stock, 0)
            if stock in context.quarters_alive:
                del(context.quarters_alive[stock])
        elif stock not in context.fundamental_df:
            stock_already_bought_not_matching_latest_search.append(stock)
             

    # Filter out stocks without data and apply the momentum criteria
    # -0.6745 is an approximation for the top three-quarters of the market
    context.stocks = [stock for stock in context.stocks
                      if data.can_trade(stock) and context.relative_strength_6m[stock] > -0.6745]
    
    # make sure to get out of delisted stocks so they don't sit stagnant in portfolio
    # see this thread for discussion of this approach: 
    # https://www.quantopian.com/posts/why-does-the-number-of-positions-and-leverage-creep-up-for-this-algorithm
    context.stocks = [stock for stock in context.stocks
                      if (stock.end_date - get_datetime()).days > 100]
    
    new_stock =  [stock for stock in context.stocks
                      if stock not in context.portfolio.positions]
    
    # save price for SL
    save_buy_price(context, data, new_stock)
        
    context.stocks = context.stocks + stock_already_bought_not_matching_latest_search
    
    # remove banned stock
    context.stocks = remove_banned_stock(context, context.stocks)
    
    if len(context.stocks) == 0:
        log.info("No Stocks to buy")
        return
   
    weight = 1.0/len(context.stocks)

    log.info("Ordering %0.0f%% for each of %s (%d stocks)" % (weight * 100, ', '.join(stock.symbol for stock in context.stocks), len(context.stocks)))
    
    # buy all stocks equally
    for stock in context.stocks:
        if data.can_trade(stock):
            if stock not in context.quarters_alive:
                context.quarters_alive[stock] = 0
            order_target_percent(stock,  weight)

def save_buy_price(context, data, new_stock):
    context.prices = data.history(new_stock, 'price', 1, '1d')
    for stock in new_stock:
        if stock in context.banned_days and context.banned_days[stock] > 0:
            continue
        context.bought_for[stock] = context.prices[stock][0]

def remove_banned_stock(context, stocks):
    for stock in stocks:
        if stock in context.banned_days:
            if context.banned_days[stock] > 0:
                stocks.remove(stock) 
    return stocks
    
    
def compute_relative_strength(context, data):   
    prices = data.history(context.security_list + [symbol('SPY')], 'price', 150, '1d')
    # Price % change in the last 6 months
    pct_change = (prices.ix[-130] - prices.ix[0]) / prices.ix[0]
    
    pct_change_spy = pct_change[symbol('SPY')]
    pct_change = pct_change - pct_change_spy
    if pct_change_spy != 0:
        pct_change = pct_change / abs(pct_change_spy)
    pct_change = pct_change.drop(symbol('SPY'))
    context.relative_strength_6m = pct_change
    
def update_stocks_qarters(context):
    for stock in context.quarters_alive:
        context.quarters_alive[stock] +=1

def update_banned_days(context):
    for stock in context.banned_days:
        context.banned_days[stock] -=1

def day_trading(context, data):
    context.prices = data.history(context.all_current_stocks, 'price', 1, '1d')
    for stock in context.portfolio.positions:
        if stock not in context.bought_for:
            continue
        if stock not in context.prices:
            continue
            
        if context.bought_for[stock] == 0.0:
            continue
            
        if len(context.prices[stock]) == 0:
            continue
            
        try:
            if hit_stop_loss(context, stock):
                log.info("%s hit stop loss" % stock.symbol)
                context.banned_days[stock] = 130 #banned for half year
                order_target_percent(stock, 0)
        except:
            pass
    
def hit_stop_loss(context, stock):
     pct_change = (context.prices[stock][0] - context.bought_for[stock])/context.bought_for[stock]
     return pct_change <= -0.15
    
def compute_strength_and_rebalance(context, data):
    record(num_positions = len(context.portfolio.positions))
    day_trading(context, data)
    update_banned_days(context)
    
    if not quarter_passed(context):
        return
    
    update_stocks_qarters(context)
    
    compute_relative_strength(context, data)
    rebalance(context, data)