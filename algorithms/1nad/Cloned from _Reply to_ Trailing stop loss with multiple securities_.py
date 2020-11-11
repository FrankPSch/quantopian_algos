"""
    Trading Strategy using Fundamental Data
    
    1. Filter the top 50 companies by market cap 
    2. Find the top two sectors that have the highest average PE ratio
    3. Every month exit all the positions before entering new ones at the month
    4. Log the positions that we need 
    5. Maintain a list of trailing stops for all securities.  If any one falls below its trailing stop, liquidate the whole basket.
      Note that Quantopian does not allow to sell positions which are no longer in the universe, or to sell them before updating the universe in before_trading_start.
    
By Harrison Schwartz, amended by Andre Pajak.
"""

import pandas as pd
import numpy as np

def initialize(context):
    # Dictionary of stocks and their respective weights
    context.stock_weights = {}
    # Count of days before rebalancing
    context.days = 0
    # Number of sectors to go long in
    context.sect_numb = 2
    
    context.stop_pct = 0.98
    
    context.stop_price = {}
    
    # Sector mappings
    context.sector_mappings = {
       101.0: "Basic Materials",
       102.0: "Consumer Cyclical",
       103.0: "Financial Services",
       104.0: "Real Estate",
       205.0: "Consumer Defensive",
       206.0: "Healthcare",
       207.0: "Utilites",
       308.0: "Communication Services",
       309.0: "Energy",
       310.0: "Industrials",
       311.0: "Technology"
    }
    
    # Rebalance monthly on the first day of the month at market open
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())

def rebalance(context, data):
    # Exit all positions before starting new ones
    for stock in context.portfolio.positions:
        if stock not in context.fundamental_df and stock in data:
            order_target_percent(stock, 0)

    log.info("The two sectors we are ordering today are %r" % context.sectors)

    # Create weights for each stock
    weight = create_weights(context, context.stocks)

    # Rebalance all stocks to target weights
    for stock in context.fundamental_df:
        if stock in data:
          if weight != 0:
              log.info("Ordering %0.0f%% percent of %s in %s" 
                       % (weight * 100, 
                          stock.symbol, 
                          context.sector_mappings[context.fundamental_df[stock]['morningstar_sector_code']]))
              
          order_target_percent(stock, weight)
    
def before_trading_start(context): 
    """
      Called before the start of each trading day. 
      It updates our universe with the
      securities and values found from get_fundamentals.
    """
    
    num_stocks = 50
    
    # Setup SQLAlchemy query to screen stocks based on PE ratio
    # and industry sector. Then filter results based on 
    # market cap and shares outstanding.
    # We limit the number of results to num_stocks and return the data
    # in descending order.
    fundamental_df = get_fundamentals(
        query(
            # put your query in here by typing "fundamentals."
            fundamentals.valuation_ratios.pe_ratio,
            fundamentals.operation_ratios.roe,
            fundamentals.earnings_report.basic_eps,
            fundamentals.operation_ratios.debtto_assets,
            fundamentals.valuation_ratios.book_value_yield,
            fundamentals.asset_classification.morningstar_sector_code,
            fundamentals.asset_classification.value_score,
            fundamentals.valuation.market_cap
            )
        .filter(fundamentals.valuation.market_cap >= 200000000)
        .filter(fundamentals.valuation.shares_outstanding != None)
        .filter(fundamentals.valuation_ratios.pe_ratio <= 15)
        .filter(fundamentals.operation_ratios.roe >=.05)
        .filter(fundamentals.earnings_report.basic_eps >= .05)
        .filter(fundamentals.operation_ratios.debtto_assets <= .5)
        .filter(fundamentals.valuation_ratios.book_value_yield <= 1.5)
        .order_by(fundamentals.valuation.market_cap.desc())
        .limit(num_stocks)
    )
    # Find sectors with the highest average PE
    sector_pe_dict = {}
    for stock in fundamental_df:
        sector = fundamental_df[stock]['morningstar_sector_code']
        pe = fundamental_df[stock]['pe_ratio']
        
        # If it exists add our pe to the existing list. 
        # Otherwise don't add it.
        if sector in sector_pe_dict:
            sector_pe_dict[sector].append(pe)
        else:
            sector_pe_dict[sector] = []
    
    # Find average PE per sector
    sector_pe_dict = dict([(sectors, np.average(sector_pe_dict[sectors])) 
                               for sectors in sector_pe_dict if len(sector_pe_dict[sectors]) > 0])
    
    # Sort in ascending order
    sectors = sorted(sector_pe_dict, key=lambda x: sector_pe_dict[x], reverse=True)[:context.sect_numb]
    
    # Filter out only stocks with that particular sector
    context.stocks = [stock for stock in fundamental_df
                      if fundamental_df[stock]['morningstar_sector_code'] in sectors]
    
    # Initialize a context.sectors variable
    context.sectors = [context.sector_mappings[sect] for sect in sectors]

    # Update context.fundamental_df with the securities (and pe_ratio) that we need
    context.fundamental_df = fundamental_df[context.stocks]
    context.liquidate = False
    
    update_universe(context.fundamental_df.columns.values)   
    
    
def create_weights(context, stocks):
    """
        Takes in a list of securities and weights them all equally 
    """
    if len(stocks) == 0:
        return 0 
    else:
        weight = 1.0/len(stocks)
        return weight
        
def set_trailing_stop(context, data):
    for stock in context.fundamental_df:
        if stock in data:
                price = data[stock].price
                context.stop_price[stock] = max(context.stop_price[stock] if stock in context.stop_price else 0, context.stop_pct * price) 
            
def handle_data(context, data):
    """
      Code logic to run during the trading day.
      handle_data() gets called every bar.
    """
    record(leverage = context.account.leverage)
    # track how many positions we're holding
    record(num_positions = len(context.portfolio.positions))
    if context.liquidate:
        return # No more trading today after liquidating
    for stock in context.fundamental_df:
        current_position = context.portfolio.positions[stock].amount
        set_trailing_stop(context, data)
        if (stock in data) and (data[stock].price < context.stop_price[stock]) and (current_position > 0):
            context.liquidate = True
            log.info("Selling all because of {}*{}: {}<{}".format(current_position, stock.symbol, data[stock].price, context.stop_price[stock]))
    if context.liquidate:
        for stock in set(context.stocks) & set( context.portfolio.positions.keys()):        
          if context.portfolio.positions[stock]>0:
            #order_target_percent(stock, 0)
            order_target(stock, 0)
            #del context.stop_price[stock] 
            log.info("Trail Selling {} at {}".format(stock.symbol, data[stock].price if stock in data else "??"))