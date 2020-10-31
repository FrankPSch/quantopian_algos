"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline import CustomFactor  
from quantopian.pipeline.data import morningstar
import pandas as pd
import numpy as np


def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    # Rebalance every day, 1 hour after market open.
    #schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
     
    # Record tracking variables at the end of each day.
    #schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
     
    # Create our dynamic stock selector.
    context.capacity = 25.0
    context.weight = 1.0/context.capacity
    context.buy = True
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    set_long_only()
    
    #schedule for buying a week after the year start
    schedule_function(func=schedule_task_a,
                      date_rule=date_rules.month_start(4),
                      time_rule=time_rules.market_open())
    #schedule for selling losers a week before the year start
    schedule_function(func=schedule_task_b,
                      date_rule=date_rules.month_end(4),
                      time_rule=time_rules.market_open())
    #schedule for selling winners on the 7th day of year start
    schedule_function(func=schedule_task_c,
                      date_rule=date_rules.month_start(3),
                      time_rule=time_rules.market_close())
                      
    
    
def schedule_task_a(context, data):
    today = get_datetime('US/Eastern')
    if today.month == 1:
        for stock in context.portfolio.positions:
            #print stock
            print stock
        for stock in context.stocks.index:
            order_target_percent(stock, context.weight)
            
#selling losers
def schedule_task_b(context, data):
    today = get_datetime('US/Eastern')
    if today.month == 12 and context.portfolio.positions_value != 0:
        for stock in context.portfolio.positions:
            if context.portfolio.positions[stock].cost_basis > data[stock].price:
                order_target_percent(stock, 0)            
        print today, 'losers sold'

#selling winners
def schedule_task_c(context, data):
    today = get_datetime('US/Eastern')
    if today.month == 1:
        for stock in context.portfolio.positions:
            order_target_percent(stock, 0)
    
    
def before_trading_start(context, data):
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """
    
    fundamental_df = get_fundamentals(
        
        query(
            #min market cap at 50 mil, finance and foreign stocks excluded, 
            #earnings yield, return on capital
            fundamentals.asset_classification.morningstar_sector_code,
            fundamentals.income_statement.ebit,
            fundamentals.valuation.enterprise_value, 
            fundamentals.operation_ratios.roic, 
            fundamentals.income_statement.ebitda
            )
    
#    .filter(fundamentals.valuation.market_cap > 50000000)    
#    .filter(fundamentals.valuation.market_cap < 500000000)   

    .filter(fundamentals.valuation.market_cap > 1000000000)    
    .filter(fundamentals.valuation.market_cap < 10000000000)   
    #.filter(fundamentals.valuation_ratios.ev_to_ebitda > 0)
    .filter(fundamentals.asset_classification.morningstar_sector_code != 103)
    .filter(fundamentals.asset_classification.morningstar_sector_code != 207)
    .filter(fundamentals.asset_classification.morningstar_sector_code != 206)
    .filter(fundamentals.asset_classification.morningstar_sector_code != 309)
    .filter(fundamentals.asset_classification.morningstar_industry_code != 20533080)
    .filter(fundamentals.asset_classification.morningstar_industry_code != 10217033)    
    .filter(fundamentals.asset_classification.morningstar_industry_group_code != 10106)
    .filter(fundamentals.asset_classification.morningstar_industry_group_code != 10104)
    .filter(fundamentals.valuation.shares_outstanding != None)
    .filter(fundamentals.valuation.market_cap != None)
    .filter(fundamentals.valuation.shares_outstanding != None)  
    .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
    .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
    .filter(fundamentals.company_reference.country_id == "USA")
    .filter(fundamentals.asset_classification.morningstar_sector_code != None) # require sector
    .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
    .filter(((fundamentals.valuation.market_cap*1.0) / (fundamentals.valuation.shares_outstanding*1.0)) > 10.0)  # stock price > $1
    .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
    .filter(~fundamentals.company_reference.standard_name.contains(' LP')) # exclude LPs
    .filter(~fundamentals.company_reference.standard_name.contains(' L P'))
    .filter(~fundamentals.company_reference.standard_name.contains(' L.P'))
    .filter(fundamentals.balance_sheet.limited_partnership == None) # exclude LPs

        
    #.order_by(fundamentals.valuation_ratios.ev_to_ebitda.asc())
    )
    fundamental_df.loc['earnings_yield'] = fundamental_df.loc['ebit']/fundamental_df.loc['enterprise_value']
    #print fundamental_df.loc['ebit'], fundamental_df.loc['enterprise_value'], fundamental_df.loc['earnings_yield']                                                                                         
    #rank the companies based on their earnings yield
    earnings_yield = fundamental_df
    ey = earnings_yield.loc['earnings_yield']
    rank_ey = ey.rank(ascending = 0)
    #rank the companies based on the return on capital
    rank_roic = fundamental_df.loc['roic'].rank(ascending = 0)
    total_rank = rank_ey + rank_roic
    sorted_rank = total_rank.sort_values()
    
    
    print ey, rank_ey
    print rank_roic, total_rank
    print sorted_rank
    #get 100 best stocks 
    context.stocks = sorted_rank[0:int(context.capacity)]