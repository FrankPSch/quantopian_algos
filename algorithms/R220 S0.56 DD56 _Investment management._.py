from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume
from quantopian.pipeline.data import morningstar

import pandas as pd
import numpy as np



class Value(CustomFactor):
    
    inputs = [morningstar.balance_sheet.tangible_book_value,
              morningstar.valuation_ratios.fcf_yield,
              morningstar.valuation.market_cap,
             morningstar.operation_ratios.long_term_debt_equity_ratio]
    
    window_length = 1
    
    def compute(self, today, assets, out, book_value, fcf, cap, debt):
        value_table = pd.DataFrame(index=assets)
        value_table["fcf"] = fcf[-1]
        value_table["bm"] = cap[-1] / book_value[-1]
        value_table["debt"] = debt[-1]
        out[:] = value_table.rank().mean(axis=1)
    
    
class Momentum(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):
        out[:] = close[-16] / close[0]

            
class Quality(CustomFactor):
    
    inputs = [morningstar.income_statement.gross_profit, morningstar.balance_sheet.total_assets]
    window_length = 1
    
    def compute(self, today, assets, out, gross_profit, total_assets):       
        out[:] = gross_profit[-1] / total_assets[-1]
        
        
def before_trading_start(context, data):
    results = pipeline_output('factors').dropna()
    ranks = results.rank().mean(axis=1).order()
   
    
    context.hold = ranks[results["momentum"] > 1].tail(40)
    context.hold /= context.hold.sum()
    
    context.security_list = context.hold.tolist()
    
    

def initialize(context):
    pipe = Pipeline()
    pipe = attach_pipeline(pipe, name='factors')
        
    value = Value()
    momentum = Momentum()
    quality = Quality()
    
    pipe.add(value, "value")
    pipe.add(momentum, "momentum")
    pipe.add(quality, "quality")

    dollar_volume = AverageDollarVolume(window_length=20)
    
    pipe.set_screen(dollar_volume > 10**6)
    
    context.spy = sid(8554)
    context.hold = None
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    schedule_function(rebalance, date_rules.month_start())
    schedule_function(cancel_open_orders, date_rules.every_day(),
                      time_rules.market_close())
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

    
def record_vars(context, data):
    record(lever=context.account.leverage,
           exposure=context.account.net_leverage,
           num_pos=len(context.portfolio.positions),
           oo=len(get_open_orders()))

    
def cancel_open_orders(context, data):
    open_orders = get_open_orders()
    for security in open_orders:
        for order in open_orders[security]:
            cancel_order(order)
        
    
def rebalance(context, data):
        
    for security in context.portfolio.positions:
        if get_open_orders(security):
            continue
        if data.can_trade(security) and security not in context.hold:
            order_target_percent(security, 0)
                
        
        
    for security in context.hold.index:
        if get_open_orders(security):
            continue
        if data.can_trade(security):
            order_target_percent(security, context.hold[security])
            

def handle_data(context, data):
    pass