from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.filters import Q500US
import pandas as pd
import numpy as np

class MarketCap(CustomFactor):
    #рыночная капитализация - количество акций на их цену
    inputs = [morningstar.valuation.shares_outstanding, USEquityPricing.close]
    window_length = 1
    
    def compute(self, today, assets, out, shares, close_price):
        out[:] = shares * close_price

class BookToPrice(CustomFactor):
    #берем b/p как 1/(p/b)
    inputs = [morningstar.valuation_ratios.pb_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, pb):
        out[:] = 1 / pb
class Quality(CustomFactor):
    #return on equity
    inputs = [morningstar.operation_ratios.roe]
    window_length = 1
    
    def compute(self, today, assets, out, roe):       
        out[:] = roe[-1]
def make_pipeline(context):

    pipe = Pipeline()
    # Q1500US - вселенная из 1500 самых ликвидных активов
    universe = Q1500US()
    market_cap = MarketCap(mask=universe)
    market_cap_rank = market_cap.rank()
    # берем половину активов с большой капитализацией
    market_cap_high = market_cap_rank.top(750)
    quality=Quality(mask=market_cap_high)
    quality_rank=quality.rank()
    # 100 самых целесообразных в плане бизнеса
    qualit=quality_rank.top(100)
    book_to_price = BookToPrice(mask=qualit) 
    book_to_price_rank = book_to_price.rank()
    # 50 недооцененных в низким b/p
    highbp = book_to_price_rank.top(15)
    securities_to_trade = ( highbp)
    pipe = Pipeline(
        columns={
            'highbp':highbp,
        },
        screen = securities_to_trade
    )
  
    
    return pipe

def initialize(context):  

  # будем ребалансировать наш портфель каждый январь в начале, через 1.5 часа после открытия

    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance,
                      date_rules.month_start(days_offset=0),
                      time_rules.market_open(hours=1, minutes=30))
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def before_trading_start(context, data):
# включаем все отобранные активы в наш портфель
    context.output = pipeline_output('my_pipeline')

    context.highbp = context.output[context.output['highbp']]

    context.security_list = context.highbp.index.tolist()


def rebalance(context,data):
    # у нас будет равно-взвешенный портфель
    today = get_datetime('US/Eastern')  
    if  today.month==1:
        for stock in context.portfolio.positions: 
            if stock not in context.security_list:
                if data.can_trade(stock):  
                    order_target_percent(stock, 0)
        print context.security_list
        for security in context.security_list:
            if data.can_trade(security):
                order_target_percent(security, 0.066)
                record(leverage = context.account.leverage)