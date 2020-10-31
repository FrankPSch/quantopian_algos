from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.filters import Q500US
import pandas as pd
import numpy as np
class Volatility(CustomFactor):
#оцениваем волатильность на основании данных прошлого года    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):  
        close = pd.DataFrame(data=close, columns=assets) 
        # берем логарифм цены, разность между этими логарфмами и считаем ст.отклон.
        out[:] = np.log(close).diff().std()
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
    volatility=Volatility(mask=market_cap_high)
    volatility_rank=volatility.rank()
    # оставляем 1/3 наименее волатильных
    volatility_low=volatility_rank.bottom(250)
    quality=Quality(mask=volatility_low)
    quality_rank=quality.rank()
    # 100 самых целесообразных в плане бизнеса
    qualit=quality_rank.top(120)  # was 100
    book_to_price = BookToPrice(mask=qualit) 
    book_to_price_rank = book_to_price.rank()
    # 50 недооцененных в низким b/p
    highbp = book_to_price_rank.top(110)  # was 50
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
   
def before_trading_start(context, data):
# включаем все отобранные активы в наш портфель
    context.output = pipeline_output('my_pipeline')

    context.highbp = context.output[context.output['highbp']]

    context.security_list = context.highbp.index.tolist()


def rebalance(context,data):
    # у нас будет равно-взвешенный портфель
    today = get_datetime('US/Eastern')  
    if  (today.month==1) or (today.month==4) or (today.month==7) or (today.month==10): 
        # print context.security_list
        for security in context.security_list:
            if data.can_trade(security):
                order_target_percent(security, 0.02)

        # @KSENIA: your algo will not sell assets that are not in the target list anymore.
        # As US1500 is an unstable universe you need something like this in the rebalance function
        for security in context.portfolio.positions:  
            if security not in context.security_list:  
                order_target_percent(security, 0.0)  
                
    record(leverage = context.account.leverage)

"""    
Mods (Guy Fleury) 18/12/17:
Increased stake to $10 million

Total Returns 602.47%
Benchmark Returns 176.22%
Alpha 0.09
Beta 1.30
Sharpe 1.28
Sortino 1.86
Volatility 0.21
Max Drawdown -19.83%
Total Profit: $60.2 million.

Mods:
Increased tradable candidates
Set rebalance twice a year.

Total Returns 1573.34%
Benchmark Returns 176.22%
Alpha 0.12
Beta 2.11
Sharpe 1.25
Sortino 1.81
Volatility 0.33
Max Drawdown -36.93%
Total Profit: $157 million

Mods:
Set rebalance to three times a year.

Total Returns 2191.37%
Benchmark Returns 176.22%
Alpha 0.13
Beta 2.39
Sharpe 1.26
Sortino 1.83
Volatility 0.37
Max Drawdown -40.33%
Total Profit: $219.1 million
added: $61.8 million just for rebalancing 3x a year.

Mods:
Set rebalance to four times a year.
Increased tradable candidates

Total Returns 3809.66%
Benchmark Returns 176.22%
Alpha 0.16
Beta 2.95
Sharpe 1.24
Sortino 1.82
Volatility 0.46
Max Drawdown -50.96%
Total Profit: $380.9 million
added: $161.8 million to previous scenario

Mods:
Set initial stake to $50 million
Increased tradable candidates

Total Returns 2861.72%
Benchmark Returns 176.22%
Alpha 0.13
Beta 2.84
Sharpe 1.20
Sortino 1.75
Volatility 0.44
Max Drawdown -51.26%
Total Profit: $1,430,862,069 

"""