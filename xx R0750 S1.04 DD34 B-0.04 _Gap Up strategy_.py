from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume

from quantopian.pipeline.experimental import QTradableStocksUS
import pandas as pd 
import numpy as np 


def logging(msgs):
    dt = get_datetime('US/Eastern')
    youbi = dt.strftime("%w")
    youbidict = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    msgs = '\t%s\t%s:\t%s'%(dt.strftime('%Y/%m/%d %H:%M (ET)'), youbidict[int(youbi)], msgs)
    log.info(msgs)


class ValueDaybeforeYesterday(CustomFactor):
    window_length = 2
    def compute(self, today, assets, out, values):
        out[:] = values[0]    

        
class ChangeAverage(CustomFactor):
    def compute(self, today, assets, out, values):
        mean = pd.DataFrame(values).pct_change().mean()
        out[:] = mean.values
        
class ChangeAverageLog(CustomFactor):
    def compute(self, today, assets, out, values):
        df = pd.DataFrame(values)
        mean = (df.shift(1) / df).mean().apply(np.log)
        out[:] = mean.values

def make_pipeline(context):
    pipe = Pipeline()
    base_universe = QTradableStocksUS()
    dollar_volume = AverageDollarVolume(window_length=30)
    high_dollar_volume = dollar_volume.percentile_between(98, 100)
    
    close_day_before_yeseterday = ValueDaybeforeYesterday(inputs = [USEquityPricing.close])
    volume_day_before_yeseterday = ValueDaybeforeYesterday(inputs = [USEquityPricing.volume])
    pipe.add(close_day_before_yeseterday, "close_day_before_yeseterday")
    
    return_change_mean = ChangeAverage(inputs = [USEquityPricing.close], window_length = 5)
    volume_change_mean = ChangeAverage(inputs = [USEquityPricing.volume], window_length = 5)
    
    
    my_screen = base_universe  & high_dollar_volume  #& (return_change_mean < 0)
    pipe.set_screen(my_screen)
    return pipe 

    
def initialize(context):
    attach_pipeline(make_pipeline(context), 'pipe')
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open())
    schedule_function(my_close, date_rules.every_day(), time_rules.market_open(minutes=40))
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

        
def before_trading_start(context, data):
    context.output = pipeline_output('pipe')  

def calc_gap(context, data):
    
    sids = context.output.index

    df = pd.DataFrame()
    if len(sids)>0:
        open_price = data.current(sids,'price').rename("open")
        df = pd.concat([context.output, open_price], axis = 1) 
        df['gap'] = df['open'] / df['close_day_before_yeseterday'] - 1
        df = df[(df['gap'] > 0.05)  ]#& (df['gap'] < 1.0)
    return df
    
def my_rebalance(context, data):
    targets = calc_gap(context, data)
    
        
    if not targets.empty:
        cnt = len(targets)
    
        targets['nomalized_gap'] = targets.gap /targets.gap.abs().sum()
        for sid in targets.index:
            # order_percent(sid, -1.0 / cnt)
            logging("{0}\t{1: .5f}".format(sid.symbol, targets['gap'].loc[sid]))
            order_percent(sid, targets['nomalized_gap'].loc[sid] * -1.0)
            #logging("{0}\t{1: .5f}".format(sid.symbol, targets['nomalized_gap'].loc[sid] * -1.0))
        
        

def my_close(context, data):
    if len(context.portfolio.positions) > 0:
        for sid in context.portfolio.positions:
            order_target(sid, 0)