# Import the libraries we will use here.
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.pipeline.factors import AverageDollarVolume, Returns
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as psychsignal

import pandas as pd
import numpy as np

trading_symbols = (sid(8554),   # SPY
                   sid(2174),   # DIA
                   sid(19920),  # QQQ
                   sid(21519),  # IWM
                   sid(21513),  # IVV
                   sid(21520),  # IWV
                   sid(34385),  # VEA
                   sid(28073),  # XBI
                   sid(19658),  # XLK
                   sid(19661),  # XLV
                   sid(19659),  # XLP
                   sid(19660),  # XLU
                   sid(19655),  # XLE
                   sid(19657),  # XLI
                   sid(22972),  # EFA
                   sid(14522),  # EWL
                   sid(14520),  # EWJ
                   sid(14529),  # EWU
                   sid(27102),  # VWO
                   sid(33655),  # HYG
                   sid(19654),  # XLB
                   sid(19662),  # XLY
                   sid(32275), # XRT
                   sid(26432),  # FEZ
                   sid(14516),  # EWA
                   sid(14519),  # EWH
                   sid(32270),  # SSO
                   sid(39214),  # TQQQ
                   sid(22739),  # VTI
                   sid(38533),  # UPRO
                   sid(21512),  # IVE
                   sid(26669),  # VNQ
                   sid(21518),  # IWF
                   sid(21507),  # IJH
                   sid(21517),  # IWD
                   sid(25909),  # VTV
                   sid(28364),  # VIG
                   sid(25910),  # VUG
                   sid(21508),  # IJR
                   sid(32620),  # PFF
                   sid(12915),  # MDY
                   sid(25647),  # DVY
                   sid(21516),  # IWB
                   sid(32888),  # VYM
                   sid(25907),  # VO
                   sid(40107),  # VOO
                   sid(21514),  # IVW
                   sid(25899),  # VB
                   sid(22908),  # IWR
                   sid(21786))  # IWO

anchor_symbol  = sid(8554)

class SidInList(CustomFilter):  
    inputs = []
    window_length = 1
    params = ('sid_list',)

    def compute(self, today, assets, out, sid_list):
        out[:] = np.in1d(assets, sid_list)       
 
def initialize(context):
    my_sid_filter = SidInList(
        sid_list = (anchor_symbol)
    )
    pipe = Pipeline(screen = my_sid_filter)
    attach_pipeline(pipe, 'my_pipeline')

    url = 'http://hive.psychsignal.com/public/historical/hivebot/SPY'  
    fetch_csv(url, 
              date_column='date',
              symbol='spy_sas',
              usecols=['SAS'],
              date_format='%Y-%m-%d')
    
    schedule_function(rebalance,
                      date_rules.every_day(),#(days_offset=1),
                      time_rules.market_open(hours=0, minutes=30))
    
def handle_data(context, data):
    context.SAS = data.current('spy_sas','SAS')
    
def my_pipeline(context):
    my_sid_filter = SidInList(
        sid_list = (anchor_symbol)
    )

    pipe = Pipeline(
            columns = {},
            screen = my_sid_filter,
            )
    
    return pipe

    
def before_trading_start(context, data):
    record(leverage_ratio=context.account.leverage)
    context.output = pipeline_output('my_pipeline')

def rebalance(context,data):
    sas_thresh = 0.66
    sas  = context.SAS
    bull_weight = 1.0 / len(trading_symbols)
    bear_weight = -0.8 / len(trading_symbols)
    momo_thresh = 0.
    revs_thresh = 0.
    price_history = data.history(anchor_symbol, "price", 40, "1d")
    momo1 = (price_history.ix[-1] - price_history.ix[0]) / price_history.ix[0]
    momo2 = (price_history.ix[-1] - price_history.ix[-2]) / price_history.ix[-2]
    for trading_symbol in trading_symbols:
        #price_history = data.history(trading_symbol, "price", 40, "1d")
        #momo1 = (price_history.ix[-1] - price_history.ix[0]) / price_history.ix[0]
        #momo2 = (price_history.ix[-1] - price_history.ix[-2]) / price_history.ix[-2]
        pos_weight1 = 0
        pos_weight2 = 0
        if sas > sas_thresh:
            # momo mode
            # check if momo1 or momo2 is triggered
            if momo1 > momo_thresh:
                pos_weight1 = bull_weight
            elif momo1 < -momo_thresh:
                pos_weight1 = bear_weight
            if momo2 > momo_thresh:
                pos_weight2 = bull_weight
            elif momo2 < -momo_thresh:
                pos_weight2 = bear_weight
        else:
            # mean reversion mode
            # chick if momo1 is triggered
            if momo1 > revs_thresh:
                pos_weight1 = 2. * bear_weight
            elif momo1 < -revs_thresh:
                pos_weight1 = 2. * bull_weight
                
        pos_weight = pos_weight1 + pos_weight2
        order_target_percent(trading_symbol, pos_weight)
            
    del price_history
    del momo1
    del momo2
    del sas