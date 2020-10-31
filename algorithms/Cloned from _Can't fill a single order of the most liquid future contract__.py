from zipline.api import continuous_future, schedule_function
from zipline.api import time_rules, date_rules, record
#from zipline.algorithm import log
#from zipline.assets.continuous_futures import ContinuousFuture
from zipline.api import order_target_value, order_target_percent
from zipline.utils.calendars import get_calendar
import numpy as np
import scipy as sp
import pandas as pd
 
    
def initialize(context):
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))
         
def my_rebalance(context, data):
    f = continuous_future("SY", offset=0, roll='volume', adjustment='mul')
    contract =  data.current_chain(f)[0]
    log.info(contract)
    order(contract, 1)