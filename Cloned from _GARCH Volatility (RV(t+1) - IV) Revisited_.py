from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, CustomFactor
from zipline.utils.tradingcalendar import get_early_closes,trading_day
import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels as sm
#from quantopian.pipeline.data.quandl import cboe_vix,cboe_vxn
import math

def GARCH11_logL(param, r, context):
    omega, alpha, beta = param
    n = len(r)
    s = np.ones(n)*0.01
    s[2] = np.var(r[0:3])
    for i in range(3, n):
        s[i] = omega + alpha*r[i-1]**2 + beta*(s[i-1])  # GARCH(1,1) model
    context.last_sigma = s[-1]
    logL = -((-np.log(s) - r**2/s).sum())
    return logL

def initialize(context):
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close(hours=0,minutes=5))
    fetch_csv('http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv', skiprows=1, \
              date_column='Date',pre_func=preview, symbol='vx1',post_func=rename_col,date_format='%Y-%m-%d')
    
    context.vxx = symbol('VXX')

    # pipe = Pipeline()
    # avg = AverageDollarVolume(window_length=1).top(1)
    # pipe.add(VIX(),'vix')
    # pipe.set_screen(avg)
    # attach_pipeline(pipe,'my_pipeline')

    context.volatility_threshhold = .02
    context.stop_price = 0
    context.stop_pct = 0.15
    
def set_trailing_stop(context, data):
    if context.portfolio.positions[context.vxx].amount:
        price = data.current(context.vxx,'price')
        sign = np.sign(context.portfolio.positions[context.vxx].amount)
        if sign > 0:
            context.stop_price = max(context.stop_price, (1-context.stop_pct) * price)
        else:
            context.stop_price = min(context.stop_price, (1+context.stop_pct) * price)
        
def my_rebalance(context,data):
    
    set_trailing_stop(context,data)
    sign = np.sign(context.portfolio.positions[context.vxx].amount)
    price = data.current(context.vxx,'price')
    
    if sign > 0 and price < context.stop_price:
        order_target(context.vxx, 0)
        context.stop_price = 0
        print("trailing stop long")
        return

    if sign < 0 and price > context.stop_price:
        order_target(context.vxx, 0)
        context.stop_price = 0
        print("trailing stop short")
        return
    
    days = 200
    r=np.array(data.history(symbol('SPY'),'price',days,'1d')[:-1])
    r=np.diff(np.log(r))
        
    R = optimize.fmin(GARCH11_logL,np.array([.1,.1,.1]),args=(r,context),full_output=1)
    print(("omega = %.6f\nbeta  = %.6f\nalpha = %.6f r=%.6f\n") % (R[0][0],R[0][2],R[0][1],r[-1]))
    
    omega = R[0][0]
    alpha = R[0][1]
    beta = R[0][2]
    
    sigma2 = omega + alpha*(r[-1])**2 + beta*context.last_sigma
    sigma = math.sqrt(sigma2)
    
    rv = 100*math.sqrt(sigma2*252)
    iv = data.current('vx1','close')
    record(rv = rv, iv = iv)
    
    delta = rv - iv
    if delta > 0.01:
        order_target_percent(context.vxx,1)
    elif delta < -0.01:
        order_target_percent(context.vxx,-1)

#
# Data pipeline functions
#
# class VIX(CustomFactor):
#     inputs = [cboe_vix.vix_close]
#     window_length = 1
    
#     def compute(self, today, assets, out, vix):
#         out[:] = vix[-1]

def rename_col(df):
    df = df.rename(columns={'VIX Close': 'close'})
    df = df.fillna(method='ffill')
    df = df[['close','sid']]
    # Correct look-ahead bias in mapping data to times   
    df = shift_to_today(df)
    log.info(' \n %s ' % df.tail())
    return df

def shift_to_today(df):
    todayts = get_environment('end')
    tdays = 1
    
    if todayts.date() > df[-1:].index[0].date() :
        tdays = max(0, len(pd.date_range(df[-1:].index[0],todayts, freq=trading_day)) - 1)
        
    log.info("Shift time to {} today={} last={}".format(tdays, todayts.date(),df[-1:].index[0].date()))    
    return df.tshift(tdays, freq=trading_day)

def preview(df):
    log.info(' \n %s ' % df.head())
    return df