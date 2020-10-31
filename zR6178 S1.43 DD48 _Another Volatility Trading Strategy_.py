import pandas as pd
import numpy as np
import math
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor
from quantopian.pipeline.data.quandl import cboe_vix
from scipy import optimize

NDays = 5                                    #5


class ImpliedVolatility(CustomFactor):
    inputs = [cboe_vix.vix_close]
    outputs = ['md', 'lst'] 
    def compute(self, today, assets, out, vix):
        df = pd.DataFrame(np.array(vix), columns=['vix'])
        out.md[:] = df.quantile(.5)
        out.lst[:] = vix[-1]

        
def initialize(context):
    
    # using less volatile instruments SPY and IEF
    context.vxx = sid(38054)
    context.xiv = sid(40516)
    context.spy = sid(8554)
    
    set_benchmark(context.xiv)
    
    pipe = Pipeline()
    attach_pipeline(pipe, 'example')
    
    iv = ImpliedVolatility(window_length = NDays)
    pipe.add(iv, 'iv')
    
    schedule_function(func=allocate, 
                      time_rule=time_rules.market_open(),
                      half_days=True)

    
def before_trading_start(context, data):

    output = pipeline_output('example')
    output = output.dropna() 
    #implied vol
    md, lst = output["iv"].iloc[0]
    context.hv = calculate_hv(context, data, NDays)  
    
    calculate_gv(context, data, context.spy)
    
    context.vrp = md-context.hv
    context.volUp = context.gv1-lst
    context.volUp2 = context.gv2 - context.gv1

    record(hv = context.hv, iv = md, gv1 = context.gv1, gv2 = context.gv2)

    
def calculate_hv(context, data, days):    
    close = data.history(context.spy, ["price"], (days+1), "1d")
    close["ret"] = (np.log(close.price) - np.log(close.price).shift(1))
    return close.ret.std()*math.sqrt(252)*100


def GARCH11_logL(param, r, context):
    omega, alpha, beta = param
    n = len(r)
    s = np.ones(n)*0.01
    s[2] = np.var(r[0:3])
    for i in range(3, n):
        s[i] = omega + alpha*r[i-1]**2 + beta*(s[i-1])  # GARCH(1,1) model
    context.last_sigma2 = s[-1]
    logL = -((-np.log(s) - r**2/s).sum())
    return logL


def calculate_gv(context, data, asset):
    days = 200
    r=np.array(data.history(asset,'price',days,'1d')[:-1])
    r=np.diff(np.log(r))
        
    R = optimize.fmin(GARCH11_logL,np.array([.1,.1,.1]),args=(r,context),full_output=1)
    #print("omega = %.6f\nbeta  = %.6f\nalpha = %.6f r=%.6f\n") % (R[0][0],R[0][2],R[0]                    [1],r[-1])
    omega = R[0][0]
    alpha = R[0][1]
    beta  = R[0][2]
    
    uconSigma2 = omega/(1-alpha-beta)
    
    #calculation of GV(t+1)
    sigma21 = omega + alpha*(r[-1])**2 + beta*context.last_sigma2
    
    #calculation of GV(t+2)
    sigma22 = uconSigma2+(alpha+beta)**2*(context.last_sigma2-uconSigma2)
    
    #normalised to percentage
    context.gv1 = 100*math.sqrt(sigma21*252)
    context.gv2 = 100*math.sqrt(sigma22*252)
    
    return 
    

def allocate(context, data):

    if context.volUp > 0 and context.volUp2 > 0 and context.vrp < 0 :
        order_target_percent(context.vxx, 1.00)
        order_target_percent(context.xiv, 0.00)
        
    if context.vrp > 0  and context.volUp < 0:
        order_target_percent(context.vxx, 0.00)
        order_target_percent(context.xiv, 1.00)
    else:    
        order_target_percent(context.vxx, 0.00)
        order_target_percent(context.xiv, 0.00)