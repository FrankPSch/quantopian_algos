'''
    Hurst Exponent implemented from Ernie Chan's 'Algorithmic Trading: winning strategies
    and their rationale'. One of several tools to test whether a strategy is mean-reverting
'''
import numpy

def initialize(context):
    context.past_prices = []
    context.spy = sid(8554)

def handle_data(context, data):
    hurst_val = hurst(context,data,context.spy)
    log.info(hurst_val)
    record(Hurst = hurst_val)
    
''' 
    Adjusts a list of past prices to the number of periods you want 
    So if you want the nummber of prices in the last forty days, set period = 40
'''
def gather_prices(context, data, sid, period):
    context.past_prices = data.history(sid, fields="price", bar_count=period, frequency="1d")
    if len(context.past_prices) > period:
        context.past_prices.pop(0)
    return
    
'''
    Hurst exponent helps test whether the time series is:
    (1) A Random Walk (H ~ 0.5)
    (2) Trending (H > 0.5)
    (3) Mean reverting (H < 0.5)
'''
def hurst(context, data, sid):
    # Gathers all the prices that you need
    gather_prices(context, data, sid, 128)
    
    tau, lagvec = [], []
    # Step through the different lags
    for lag in range(2,25):  
        # Produce price different with lag
        pp = numpy.subtract(context.past_prices[lag:],context.past_prices[:-lag])
        # Write the different lags into a vector
        lagvec.append(lag)
        # Calculate the variance of the difference
        tau.append(numpy.sqrt(numpy.std(pp)))
    # Linear fit to a double-log graph to get power
    m = numpy.polyfit(numpy.log10(lagvec),numpy.log10(tau),1)
    # Calculate hurst
    hurst = m[0]*2
    
    return hurst