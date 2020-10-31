#####################################################################
# Trend following algo
# Naoki Nagai, 2015
#####################################################################
# This is a trend following algo for varieties of uncorrelated assets.
# Entry signal: Regression line slope exceeds + or - 1% per day and cross the regression line
# Profit take : 1.96 standard deviation (95% bollinger band)
# Stop loss   : Trailing stop with percentage = regression line slope * look back period

from numpy import isnan, matrix, array, zeros, empty, sqrt, round, ones, dot, append, mean, cov, transpose, linspace
import numpy as np
import talib
import pandas as pd
import scipy.optimize
import operator
from pytz import timezone
from zipline.utils.tradingcalendar import get_early_closes
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

# Initialization
def initialize(context):
    set_symbol_lookup_date('2015-01-01')
    context.lookback = 252/2
    context.maxlever = 0.9       # Always hold 10% cash
    context.multiple = 5.0       # 1% of annual return translate to what weight? e.g. 5%
    context.profittake = 1.96    # 95% bollinger band
    load_symbols(context)
    context.weights = dict.fromkeys(context.secs, 0)
    context.stopprice = dict.fromkeys(context.secs, None) 
    
    context.PctDailyVolatilityTarget = 0.025 # target daily vol target in %
    
    schedule_function(trail_stop, date_rules.every_day(),  time_rules.market_open(minutes = 10))
    schedule_function(regression, date_rules.every_day(),  time_rules.market_open(minutes = 28))
    schedule_function(trade, date_rules.every_day(),  time_rules.market_open(minutes = 30))

# Calculate volatility adjustment    
def calc_vol_scalar(context, data):
    prices = data.history(context.secs, 'open', context.lookback, '1d')
    rets = np.log(prices).diff().dropna()
    block_value = prices.iloc[-1]
    price_vol = calc_std(rets)
    volatility_scalar = context.PctDailyVolatilityTarget / price_vol
    return volatility_scalar
    
    
def calc_std(returns):
    downside_only = False
    if (downside_only):
        returns = returns.copy()
        returns[returns > 0.0] = np.nan
    b = pd.ewmstd(returns, halflife=20, adjust=True, ignore_na=True).dropna()  #halflife = 20 four week half life - mid-term
    return b.iloc[-1] 


# Calculate the slopes for different assetes    
def regression(context, data):
    prices = data.history(context.secs, 'open', context.lookback, '1d')
    X=list(range(len(prices)))
        
    # Add column of ones so we get intercept
    A = sm.add_constant(X)
        
    for s in context.secs:
        if not data.can_trade(s): 
            continue
        
        # Price movement
        sd = prices[s].std() 

        # Price points to run regression
        Y = prices[s].values

        # If all empty, skip
        if isnan(Y).any():
            continue
        
        # Run regression y = ax + b
        results = sm.OLS(Y,A).fit()
        (b, a) =results.params
        
        # Normalized slope
        slope = a / b * 252.0        # Daily return regression * 1 year
        
        # Currently how far away from regression line?
        delta = Y - (dot(a,X) + b)
        
        # Don't trade if the slope is near flat 
        slope_min = 0.252        # At least %7 growth per year to trade
        
        # Current gain if trading
        gain = get_gain(context, s) * 100
        
        # Long but slope turns down, then exit
        if context.weights[s] > 0 and slope < 0:
            context.weights[s] = 0
            loggain('v %+2d%% Slope turn bull  %3s - %s' %(gain, s.symbol, s.security_name),gain)

        # Short but slope turns upward, then exit
        if context.weights[s] < 0 and 0 < slope:
            context.weights[s] = 0
            loggain('^ %+2d%% Slope turn bear  %3s - %s' %(gain, s.symbol, s.security_name),gain)

        # Trend is up
        if slope > slope_min:
            # Price crosses the regression line
            if delta[-1] > 0 and delta[-2] < 0 and context.weights[s] == 0:
                context.stopprice[s] = None
                context.weights[s] = slope
                loggain('/     Long  a = %+.2f%% %3s - %s' %(slope*100, s.symbol, s.security_name),gain)
                
            # Profit take, reaches the top of 95% bollinger band
            if delta[-1] > context.profittake * sd and context.weights[s] > 0:
                context.weights[s] = 0        
                loggain('//%+2d%% Long exit %3s - %s'%(gain, s.symbol, s.security_name) ,gain)

        # Trend is down
        if slope < -slope_min:
            # Price crosses the regression line
            if delta[-1] < 0 and delta[-2] > 0 and context.weights[s] == 0:
                context.stopprice[s] = None
                context.weights[s] = slope
                loggain('\     Short a = %+.2f%% %3s - %s' %(slope*100, s.symbol, s.security_name))
                
            # Profit take, reaches the top of 95% bollinger band
            if delta[-1] < - context.profittake * sd and context.weights[s] < 0:
                context.weights[s] = 0        
                loggain('\\%+2d%% Short exit %3s - %s' %(gain, s.symbol, s.security_name),gain)

    return context.weights

def get_gain(context, s):
    if s in context.portfolio.positions:
        cost = context.portfolio.positions[s].cost_basis
        amount = context.portfolio.positions[s].amount 
        price = context.portfolio.positions[s].last_sale_price
        if amount > 0:
            gain = price/cost - 1        
        if amount < 0:
            gain = 1 - price/cost
    else:
        gain = 0
    return gain 


def trade(context, data):
    vol_mult = calc_vol_scalar(context, data)
    w = context.weights
    record(leverage = context.account.leverage)
    record(equities = sum(w[s] for s in context.equities))
    record(fixedincome = sum(w[s] for s in context.fixedincome ))
    record(alternative = sum(w[s] for s in context.alternative))
    record(cash = max(0,context.portfolio.cash) / context.portfolio.portfolio_value)

    no_positions = 0
    for s in context.secs:
        if w[s] != 0:
            no_positions += 1
            
    for s in context.secs:
        if data.can_trade(s) and s not in get_open_orders():
            if w[s] == 0:
                order_target_percent(s, 0)
            if w[s] > 0:
                order_target_percent(s, (min(w[s] * context.multiple, context.maxlever)/no_positions)*vol_mult[s])
              
            if w[s] < 0:
                order_target_percent(s, (max(w[s] * context.multiple, -context.maxlever)/no_positions)*vol_mult[s])

def trail_stop(context, data):
    for s in context.secs:
        if not data.can_trade(s): 
            continue

        price = data.history(s, 'price', 3, '1d').mean()

        gain = get_gain(context, s) * 100
        
        # Stop loss percentage is the return over the lookback period
        stoploss = abs(context.weights[s] * context.lookback / 252) + 1    # percent change per period
        
        if context.weights[s] > 0:
            if context.stopprice[s] < 0:
                context.stopprice[s] = price / stoploss
                
            else:
                context.stopprice[s] = max(price / stoploss, context.stopprice[s])
                if price < context.stopprice[s] :
                    loggain('x %+2d%% Long  stop loss  %3s - %s' %(gain, s.symbol, s.security_name,),gain)
                    context.weights[s] = 0        
                    order_target_percent(s,0)

        elif context.weights[s] < 0:
            if context.stopprice[s] < 0:
                context.stopprice[s] = price * stoploss

            else:
                context.stopprice[s] = min(price * stoploss, context.stopprice[s])
                if price > context.stopprice[s]:
                    loggain('x %+2d%% Short stop loss  %3s - %s' %(gain, s.symbol, s.security_name,),gain)
                    context.weights[s] = 0        
                    order_target_percent(s,0)
                
        else:
            context.stopprice[s] = None

        #record(stoploss = context.stopprice[s])

def handle_data(context, data):
    exchange_time = pd.Timestamp(get_datetime()).tz_convert('US/Eastern')
    if exchange_time.minute % 5 == 0:    # Check trailing stop every 5 minutes
        trail_stop(context,data)

        
def loggain(text, gain=0):
    # Loss settle is WARN and gain settle is INFO
    if gain < 0:
        log.warn(text)
    else:
        log.info(text)

            
def load_symbols(context) :
    context.equities = symbols(
        # Equity
        'DIA',    # Dow
        'QQQ',    # NASDAQ
    )
    context.fixedincome = symbols(
        # Fixed income
        'LQD',    # Corporate bond
        'HYG',    # High yield
    )
    context.alternative = symbols(
        'USO',    # Oil
        'GLD',    # Gold
        'VNQ',    # US Real Estate
        'RWX',    # Dow Jones® Global ex-U.S. Select Real Estate Securities Index
        'UNG',    # Natual gas
        'DBA',    # Agriculture
    )
    context.secs = context.equities + context.fixedincome + context.alternative