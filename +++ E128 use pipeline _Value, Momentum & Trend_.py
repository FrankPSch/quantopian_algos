from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, Latest
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar

import numpy as np
import pandas as pd
from scipy import stats
import talib
import datetime
import calendar

#ALGORITHM INITIAL STATES
START_HOLD             = 0
START_CASH             = 0
START_HEDGED           = 0
START_NEW_TOP_TEN      = 1
START_BALANCE          = 0

START_STATES = [START_HOLD, \
                START_CASH, \
                START_HEDGED, \
                START_NEW_TOP_TEN, \
                START_BALANCE]

# DASHBOARD CONFFESSIONAL
HISTORY_PERIOD         = 200
MAX_POSITIONS          = 10

HEDGE_CHECK_INTERVAL   = 22
BOND_SMA_DAYS          = 200
GOLD_SMA_DAYS          = 200

######### STRATEGY DASHBOARD #########
######################################

#TODO Set this to true to rebalance immediately
#REBALANCE_IMMEDIATE  = True
# Use the GLD and TLT MA strat 50% max from either, cash otherwise
S_GLD_TLT_MA         = False
# Use the GLD and TLT MA, if both below, go to SHY
S_GLD_TLT_SHY_MA     = True
# Suppress or enable end of day logging (disable for BT performance)
SUPPRESS_LOGGING     = False


def initialize(context):

    set_asset_restrictions(security_lists.restrict_leveraged_etfs)
    
    schedule_function(rebalance_new_top_ten,
        date_rule=date_rules.month_start(),
        time_rule=time_rules.market_open())
    
    #schedule_function(hedge_check,
        #date_rule=date_rules.every_day(),
        #time_rule=time_rules.market_open())
    
    schedule_function(record_things,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close())
    
    context.mkt         = sid(8554) # SPY - Large cap Market cap weighted index
    context.spy         = sid(8554)
    context.rsp         = sid(8554)
    context.tlt         = sid(23921) #10 year treasury
    context.gld         = sid(23921) #10 year treasury
    context.uup         = sid(23921) #10 year treasury
    context.shy         = sid(23921) #10 year treasury
    
    context.last_month = -1
    context.days_running = 0
    context.hedge_triggered_count = 0
    context.value_stocks = []
    context.momentum_stocks = []

    context.start = True
    
    context.market = sid(8554)
    context.market_window = 200 #200
    context.atr_window = 80 #80 
    context.talib_window = context.atr_window + 5 #5
    context.risk_factor = 0.003  #0.003                   # 0.01 = less position, more % but more risk
    
    context.momentum_window_length = 180 #180
    context.market_cap_limit = 700 # 700 original # MAKE DYNAMIC?
    context.rank_table_percentile = 0.30 #0.30
    context.significant_position_difference = 0.1 #0.1
    context.min_momentum = 30.0 #30
    context.leverage_factor = 1.0    #1.0                
    context.use_stock_trend_filter = 0 #0             # either 0 = Off, 1 = On #NOT SURE IF THIS IS EVEN WORKING
    context.sma_window_length = 200 #200                # Used for the stock trend filter
    context.use_market_trend_filter = 1 #1             # either 0 = Off, 1 = On. Filter on SPY 
    context.use_average_true_range = 1 #1            # either 0 = Off, 1 = On. Manage risk with individual stock volatility
    context.average_true_rage_multipl_factor = 2.0 #2.0   # Change the weight of the ATR. 1327%
    
    
    attach_pipeline(make_pipeline(context, context.sma_window_length,
                                  context.market_cap_limit), 'screen')
     
    # Schedule my rebalance function
    schedule_function(rebalance,
                      date_rules.month_start(),  
                      time_rules.market_open(hours=0.1))
    
    # Cancel all open orders at the end of each day.
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_close())
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    

def before_trading_start(context, data):
    ''' Gather some data related to the current day and time '''
    context.selected_universe = pipeline_output('screen')
    context.assets = context.selected_universe.index
    
    context.now = datetime.datetime.now()
    context.days_running = context.days_running + 1
    context.current_month = get_datetime().month
    #set_market_benchmark(context, data)

    if context.last_month == context.current_month:
        return
    
    context.last_month = context.current_month
    set_fundamentals(context, data)
    
def handle_data(context, data):
    
    if context.start:
        set_start_state(context, data)
        context.start = False
    else:
        pass

    
def set_start_state(context, data):
    
    # TODO Test this safeguard
    if sum(START_STATES) > 1:
        log.info("ERROR: Multiple start states have been set. Only one should be set, exiting.")
        cancel = cancel
        return
    
    if START_HOLD == 1:
        return
    #if START_CASH == 1:
    #    sell_all_positions(context, data)
    if START_HEDGED == 1:
        hedge(context, data)
    if START_NEW_TOP_TEN == 1:
        rebalance_new_top_ten(context, data)
        rebalance(context, data)
    if START_BALANCE == 1:
        buy_stuff(context, data)       

def set_fundamentals(context, data):

    df = get_fundamentals(
        query(fundamentals.valuation_ratios.ev_to_ebitda,
            fundamentals.valuation_ratios.sales_yield,
            fundamentals.operation_ratios.roic)
        .filter(fundamentals.company_reference.primary_exchange_id.in_(["NYSE", "NYS"]))
        .filter(fundamentals.operation_ratios.total_debt_equity_ratio != None)         
        .filter(fundamentals.operation_ratios.total_debt_equity_ratio < 1.0)   
        #.filter(fundamentals.operation_ratios.quick_ratio > 1.0)
        .filter(fundamentals.operation_ratios.current_ratio > 1.5)
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.valuation.shares_outstanding != None)  
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
        .filter(fundamentals.asset_classification.morningstar_sector_code != None) # require sector
        .filter(fundamentals.share_class_reference.security_type == 'ST00000001') # common stock only
        .filter(~fundamentals.share_class_reference.symbol.contains('_WI')) # drop when-issued
        .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
        .filter(((fundamentals.valuation.market_cap*1.0) / (fundamentals.valuation.shares_outstanding*1.0)) > 1.0)  # stock price > $1
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
        #.filter(fundamentals.valuation.market_cap >= 10000.0E+06) #OFF
        #.filter(fundamentals.valuation.market_cap <= 250.0E+06) #OFF
        .filter(~fundamentals.company_reference.standard_name.contains(' LP')) # exclude LPs
        .filter(~fundamentals.company_reference.standard_name.contains(' L P'))
        .filter(~fundamentals.company_reference.standard_name.contains(' L.P'))
        .filter(fundamentals.balance_sheet.limited_partnership == None) # exclude LPs
        .filter(fundamentals.valuation_ratios.ev_to_ebitda != None)
        .filter(fundamentals.valuation_ratios.ev_to_ebitda >= 0.0)
        .filter(fundamentals.valuation_ratios.ev_to_ebitda <= 6.0) #NEED TO MAKE DYNAMIC (only pick bottom 10% cheapest of current market)
        #.order_by(fundamentals.valuation_ratios.ev_to_ebitda.asc())
        #.order_by(fundamentals.valuation.market_cap.desc())
        #.order_by(fundamentals.valuation_ratios.fcf_ratio.asc())
        .order_by(fundamentals.valuation_ratios.ps_ratio.asc())
        .limit(2000))
    
    dft = df.T
    
    context.score = pd.Series(0, index=dft.index)
    
    # EV/EBITDA, in-order (lower is better), nan goes last
    context.score += dft['ev_to_ebitda'].rank(ascending=True, na_option='bottom')
    
    # sales yield, inverse (higher is better), nan goes last
    context.score += dft['sales_yield'].rank(ascending=False, na_option='top')
    
    # return on invested capital, inverse (higher is better), nan goes last
    context.score += dft['roic'].rank(ascending=False, na_option='top')
    
def rebalance_new_top_ten(context, data):
    ''' Sell/buy logic '''

    
    P = data.history(context.score.index, 'price', HISTORY_PERIOD, '1d')
    V = data.history(context.score.index, 'volume', HISTORY_PERIOD, '1d')

    w = (P * V).median()
    w = w[w > 1.0E+06]

    context.score = context.score[w.index]  
    context.value_stocks = context.score.dropna().sort_values().head(MAX_POSITIONS).index

    P = data.history([context.mkt], 'price', HISTORY_PERIOD, '1d')
    u = P[context.mkt]

    if u.tail(MAX_POSITIONS).median() < u.median():
        log.info("Market below the median when rebalancing at month start, hedging.")
        hedge(context, data)
    else:        
        log.info("Generating new list of top companies and buying orders using market comparison %s." % context.mkt)
        buy_stuff(context, data)
        
def hedge_check(context, data):
    
    interval_triggered = (context.days_running % HEDGE_CHECK_INTERVAL == 0)
    okay_to_hedge = (calendar.monthrange(context.now.year, context.now.month)[1]) > 5
    
    if(interval_triggered and okay_to_hedge):

        P = data.history([context.mkt], 'price', HISTORY_PERIOD, '1d')
        u = P[context.mkt]

        if u.tail(MAX_POSITIONS).median() < u.median():
            log.info("Hedge Check - %s is below median, hedging." % context.mkt)
            hedge(context, data)
          

def hedge(context, data):            

    ''' Start S_LIBOR Strategy '''
    options = []

    ''' Start S_GLD_TLT_MA Strategy '''
    
    if(S_GLD_TLT_MA):
        # Add goldz to our buy list if above SMA
        if(symbol_above_sma(data, context.gld, GOLD_SMA_DAYS)):
            options.append(context.gld)
        if(symbol_above_sma(data, context.tlt, BOND_SMA_DAYS)):
            options.append(context.tlt)
            
    ''' Start S_GLD_TLT_SHY_MA Strategy '''
    
    if(S_GLD_TLT_SHY_MA):
        # Add goldz to our buy list if above SMA
        if(symbol_above_sma(data, context.gld, GOLD_SMA_DAYS)):
            options.append(context.gld)
        # Add bonds to our list of stocks to order if above SMA
        if(symbol_above_sma(data, context.tlt, BOND_SMA_DAYS)):
            options.append(context.tlt)
        if(len(options) == 0):
            options.append(context.shy)
    
    context.value_stocks = options
    buy_stuff(context, data)
    
    
def buy_stuff(context, data):
    ''' Buys the stocks within longs '''
    
    for s in context.portfolio.positions:
        if s in context.value_stocks:
            continue
        if not data.can_trade(s):
            continue 
        if s in context.momentum_stocks:
            continue
        order_target(s, 0)    
            
    for s in context.value_stocks:
        if s in security_lists.leveraged_etf_list.current_securities(get_datetime()):
            continue
        if not data.can_trade(s):
            continue
        order_target_percent(s, 0.5 / len(context.value_stocks))

        
def set_market_benchmark(context, data):
    if(context.now.time() < datetime.datetime(2003, 5, 1, 5, 0).time()):
        context.mkt = context.spy
    else:
        context.mkt = context.rsp
        
def benchmark_above_sma(context, data):
    ''' Calculate the market's SMA for the last SMA_THRESHOLD days '''
    market_sma = data.history(context.mkt, fields='price', bar_count=HISTORY_PERIOD+1, frequency='1d')[:-1].mean()
    current_price = data.current(context.mkt, 'price')
    #TODO: Replace 0 with SMA_BUFFER if you want
    return (current_price > ((1 + 0) * market_sma))

    
def symbol_above_sma(data, symbol, days):
    ''' Calculate the SMA for a single symbol '''
    current_symbol_price = data.current(symbol, 'price')
    symbol_sma = data.history(symbol, fields='price', bar_count=days+1, frequency='1d')[:-1].mean()
    return (current_symbol_price > symbol_sma)


def record_things(context, data):
    ''' Plot some custom signals ''' 
    market_sma = data.history(context.mkt, fields='price', bar_count=HISTORY_PERIOD+1, frequency='1d')[:-1].mean()
    current_price = data.current(sid(8554), 'price')
    #record(MKT=current_price)
    #record(ACC_LEV=context.account.leverage, TOTAL=context.portfolio.portfolio_value)
    
    
def cancel_open_orders(context, data):
    open_orders = get_open_orders()
    for security in open_orders:
        for order in open_orders[security]:
            cancel_order(order)
    
    #record(lever=context.account.leverage,
    record(Exposure=(context.account.leverage)*10)
    record(Fund_Value=context.portfolio.portfolio_value)
    pos_count = len([s for s in context.portfolio.positions if context.portfolio.positions[s].amount != 0])
    record(Stocks=(pos_count)) 

    
def rebalance(context, data):
    highs = data.history(context.assets, "high", context.talib_window, "1d")
    lows = data.history(context.assets, "low", context.talib_window, "1d")
    closes = data.history(context.assets, "price", context.market_window, "1d")
    
    estimated_cash_balance = context.portfolio.cash
    slopes = closes[context.selected_universe.index].tail(context.momentum_window_length).apply(slope)
    print slopes.order(ascending=False).head(10)
    slopes = slopes[slopes > context.min_momentum]
    ranking_table = slopes[slopes > slopes.quantile(1 - context.rank_table_percentile)].order(ascending=False)
    log.info( len(ranking_table.index))
    # close positions that are no longer in the top of the ranking table
    positions = context.portfolio.positions
    for security in positions:
        if security in context.value_stocks:
            continue
        price = data.current(security, "price")
        position_size = positions[security].amount
        if data.can_trade(security) and security not in ranking_table.index:
            order_target(security, 0, style=LimitOrder(price)) #MARKET ORDER
            if security in context.momentum_stocks:
                context.momentum_stocks.remove(security)
            estimated_cash_balance += price * position_size
        elif data.can_trade(security):
            new_position_size = get_position_size(context, highs[security], lows[security], closes[security],security)
            if significant_change_in_position_size(context, new_position_size, position_size):
                estimated_cost = price * (new_position_size * context.leverage_factor - position_size)
                order_target(security, new_position_size * context.leverage_factor, style=LimitOrder(price)) #MARKET ORDER
                context.momentum_stocks.append(security)
                estimated_cash_balance -= estimated_cost
    
    
    # Market history is not used with the trend filter disabled
    # Removed for efficiency
    if context.use_market_trend_filter:
        market_history = data.history(context.market, "price", context.market_window, "1d")  ##SPY##
        current_market_price = market_history[-1]
        average_market_price = market_history.mean()
    else:
        average_market_price = 0
    
    if (current_market_price > average_market_price) :  #if average is 0 then jump in
        for security in ranking_table.index:
            if data.can_trade(security) and security not in context.portfolio.positions:
                new_position_size = get_position_size(context, highs[security], lows[security], closes[security],
                                                     security)
                estimated_cost = data.current(security, "price") * new_position_size * context.leverage_factor
                if estimated_cash_balance > estimated_cost:
                    order_target(security, new_position_size * context.leverage_factor, style=LimitOrder(data.current(security, "price"))) #MARKET ORDER
                    context.momentum_stocks.append(security)
                    estimated_cash_balance -= estimated_cost
    
     
def get_position_size(context, highs, lows, closes, security):
    try:
        average_true_range = talib.ATR(highs.ffill().dropna().tail(context.talib_window),
                                       lows.ffill().dropna().tail(context.talib_window),
                                       closes.ffill().dropna().tail(context.talib_window),
                                       context.atr_window)[-1] # [-1] gets the last value, as all talib methods are rolling calculations#
        if not context.use_average_true_range: #average_true_range
            average_true_range = 1 #divide by 1 gives... same initial number
            context.average_true_rage_multipl_factor = 1
        
        return (context.portfolio.portfolio_value * context.risk_factor)  / (average_true_range * context.average_true_rage_multipl_factor) 
    except:
        log.warn('Insufficient history to calculate risk adjusted size for {0.symbol}'.format(security))
        return 0
        

def significant_change_in_position_size(context, new_position_size, old_position_size):
    return np.abs((new_position_size - old_position_size)  / old_position_size) > context.significant_position_difference

def slope_(ts): ## new version log(log(ts))
    x = np.arange(len(ts))  
    log_ts = np.log(np.log(ts))  
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)  
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100  
    return annualized_slope * (r_value ** 2)     

def slope(ts): ## new version
    x = np.arange(len(ts))  
    log_ts = np.log(ts)  
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)  
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100 
    return annualized_slope * (r_value ** 2)

def slope_v(ts): # new (Vladimir)
    x = np.arange(len(ts))
    log_ts = np.log(ts) 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = ((1 + slope)**250 -1.0) * 100 
    return annualized_slope * (r_value ** 2) 

def _slope(ts): # original (James?)
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = ((1 + slope)**250 ) * 100 
    return annualized_slope * (r_value ** 2) 


class MarketCap(CustomFactor):   
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        

def make_pipeline(context,sma_window_length, market_cap_limit):
    pipe = Pipeline()  
    
    # Now only stocks in the top N largest companies by market cap
    market_cap = MarketCap()
    top_N_market_cap = market_cap.top(market_cap_limit)
    
    #Other filters to make sure we are getting a clean universe
    is_primary_share = morningstar.share_class_reference.is_primary_share.latest
    is_not_adr = ~morningstar.share_class_reference.is_depositary_receipt.latest
    
    #### TREND FITLER ##############
    #### We don't want to trade stocks that are below their sma_window_length(100) moving average price.
    if context.use_stock_trend_filter:
        latest_price = USEquityPricing.close.latest
        sma = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=sma_window_length)
        above_sma = (latest_price > sma)
        initial_screen = (above_sma & top_N_market_cap & is_primary_share & is_not_adr)
        log.info("Init: Stock trend filter ON")
    else: #### TREND FITLER OFF  ##############
        initial_screen = (top_N_market_cap & is_primary_share & is_not_adr)
        log.info("Init: Stock trend filter OFF")

    pipe.add(market_cap, "market_cap")
    
    pipe.set_screen(initial_screen)
    
    return pipe