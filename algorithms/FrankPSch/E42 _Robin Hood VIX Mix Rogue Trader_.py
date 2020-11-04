import functools
import itertools
import math
import re
import talib
import time
import numpy as np
import pandas as pd
from datetime import datetime  
from pytz import timezone      
from scipy import stats
from zipline.utils import tradingcalendar

History = 128

def initialize(context):

    ##
    # CherryPicker
    ##

    context.rsi_length = 3
    context.rsi_trigger = 50
    context.wvf_length = 100
    context.ema1 = 10
    context.ema2 = 30
    context.sell = False

    ##
    # Not CherryPicker
    ##
    
    vixUrl = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
    vxvUrl = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vxvdailyprices.csv'
    nokoUrl = 'http://52.15.233.150/noko.csv'  
    
    fetch_csv(nokoUrl, 
              symbol='v1', 
              date_column='Date', 
              date_format='%Y-%m-%d', 
              pre_func=addFieldsVX1, 
              post_func=shift_data)
    
    fetch_csv(nokoUrl, 
              symbol='v2', 
              date_column='Date', 
              date_format='%Y-%m-%d', 
              pre_func=addFieldsVX2, 
              post_func=shift_data)

    fetch_csv(vixUrl, 
              symbol='v', 
              skiprows=1,
              date_column='Date', 
              pre_func=addFieldsVIX,
              post_func=shift_data)

    fetch_csv(vxvUrl, 
              symbol='vxv', 
              skiprows=2,
              date_column='Date', 
              pre_func=addFieldsVXV,
              post_func=shift_data)

    context.xiv = sid(40516)
    context.tqqq = sid(39214)
    context.uvxy = sid(41969)
    context.spyg = sid(22009)
    
    context.vix = -1
    context.xiv_day = 0
    set_benchmark(context.xiv) # ALPHA and BETA will be different
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) # FSC for IB
    
    context.SetAsideLeverageTotal = 0.20 # 100% allocate when BigSignalTQQQ
    context.VIX_GrowthLeverage    = 1 - context.SetAsideLeverageTotal
    context.VIX_MinHedgeLeverage  = context.VIX_HedgeLeverage  = 0.666
    context.VIX_MaxHedgeLeverage  = 1.00 # if certain conditions are met
    context.Agree                 = False
    context.ShowMaxLev            = True

    # 3x NASDAQ nonfinancial, generally profitable
    context.SetAsideStocks = symbols('TQQQ')

    context.PriceXIV = context.PriceUVXY = 0.00
    context.BoughtShortVIX = context.BoughtLongVIX = False
    context.XIV       = symbol('XIV')
    context.UVXY      = symbol('UVXY') # approx 8% weekly erosion, bigger spikes
    context.VIXstocks = (context.XIV, context.UVXY)
    context.LongVIX  = False  # no decisions until one of three conditions exists
    context.ShortVIX = False  # no decisions until one of three conditions exists
    # not necessary to set BigSignal variables because they get set by MoreSignals function early every day
    
    # apparently highly successful VIX signals
    schedule_function(CherryPickerOpen, date_rules.every_day(), time_rules.market_open(), False)
    schedule_function(MoreSignals, date_rules.every_day(), time_rules.market_open(minutes = 2))
    schedule_function(Rebalance, date_rules.every_day(), time_rules.market_open(minutes = 4))
    schedule_function(RogueTrader, date_rules.every_day(),time_rules.market_open(minutes = 60))
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_close())    
    schedule_function(RecordVars, date_rules.every_day(), time_rules.market_close())    
    schedule_function(CherryPickerClose, date_rules.every_day(), time_rules.market_close(), False)

    #VXX used for strategy to buy XIV
    context.VXX = context.vxx = sid(38054) # VXX

    #Editable values to tweak backtest
    context.AvgLength = 20
    context.LengthWVF = 100
    context.LengthEMA1 = 10
    context.LengthEMA2 = 30
    
    #internal variables to store data
    context.vxxAvg = 0 
    context.SmoothedWVF1 = 0
    context.SmoothedWVF2 = 0
    context.vxxLow = 0
    context.vxxHigh = 0
    
    #Internal stop variables
    # context.stoploss is set in RogueTrader
    context.stoploss = 0.50
    context.stops         = {}
    context.stoppedout    = []
    context.SellAndWait   = 0
    
def before_trading_start(context, data):

    context.RogueTrader = False # Not True until RogueTrader scheduled function runs

    context.mx_lvrg  = 0 # daily max leverage

    # To help determine previous signal when start algo in live trade Robinhood
    if not context.ShortVIX and not context.LongVIX: # Maybe we can figure it out
        context.ShortVIX = True if 0 < context.portfolio.positions[context.XIV].amount  else False
        context.LongVIX  = True if 0 < context.portfolio.positions[context.UVXY].amount else False
        if context.ShortVIX and context.LongVIX: # Do no harm
            context.ShortVIX = context.LongVIX = False

def RecordVars(context, data):
    
    # handling this in handle_data now to show mx_lvrg
    if not context.ShowMaxLev:
        record(Leverage=context.account.leverage)
        pass

def CherryPickerOpen(context,data):
    c = context
    if c.sell:
        c.sell = False
        if c.portfolio.positions[c.XIV].amount > 0: 
            cancel_open_orders_for(context, data, c.XIV)
            TarPer(context, data, c.XIV, 0.00)
            
def RogueTrader(context, data):

    c = context
    c.RogueTrader = True # Not True until RogueTrader scheduled function runs

    cancel_open_orders(context, data)

    ##
    # Limit sell and Stop loss orders
    ##
    for stock in c.portfolio.positions:

        # Set stoploss
        c.stoploss = 0.21 if stock is c.UVXY else 0.035

        # Set LimitPrice for first order
        RemainingFactor  = 0.50 if stock is c.UVXY else 0.22
        MaxOrders        = 20   if stock is c.UVXY else 3
        if stock is c.UVXY:
            LimitPriceFactor = 1.06 if c.Agree else 1.05
        else:
            LimitPriceFactor = 1.04
        LimitPrice       = LimitPriceFactor * c.portfolio.positions[stock].cost_basis

        SharesRemaining = int(RemainingFactor * c.portfolio.positions[stock].amount)
        SharesPerOrder  = min(SharesRemaining, max(100, int(SharesRemaining / MaxOrders)))
        
        while 0 < SharesRemaining:

            order(stock, -SharesPerOrder, style=LimitOrder(LimitPrice))
            
            SharesRemaining -= SharesPerOrder
            SharesPerOrder = min(SharesRemaining, SharesPerOrder)
            LimitPrice *= LimitPriceFactor 
        if stock.symbol in c.stops: del c.stops[stock.symbol]
        
def Rebalance(context, data):
    c = context
    
    if 0 < c.SellAndWait:
        c.SellAndWait -= 1
        return

    cancel_open_orders(context, data) # avoid confusion

    ### Determine how much to risk in Long VIX ###
    # Load historical data for the stocks  
    hist = data.history(c.vxx, ['high', 'low', 'close'], 15, '1d')  
    # Calculate the ATR for the stock  
    atr_14 = talib.ATR(hist['high'],  
                    hist['low'],  
                    hist['close'],  
                    timeperiod=14)[-1]  
    atr_3 = talib.ATR(hist['high'],  
                    hist['low'],  
                    hist['close'],  
                    timeperiod=3)[-1]
    c.VIX_HedgeLeverage = c.VIX_MaxHedgeLeverage if atr_3 <= atr_14 else c.VIX_MinHedgeLeverage

    #Gets Moving Average of VXX
    price_hist = data.history(c.vxx, 'price', c.AvgLength, '1d')
    c.vxxAvg = price_hist.mean()
    
    #get data for calculations
    n = 200
    vxx_prices = data.history(c.vxx, "price", n + 2, "1d")
    vxx_lows = data.history(c.vxx, "low", n + 2, "1d")
    vxx_highest = vxx_prices.rolling(window = c.LengthWVF,center=False).max()
    
    #William's VIX Fix indicator a.k.a. the Synthetic VIX
    WVF = ((vxx_highest - vxx_lows)/(vxx_highest)) * 100
    
    # calculated smoothed WVF
    c.SmoothedWVF1 = talib.EMA(WVF, timeperiod=c.LengthEMA1) 
    c.SmoothedWVF2 = talib.EMA(WVF, timeperiod=c.LengthEMA2)
    
    #Do some checks for cross overs. 
    if WVF[-1] > c.SmoothedWVF1[-1] and WVF[-2] < c.SmoothedWVF1[-1]:
        wvf_crossedSmoothedWVF1 = True
    else: 
        wvf_crossedSmoothedWVF1 = False
    #Same except for smoothed2 
    if WVF[-1] > c.SmoothedWVF2[-1] and WVF[-2] < c.SmoothedWVF2[-1]:
        wvf_crossedSmoothedWVF2 = True
    else: 
        wvf_crossedSmoothedWVF2 = False
    
    #Current price of vxx
    vxxPrice = data.current(c.vxx, 'price')
    
    #SignalShortVIX 
    if ( wvf_crossedSmoothedWVF1 and WVF[-1] < c.SmoothedWVF2[-1]) or (wvf_crossedSmoothedWVF2 and wvf_crossedSmoothedWVF1):
        SignalShortVIX = True
    elif ((vxxPrice > c.vxxAvg and vxx_prices[-2] < c.vxxAvg) or (WVF[-1] < c.SmoothedWVF2[-1] and WVF[-2] > c.SmoothedWVF2[-1])):
        SignalShortVIX = False
    else:
        SignalShortVIX = True

    if (c.BigSignalShortVIX and SignalShortVIX): # Agree ShortVIX
        c.ShortVIX    = True
        c.LongVIX     = False
        c.Agree       = True

    elif (c.BigSignalLongVIX and not SignalShortVIX): # Agree LongVIX
        c.ShortVIX    = False
        c.LongVIX     = True
        c.Agree       = True

    elif (c.BigSignalLongVIX and SignalShortVIX): # Not Agree LongVIX
        c.ShortVIX    = False
        c.LongVIX     = True
        c.Agree       = False

    ##
    # Rebalance only once until signal changes again
    # or
    # leverage drops too low
    ##

    l = c.account.leverage
    LevTooLow = True if (0.33 > l) or (0.90 < l < 1.05) else False
    p = c.portfolio.positions

    if c.BigSignalTQQQ:
        if LevTooLow or (c.XIV in p or c.UVXY in p):
            TarPer(context, data, c.XIV, 0.00)
            TarPer(context, data, c.UVXY, 0.00)
            SetAsideStocks = c.SetAsideStocks
            SetAsideLeveragePositions = len(SetAsideStocks)
            for stock in SetAsideStocks:
                if not DataCanTrade(context, data, stock):
                    SetAsideStocks.remove(stock)
                    SetAsideLeveragePositions -= 1
            SetAsideLeverage = float(1.00 / SetAsideLeveragePositions) if 0 < SetAsideLeveragePositions else 0.00
            for stock in SetAsideStocks:
                TarPer(context, data, stock, SetAsideLeverage)

    elif c.ShortVIX:
        if LevTooLow or (c.XIV not in p):
            TarPer(context, data, c.UVXY, 0.00)
            SetAsideStocks = c.SetAsideStocks
            SetAsideLeveragePositions = len(SetAsideStocks)
            for stock in SetAsideStocks:
                if not DataCanTrade(context, data, stock):
                    SetAsideStocks.remove(stock)
                    SetAsideLeveragePositions -= 1
            SetAsideLeverage = float(c.SetAsideLeverageTotal / SetAsideLeveragePositions) if 0 < SetAsideLeveragePositions else 0.00
            for stock in SetAsideStocks:
                TarPer(context, data, stock, SetAsideLeverage)
            c.BoughtShortVIX = False
            c.PriceXIV = 0.00

    elif c.LongVIX:
        if LevTooLow or (c.UVXY not in p):
            TarPer(context, data, c.XIV, 0.00)
            for stock in c.SetAsideStocks:
                TarPer(context, data, stock, 0.00)
            c.BoughtLongVIX = False
            c.PriceUVXY = 0.00

    # Record / log stuff I want to know
    c.XIVprice  = data.current(c.XIV, 'price')
    c.UVXYprice = data.current(c.UVXY, 'price')
    TQQQprice = data.current(c.tqqq, 'price')
    #record(XIV  = c.XIVprice)
    #record(UVXY = c.UVXYprice)
    #record(TQQQ = TQQQprice)
    BigSignal = 'NoBigSignal'
    BigSignal = 'ShortVIX' if c.BigSignalShortVIX else BigSignal
    BigSignal = ' LongVIX' if c.BigSignalLongVIX  else BigSignal
    BigSignal = '    TQQQ' if c.BigSignalTQQQ     else BigSignal
    SyntheticVIX = 'NoSyntheticVIX'
    SyntheticVIX = 'ShortVIX' if c.ShortVIX else SyntheticVIX
    SyntheticVIX = ' LongVIX' if c.LongVIX  else SyntheticVIX
    log.info('BigSignal / SyntheticVIX: {} / {}     VIX: {:.2f}     XIV: {:.2f}     UVXY: {:.2f}     TQQQ: {:.2f}'
        .format(BigSignal, SyntheticVIX, c.VIXprice, c.XIVprice, c.UVXYprice, TQQQprice)
    )

def TarPer(context, data, stock, TargetPercent):

    if DataCanTrade(context, data, stock):

        if 0 == TargetPercent:
            order_target_percent(stock, 0.00)
        else:
            # Always want money available to withdraw 
            # and also try to prevent margin related order rejections
            PV = context.portfolio.portfolio_value
            DoNotSpend = 2000 # 200 Cushion plus cash for withdrawals
            RhMargin = 24000.00 # Set to 0 if you do not have Robinhood Gold
            MaxLeverage = 1.33 - .12 # Hard Limit for leverage minus seemingly necessary cushion
            MaxLeverage = min(MaxLeverage, max(MaxLeverage, context.account.leverage))
            RhMargin = min(RhMargin, PV * (MaxLeverage - 1))
            RhPV = PV + RhMargin - DoNotSpend  
            RhCash = RhPV - context.portfolio.positions_value
            amount = context.portfolio.positions[stock].amount
            price = data.current(stock, 'price')
            PosValue   = float(amount * price)
            TarValue   = float(RhPV * TargetPercent)
            DiffValue  = float(TarValue - PosValue)
            DiffValue  = min(DiffValue, RhCash)
            DiffAmount = int(DiffValue / price)
            DiffAmount = 0 if 0 > DiffAmount and 0 == amount else DiffAmount
            order(stock, DiffAmount)

        if stock.symbol in context.stops: del context.stops[stock.symbol]

def DataCanTrade(context, data, stock):

    try:
        if data.can_trade(stock):
            return True
        else:
            return False
    except:
        return False

def cancel_open_orders(context, data):
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        for order in orders:
            cancel_order(order)
            #message = 'Canceling order of {amount} shares in {stock}'
            #log.info(message.format(amount=order.amount, stock=stock))
    
def cancel_open_orders_for(context, data, security):
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        for order in orders:
            if stock is security:
                if order.amount < context.portfolio.positions[stock].amount:
                    cancel_order(order)
                    #message = 'Canceling order of {amount} shares in {stock}'
                    #log.info(message.format(amount=order.amount, stock=stock))
                else: # Do NOT want to cancel stop loss order
                    #message = 'NOT Canceling stop loss order of {amount} shares in {stock}'
                    #log.info(message.format(amount=order.amount, stock=stock))
                    pass

def MoreSignals(context, data):  

    update_indices(context, data)     
    last_vix = context.VIXprice = data.current('v', 'Close')
    last_vx1 = data.current('v1','Close')  
    last_vx2 = data.current('v2','Close')      
    last_vxv = data.current('vxv', 'Close')
    last_vix_200ma_ratio = data.current('v', '200ma Ratio')
           
    # Calculating the gap between spot vix and the first month vix future
    last_ratio_v_v1 = last_vix/last_vx1

    # Calculating the contango ratio of the front and second month VIX Futures 
    last_ratio_v1_v2 = last_vx1/last_vx2

    # Blending the previous two ratios together using a weighted average
    ratio_weight = 0.7
    last_ratio = (ratio_weight*last_ratio_v_v1) + ((1-ratio_weight)*last_ratio_v1_v2) - 1
    
    vix_vxv_ratio = last_vix/last_vxv
    
    # Retrieve SPY prices for technical indicators
    prices = data.history(context.spyg, 'open', 40, '1d')
    
    # Retrieve SPY MACD data
    macda, signal, hist = talib.MACD(prices, fastperiod=12,slowperiod=26,signalperiod=9)
    macd = macda[-1] - signal[-1]
    
    # Calculate how much vix moved the previous day
    if (context.vix <> -1) : 
        vix_ratio = last_vix/context.vix -1
    else :
        vix_ratio = 0
    context.vix = last_vix
    
    xiv_history = data.history(context.xiv, 'price', 2, '1d')  
    
    xiv_ratio = xiv_history[1]/xiv_history[0] - 1
    
    # Setting thresholds
    threshold_vix_too_low = 10.76   # 0 
    threshold_vix_200ma_ratio_low = 0.79  # 0 
    threshold_xiv = -0.049          # 1
    threshold_vxv_xiv = 0.87        # 2
    threshold_uvxy = 0.049          # 3
    threshold_macd = -0.55          # 3
    threshold_vxv_uvxy = 1.3        # 4
    threshold_vix_high = 19.9       # 5
    threshold_vc_low = -0.148       # 6
    threshold_vc_high = 0.046       # 8
    threshold_vc_high_2 = -0.06     # 8
    threshold_xiv_ratio = -0.053    # 10
    threshold_uvxy_ratio = 0.08     # 11

    # 0
    if last_vix < threshold_vix_too_low and last_vix_200ma_ratio < threshold_vix_200ma_ratio_low: # if VIX is too low, invest in UVXY witht he hope of a spike
        target_sid = context.uvxy
    # 1        
    elif last_ratio < threshold_xiv: # if contango is high, invest in XIV to gain from decay
        target_sid = context.xiv
    
    # 2
    elif vix_vxv_ratio < threshold_vxv_xiv: # if short term vol is low compared to mid term, invest in XIV to gain from decay
        target_sid = context.xiv

    # 3
    elif last_ratio > threshold_uvxy and macd > threshold_macd: # if backwardation is high, invest in UVXY to gain from decay
        target_sid = context.uvxy

    # 4
    elif vix_vxv_ratio > threshold_vxv_uvxy: # if short term vol is high compared to mid term, invest in UVXY to gain from growth
        target_sid = context.uvxy

    # 5
    elif last_vix > threshold_vix_high: # if VIX is too high, invest in XIV expecting that VIX will drop
        target_sid = context.xiv

    # 6
    elif vix_ratio < threshold_vc_low: # Vix down sharply, invest in XIV expecting that futures curve gets pulled down
        target_sid = context.xiv

    # 7
    elif vix_ratio > threshold_vc_high: # Vix up sharply, invest in UVXY expecting that futures curve gets pulled up
        target_sid = context.uvxy

    # 8
    elif vix_ratio > threshold_vc_high_2: #have to think
        target_sid = context.xiv

    # 9
    else:
        target_sid = context.uvxy

    # 10
    if (target_sid == context.xiv and xiv_ratio < threshold_xiv_ratio) : 
        # indicators say XIV but it just dropped overnight, so go for TQQQ
        target_sid = context.tqqq 

    # 11
    elif (target_sid == context.uvxy and xiv_ratio > threshold_uvxy_ratio) :
        # indicators say UVXY but it just dropped overnight, so go for TQQQ
        target_sid = context.tqqq
    
    context.BigSignalShortVIX = True if context.xiv  is target_sid else False
    context.BigSignalLongVIX  = True if context.uvxy is target_sid else False
    context.BigSignalTQQQ     = True if context.tqqq is target_sid else False
    
def update_indices(context, data):
    context.fetch_failed = False
    context.vix_vals = unpack_from_data(context, data, 'v')    
    context.vxv_vals = unpack_from_data(context, data, 'vxv')  
    context.vx1_vals = unpack_from_data(context, data, 'v1')
    context.vx2_vals = unpack_from_data(context, data, 'v2')

def fix_close(df,closeField):
    df = df.rename(columns={closeField:'Close'})
    # remove spurious asterisks
    df['Date'] = df['Date'].apply(lambda dt: re.sub('\*','',dt))
    # convert date column to timestamps
    df['Date'] = df['Date'].apply(lambda dt: pd.Timestamp(datetime.strptime(dt,'%m/%d/%Y')))
    df = df.sort_values(by='Date', ascending=True)
    return df

def fix_closeVX(df,closeField):
    df = df.rename(columns={closeField:'Close'})
    # remove spurious asterisks
    df['Date'] = df['Date'].apply(lambda dt: re.sub('\*','',dt))
    # convert date column to timestamps
    df['Date'] = df['Date'].apply(lambda dt: pd.Timestamp(datetime.strptime(dt,'%Y-%m-%d')))
    df = df.sort_values(by='Date', ascending=True)
    return df


def subsequent_trading_date(date):
    tdays = tradingcalendar.trading_days
    last_date = pd.to_datetime(date)
    last_dt = tradingcalendar.canonicalize_datetime(last_date)
    next_dt = tdays[tdays.searchsorted(last_dt) + 1]
    return next_dt

def add_last_bar(df):
    last_date = df.index[-1]
    subsequent_date = subsequent_trading_date(last_date)
    blank_row = pd.Series({}, index=df.columns, name=subsequent_date)
    # add today, and shift all previous data up to today. This 
    # should result in the same data frames as in backtest
    df = df.append(blank_row).shift(1).dropna(how='all')
    return df

def shift_data(df):
    df = add_last_bar(df)
    df.fillna(method='ffill') 
    df['PrevCloses'] = my_rolling_apply_series(df['Close'], to_csv_str, History)
    dates = pd.Series(df.index)
    dates.index = df.index
    df['PrevDates'] = my_rolling_apply_series(dates, to_csv_str, History)
    return df

def unpack_from_data(context, data, sym):
    try:
        v = data.current(sym, 'PrevCloses')
        i = data.current(sym, 'PrevDates')
        return from_csv_strs(i,v,True).apply(float)
    except:
        log.warn("Unable to unpack historical {s} data.".format(s=sym))
        context.fetch_failed = True

def addFieldsVIX(df):
    df = fix_close(df,'VIX Close')
    df['200ma'] = df['Close'].rolling(200).mean()
    df['200ma Ratio'] = df['Close'] / df['200ma']

    return df

def addFieldsVXV(df):
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df = fix_close(df,'CLOSE')
    return df

def addFieldsVX1(df):
    df = fix_closeVX(df,'F1')
    return df

def addFieldsVX2(df):
    df = fix_closeVX(df,'F2')
    return df

# convert a series of values to a comma-separated string of said values
def to_csv_str(s):
    return functools.reduce(lambda x,y: x+','+y, pd.Series(s).apply(str))

# a specific instance of rolling apply, for Series of any type (not just numeric,
# ala pandas.rolling_apply), where the index of the series is set to the indices
# of the last elements of each subset
def my_rolling_apply_series(s_in, f, n):
    s_out = pd.Series([f(s_in[i:i+n]) for i in range(0,len(s_in)-(n-1))]) 
    s_out.index = s_in.index[n-1:]
    return s_out

# reconstitutes a Series from two csv-encoded strings, one of the index, one of the values
def from_csv_strs(x, y, idx_is_date):
    s = pd.Series(y.split(','),index=x.split(','))
    if (idx_is_date):
        s.index = s.index.map(lambda x: pd.Timestamp(x))
    return s

def pvr(context, data):
    ''' Custom chart and/or logging of profit_vs_risk returns and related information from https://www.quantopian.com/posts/pvr#569784bda73e9bf2b7000180
    '''  
    #import time  
    #from datetime import datetime  
    #from pytz import timezone      # Python will only do once, makes this portable.  
                                   #   Move to top of algo for better efficiency.  
    c = context  # Brevity is the soul of wit -- Shakespeare [for readability]  
    if 'pvr' not in c:

        # For real money, you can modify this to total cash input minus any withdrawals  
        manual_cash = c.portfolio.starting_cash  
        time_zone   = 'US/Central'   # Optionally change to your own time zone for wall clock time

        c.pvr = {  
            'options': {  
                # # # # # # # # # #  Options  # # # # # # # # # #  
                'logging'         : 0,    # Info to logging window with some new maximums  
                'log_summary'     : 126,  # Summary every x days. 252/yr

                'record_pvr'      : 1,    # Profit vs Risk returns (percentage)  
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value  
                'record_cash'     : 1,    # Cash available  
                'record_max_lvrg' : 0,    # Maximum leverage encountered  
                'record_risk_hi'  : 0,    # Highest risk overall  
                'record_shorting' : 0,    # Total value of any shorts  
                'record_max_shrt' : 0,    # Max value of shorting total  
                'record_cash_low' : 0,    # Any new lowest cash level  
                'record_q_return' : 0,    # Quantopian returns (percentage)  
                'record_pnl'      : 1,    # Profit-n-Loss  
                'record_risk'     : 1,    # Risked, max cash spent or shorts beyond longs+cash  
                'record_leverage' : 0,    # End of day leverage (context.account.leverage)  
                # All records are end-of-day or the last data sent to chart during any day.  
                # The way the chart operates, only the last value of the day will be seen.  
                # # # # # # # # #  End options  # # # # # # # # #  
            },  
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent  
            'cagr'       : 0,  
            'max_lvrg'   : 0,  
            'max_shrt'   : 0,  
            'risk_hi'    : 0,  
            'days'       : 0.0,  
            'date_prv'   : '',  
            'date_end'   : get_environment('end').date(),  
            'cash_low'   : manual_cash,  
            'cash'       : manual_cash,  
            'start'      : manual_cash,  
            'tz'         : time_zone,  
            'begin'      : time.time(),  # For run time  
            'run_str'    : '{} to {}  ${}  {} {}'.format(get_environment('start').date(), get_environment('end').date(), int(manual_cash), datetime.now(timezone(time_zone)).strftime("%Y-%m-%d %H:%M"), time_zone)  
        }  
        if c.pvr['options']['record_pvrp']: c.pvr['options']['record_pvr'] = 0 # if pvrp is active, straight pvr is off  
        if get_environment('arena') not in ['backtest', 'live']: c.pvr['options']['log_summary'] = 1 # Every day when real money  
        log.info(c.pvr['run_str'])  
    p = c.pvr ; o = c.pvr['options'] ; pf = c.portfolio ; pnl = pf.portfolio_value - p['start']  
    def _pvr(c):  
        p['cagr'] = ((pf.portfolio_value / p['start']) ** (1 / (p['days'] / 252.))) - 1  
        ptype = 'PvR' if o['record_pvr'] else 'PvRp'  
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (p['pvr'] / p['days']), '%.3f' % p['cagr'], '%.0f' % pf.portfolio_value, '%.0f' % pnl))  
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % pnl, '%.0f' % p['risk_hi'], '%.1f' % p['pvr']))  
        log.info('  QRet {} PvR {} CshLw {} MxLv {} RskHi {} MxShrt {}'.format('%.2f' % (100 * pf.returns), '%.2f' % p['pvr'], '%.0f' % p['cash_low'], '%.2f' % p['max_lvrg'], '%.0f' % p['risk_hi'], '%.0f' % p['max_shrt']))  
    def _minut():  
        dt = get_datetime().astimezone(timezone(p['tz']))  
        return str((dt.hour * 60) + dt.minute - 570).rjust(3)  # (-570 = 9:31a)  
    date = get_datetime().date()  
    if p['date_prv'] != date:  
        p['date_prv'] = date  
        p['days'] += 1.0  
    do_summary = 0  
    if o['log_summary'] and p['days'] % o['log_summary'] == 0 and _minut() == '100':  
        do_summary = 1              # Log summary every x days  
    if do_summary or date == p['date_end']:  
        p['cash'] = pf.cash  
    elif p['cash'] == pf.cash and not o['logging']: return  # for speed

    shorts = sum([z.amount * z.last_sale_price for s, z in pf.positions.items() if z.amount < 0])  
    new_key_hi = 0                  # To trigger logging if on.  
    cash       = pf.cash  
    cash_dip   = int(max(0, p['start'] - cash))  
    risk       = int(max(cash_dip, -shorts))

    if o['record_pvrp'] and cash < 0:   # Let negative cash ding less when portfolio is up.  
        cash_dip = int(max(0, cash_dip * p['start'] / pf.portfolio_value))  
        # Imagine: Start with 10, grows to 1000, goes negative to -10, should not be 200% risk.

    if int(cash) < p['cash_low']:             # New cash low  
        new_key_hi = 1  
        p['cash_low'] = int(cash)             # Lowest cash level hit  
        if o['record_cash_low']: record(CashLow = p['cash_low'])

    if c.account.leverage > p['max_lvrg']:  
        new_key_hi = 1  
        p['max_lvrg'] = c.account.leverage    # Maximum intraday leverage  
        if o['record_max_lvrg']: record(MaxLv   = p['max_lvrg'])

    if shorts < p['max_shrt']:  
        new_key_hi = 1  
        p['max_shrt'] = shorts                # Maximum shorts value  
        if o['record_max_shrt']: record(MxShrt  = p['max_shrt'])

    if risk > p['risk_hi']:  
        new_key_hi = 1  
        p['risk_hi'] = risk                   # Highest risk overall  
        if o['record_risk_hi']:  record(RiskHi  = p['risk_hi'])

    # Profit_vs_Risk returns based on max amount actually invested, long or short  
    if p['risk_hi'] != 0: # Avoid zero-divide  
        p['pvr'] = 100 * pnl / p['risk_hi']  
        ptype = 'PvRp' if o['record_pvrp'] else 'PvR'  
        if o['record_pvr'] or o['record_pvrp']: record(**{ptype: p['pvr']})

    if o['record_shorting']: record(Shorts = shorts)             # Shorts value as a positve  
    if o['record_leverage']: record(Lvrg   = c.account.leverage) # Leverage  
    if o['record_cash']    : record(Cash   = cash)               # Cash  
    if o['record_risk']    : record(Risk   = risk)  # Amount in play, maximum of shorts or cash used  
    if o['record_q_return']: record(QRet   = 100 * pf.returns)  
    if o['record_pnl']     : record(PnL    = pnl)                # Profit|Loss

    if o['logging'] and new_key_hi:  
        log.info('{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(),  
            ' Lv '     + '%.1f' % c.account.leverage,  
            ' MxLv '   + '%.2f' % p['max_lvrg'],  
            ' QRet '   + '%.1f' % (100 * pf.returns),  
            ' PvR '    + '%.1f' % p['pvr'],  
            ' PnL '    + '%.0f' % pnl,  
            ' Cash '   + '%.0f' % cash,  
            ' CshLw '  + '%.0f' % p['cash_low'],  
            ' Shrt '   + '%.0f' % shorts,  
            ' MxShrt ' + '%.0f' % p['max_shrt'],  
            ' Risk '   + '%.0f' % risk,  
            ' RskHi '  + '%.0f' % p['risk_hi']  
        ))  
    if do_summary: _pvr(c)  
    if get_datetime() == get_environment('end'):   # Summary at end of run  
        _pvr(c) ; elapsed = (time.time() - p['begin']) / 60  # minutes  
        log.info( '{}\nRuntime {} hr {} min'.format(p['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))

def handle_data(context, data): 
    pvr(context, data)
    c = context
    if c.ShowMaxLev:
        if c.account.leverage > c.mx_lvrg:  
            c.mx_lvrg = c.account.leverage  
            record(mx_lvrg = c.mx_lvrg)    # Record maximum leverage encountered

    ##
    # Buy XIV
    ##
    if c.ShortVIX and not c.BoughtShortVIX and not c.SellAndWait:
        PriceXIV = data.current(c.XIV, 'price')
        if not c.PriceXIV: c.PriceXIV = PriceXIV
        if PriceXIV < c.PriceXIV:
            c.PriceXIV = PriceXIV
        elif PriceXIV > 1.0025 * c.PriceXIV:
            c.BoughtShortVIX = True
            TarPer(context, data, c.XIV, c.VIX_GrowthLeverage)

    ##
    # Buy UVXY
    ##
    if c.LongVIX and not c.BoughtLongVIX and not c.SellAndWait:
        PriceUVXY = data.current(c.UVXY, 'price')
        if not c.PriceUVXY: c.PriceUVXY = PriceUVXY
        if PriceUVXY < c.PriceUVXY:
            c.PriceUVXY = PriceUVXY
        elif PriceUVXY > 1.005 * c.PriceUVXY:
            c.BoughtLongVIX = True
            TarPer(context, data, c.UVXY, c.VIX_HedgeLeverage)

    ##
    # RogueTrader here and in RogueTrader scheduled function
    ##
    if c.RogueTrader and not c.SellAndWait:
        for position in c.portfolio.positions.itervalues():
            if position.amount == 0:
                if position.asset.symbol in c.stops: del c.stops[position.asset.symbol]
                continue
            elif position.asset.symbol not in c.stops:
                stoploss= c.stoploss if position.amount > 0 else -c.stoploss
                c.stops[position.asset.symbol]=position.last_sale_price*(1-stoploss)
                #log.info(' ! I have added '+str(position.asset.symbol)+' to Stops @ '+str((position.last_sale_price)*(1-stoploss)))
            elif c.stops[position.asset.symbol] < position.last_sale_price*(1- c.stoploss) and position.amount > 0:
                c.stops[position.asset.symbol]=position.last_sale_price*(1- c.stoploss)
                #log.info(' ! I have updated '+str(position.asset.symbol)+'- (Long) to stop @ '+str((position.last_sale_price)*(1- c.stoploss)))
            elif c.stops[position.asset.symbol] > position.last_sale_price and position.amount > 0:
                #sell
                log.info(' ! '+str(position.asset.symbol)+'- (Long) has hit stoploss @ '+str(position.last_sale_price))
                if get_open_orders(position.sid): cancel_open_orders_for(context, data, position.sid)
                c.stoppedout.append(position.asset.symbol)
                TarPer(context, data, position.sid, 0.00)
 
            ##
            # Sell and Wait
            ##
            if (
                    (position.sid is c.XIV and 0 < position.amount)
                    and
                    (10.76 < c.vix < 12) and (position.last_sale_price > 1.013 * c.XIVprice)
            ):
                c.SellAndWait = 1
            elif (
                    (position.sid is c.UVXY and 0 < position.amount)
                    and
                    (position.last_sale_price > 1.40 * c.UVXYprice)
            ):
                c.SellAndWait = 3

        ##
        # Sell and Wait
        ##
        if 0 < c.SellAndWait:
            for stock in c.portfolio.positions:
                cancel_open_orders_for(context, data, stock)
                TarPer(context, data, stock, 0.00)

def CherryPickerClose(context,data):    
    
    c = context
    vxx_prices = data.history(c.VXX, "high", c.wvf_length*2, "1d")
    vxx_lows = data.history(c.VXX, "low", c.wvf_length*2, "1d")
    vxx_highest = vxx_prices.rolling(window = c.wvf_length, center=False).max()
    WVF = ((vxx_highest - vxx_lows)/(vxx_highest)) * 100

    rsi = talib.RSI(vxx_prices, timeperiod=c.rsi_length)
    
    c.SmoothedWVF1 = talib.EMA(WVF, timeperiod=c.ema1) 
    c.SmoothedWVF2 = talib.EMA(WVF, timeperiod=c.ema2)
    
    ## BUY RULES
    #if WVF crosses over smoothwvf1 and wvf < smoothwvf2
    if (
        (WVF[-1] > c.SmoothedWVF1[-1] and WVF[-2] < c.SmoothedWVF1[-2] and WVF[-1] < c.SmoothedWVF2[-1])
        or
        (c.SmoothedWVF1[-2] < c.SmoothedWVF2[-2] and c.SmoothedWVF1[-1] > c.SmoothedWVF2[-1])
        or
        (WVF[-1] > c.SmoothedWVF1[-1] and WVF[-2] < c.SmoothedWVF1[-2] and WVF[-1] > c.SmoothedWVF2[-1] and WVF[-2] < c.SmoothedWVF2[-2])
    ):
        c.sell = False
        for stock in c.portfolio.positions:
            if stock is not c.XIV:
                cancel_open_orders_for(context, data, stock)
                TarPer(context, data, stock, 0.00)
            else:
                cancel_open_orders_for(context, data, stock)
                TarPer(context, data, stock, 1.00)
      
    ## SELL RULES
    if c.portfolio.positions[c.XIV].amount > 0:
        #if rsi crosses over rsi_trigger
        if rsi[-2] < c.rsi_trigger and rsi[-1] > c.rsi_trigger:
            c.sell = True
            
        #if wvf crosses under smoothwvf2: sell
        elif WVF[-2] > c.SmoothedWVF2[-2] and WVF[-1] < c.SmoothedWVF2[-1]:
            c.sell = True

        else:
            c.sell = False