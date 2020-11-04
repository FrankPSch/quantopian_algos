# Use the previous 10 bars' movements to predict the next movement.

# Use elements of http://scikit-learn.org/stable/user_guide.html
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
#from statistics import mean
import numpy as np
import talib
import math


def initialize(context):
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    #set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

#    context.security_list = [
#                             sid(5061),  # MSFT - Tech
#                             sid(23103)  # ANTM - Healthcare
#                            ]
    context.security_list = [sid(8554)]   # SPY for best comparison

    context.lookback = 5 # Look back # days
    context.history_range = 600 # Consider the past # days' history
    context.ma_fast = 3
    context.ma_slow = 18

    context.price_change_bins=np.linspace(start=-5, stop=5, num=21)
    context.vol_change_bins=np.linspace(start=-1e9, stop=1e9, num=50)
    context.vol_abs_bins=np.logspace(start=6, stop=12, num=30)
    print("price_change_bins", context.price_change_bins)
    print("vol_change_bins", context.vol_change_bins)
    print("vol_abs_bins", context.vol_abs_bins)

    # Initialize scaler

    # http://fatihsarigoz.com/scaling-ml.html
    # impact of scaling on bagging algorithm 'RandomForestRegressor' and boosting algo 'GradientBoostingRegressor'
    # is negligible since decision-tree based

    #MinMaxScaler standardizes each feature (column) individually!!
    #context.scaler_price_change = MinMaxScaler(feature_range=(-1, 1))
    #context.scaler_vol_change = MinMaxScaler(feature_range=(-1, 1))
    #context.scaler_vol_abs = MinMaxScaler(feature_range=(0, 1))

    #PowerTransformer applies a non-linear transformation for zero-mean and unit variance normalization
    #context.scaler_price_change = PowerTransformer(method='yeo-johnson')
    #context.scaler_vol_change = PowerTransformer(method='yeo-johnson')
    #context.scaler_vol_abs = PowerTransformer(method='yeo-johnson')

    #context.price_change_scaler  = {}
    #context.vol_abs_scaler = {}
    #context.vol_change_scaler = {}

    context.prediction = np.zeros_like(context.security_list)

    # Initialize models
    context.rfr_models = {}
    context.rfr_me = {}
    context.rfr_fit = None

    context.gbr_models = {}
    context.gbr_me = {}
    context.gbr_r2 = {}
    context.gbr_fit = None

    context.gbr_models_slow = {}
    context.gbr_me_slow = {}
    context.gbr_fit_slow = None

    # Generate new models every week
    schedule_function(create_model, date_rules.week_end(), time_rules.market_close(minutes=10))

    # Trade 1 minute after the start of every day
    schedule_function(trade_open, date_rules.every_day(), time_rules.market_open(minutes=1))
    
    # Recorde closing price of current day after closing
    schedule_function(record_close, date_rules.every_day(), time_rules.market_close(minutes=10))


def create_model(context, data):
    for idx, security in enumerate(context.security_list):

        # Is called once per week after closing
        # Get the relevant daily prices and volumes
        recent_prices = data.history(context.security_list[idx], 'price', context.history_range, '1d').values # ndarray 0 - 149
        recent_p_open = data.history(context.security_list[idx], 'open', context.history_range, '1d').values  # ndarray 0 - 149
        recent_vols = data.history(context.security_list[idx], 'volume', context.history_range, '1d').values  # ndarray 0 - 149

        # Get price and volume change. The last change is yesterday's close to current price=todays close.
        price_change = np.diff(recent_prices).tolist() # list 149
        vol_change = np.diff(recent_vols).tolist()     # list 149
        vol_abs = recent_vols.tolist()[1:]             # list 149

        # Get scalers
        #context.price_change_scaler[idx] = context.scaler_price_change.fit(price_change)
        #context.vol_change_scaler[idx] = context.scaler_vol_change.fit(vol_change)
        #context.vol_abs_scaler[idx] = context.scaler_vol_abs.fit(vol_abs)

        # Apply scaling
        #X_price_change = context.price_change_scaler[idx].transform(price_change).tolist()
        #X_vol_change = context.vol_change_scaler[idx].transform(vol_change).tolist()
        #X_vol_abs = context.vol_abs_scaler[idx].transform(vol_abs).tolist()
        
        X_price_change = np.digitize(price_change, context.price_change_bins)
        X_vol_change = np.digitize(vol_change, context.vol_change_bins)
        X_vol_abs = np.digitize(vol_abs, context.vol_abs_bins)
        
        # Calc HLCC
        prices = data.history(security, ['high','low', 'close'], context.ma_slow+context.history_range, '1d')  
        H = prices['high']  
        L = prices['low']  
        C = prices['close'] 
        HLCC_prices = (H + L + 2*C) / 4

        # Create input and output for ML
        X = [] # Independent, or input variables
        Y = [] # Dependent, or output variable

        # For each day in our history range except lookback period
        # -1 as list starts with a 0
        # -1 as last day is used for today
        for i in range(context.history_range-context.lookback-1-1-1):
            yesterday = context.lookback + i
            today = yesterday + 1
            # Append prior price and volume change within lookback period
            # np.diff is a[n+1] - a[n]
            p_open_change = np.diff(np.array([recent_prices[yesterday], recent_p_open[today]])).tolist()
            #X_p_open_change = context.price_change_scaler[idx].transform(p_open_change)
            X_p_open_change = p_open_change

            Xi = np.append(X_price_change[i:yesterday], X_p_open_change) # ndarray 6 = 0-5
            Xi = np.append(Xi, X_vol_change[i:yesterday])                # ndarray 6 + 5 = 0-10
            Xi = np.append(Xi, X_vol_abs[i:yesterday])                   # ndarray 6 + 5 + 5 = 0-15

            mavg_fast = HLCC_prices[context.ma_slow-context.ma_fast+yesterday:context.ma_slow+yesterday-1].mean()
            mavg_slow = HLCC_prices[yesterday:context.ma_slow+yesterday-1].mean()
            Xi = np.append(Xi, mavg_fast/mavg_slow)                      # ndarray 6 + 5 + 5 + 1 = 0-16

            X.append(Xi)                                                 # list 1 with ndarray 0-15

            # Append the price change of today
            Y.append(price_change[today])                                # list 1
            #Y.append(price_change[today]+price_change[today+1])          # list 1

        # Generate our models
        rfr = RandomForestRegressor()
        # https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
        gbr = GradientBoostingRegressor(learning_rate = 0.01, n_estimators = 100, max_depth = 4, min_samples_split = 0.9, min_samples_leaf=0.3, max_features=8)
        gbr_slow = GradientBoostingRegressor(learning_rate = 0.1, n_estimators = 50)

        # Test our models on independent test data
        offset = int(len(X) * 0.8)
        X_train, Y_train = X[:offset], Y[:offset]
        X_test, Y_test = X[offset:], Y[offset:]
        
        rfr.fit(X_train, Y_train)
        rfr_me = math.sqrt(mean_squared_error(Y_test, rfr.predict(X_test)))
        context.rfr_me[idx] = rfr_me
        
        gbr.fit(X_train, Y_train)
        gbr_me = math.sqrt(mean_squared_error(Y_test, gbr.predict(X_test)))
        gbr_r2 = r2_score(Y_test, gbr.predict(X_test))
        context.gbr_me[idx] = gbr_me
        context.gbr_r2[idx] = gbr_r2

        gbr_slow.fit(X_train, Y_train)
        gbr_me_slow = math.sqrt(mean_squared_error(Y_test, gbr_slow.predict(X_test)))
        context.gbr_me_slow[idx] = gbr_me_slow

        # Fit our models with all data
        rfr.fit(X, Y)
        gbr.fit(X, Y)
        gbr_slow.fit(X, Y)

        # Store our models
        context.rfr_models[idx] = rfr
        context.gbr_models[idx] = gbr
        context.gbr_models[idx] = gbr_slow


def trade_open(context, data):
    if context.rfr_models and context.gbr_models: # Check if our model is generated
        for idx, security in enumerate(context.security_list):

            i = 0
            yesterday = i
            today = yesterday + 1

            # Get recent prices and volumes within lookback period
            # (+1day to calc np.diff, price:+1day for todays open)
            recent_prices = data.history(security, 'price', context.lookback+2, '1d').values  # ndarray 0-6
            recent_vols = data.history(security, 'volume', context.lookback+1, '1d').values   # ndarray 0-5

            # Get price and volume change. The last change is yesterday's close to current price.
            price_change = np.diff(recent_prices).tolist() # list 6
            vol_abs = recent_vols.tolist()[1:]             # list 5
            vol_change = np.diff(recent_vols).tolist()     # list 5

            # Apply scaling. X_price_change includes closing prices of last days plus open price of today.
            #X_price_change = context.price_change_scaler[idx].transform(price_change).tolist() # list 6
            #X_vol_change = context.vol_change_scaler[idx].transform(vol_change).tolist()       # list 5
            #X_vol_abs = context.vol_abs_scaler[idx].transform(vol_abs).tolist()                # list 5
            X_price_change = np.digitize(price_change, context.price_change_bins)
            X_vol_change = np.digitize(vol_change, context.vol_change_bins)
            X_vol_abs = np.digitize(vol_abs, context.vol_abs_bins)

            # Calc HLCC prices
            prices = data.history(security, ['high','low', 'close'], context.ma_slow+1, '1d')
            H = prices['high']
            L = prices['low']
            C = prices['close']
            HLCC_prices = (H + L + 2*C) / 4
            #mavg_fast = HLCC_prices[context.ma_slow-context.ma_fast:context.ma_slow-1].mean()
            #mavg_slow = HLCC_prices[:context.ma_slow-1].mean()
            mavg_fast = HLCC_prices[context.ma_slow-context.ma_fast+yesterday:context.ma_slow+yesterday-1].mean()
            mavg_slow = HLCC_prices[yesterday:context.ma_slow+yesterday-1].mean()

            # Predict using our model and the recent prices
            # 01-01-2007 to 12-31-2017
            # GBR: S 0.39, DD -35%
            # RFR: S 0.23, DD -52%
            # 0.7 GBR + 0.3 RFR: S 0.30, DD -45%
            # 0.67 GBR + 0.33 RFR: S 0.45, DD -47% with other gbr learning rate and 150 days of history
            # 0.7 GBR + 0.3 RFR: S 0.44, DD -49% with other gbr learning rate and 400 days of history
            #Xi = np.append(X_price_change[i:yesterday], X_p_open_change) # ndarray 6 = 0-5
            #Xi = np.append(Xi, X_vol_change[i:yesterday])                # ndarray 6 + 5 = 0-10
            #Xi = np.append(Xi, X_vol_abs[i:yesterday])                   # ndarray 6 + 5 + 5 = 0-15
            Xi = X_price_change                                # list 6
            Xi = np.append(Xi, X_vol_change)                   # ndarray 0-10
            Xi = np.append(Xi, X_vol_abs)                      # ndarray 0-15
            Xi = np.append(Xi, mavg_fast/mavg_slow)            # ndarray 6 + 5 + 5 + 1 = 0-16

            pred_price_change = 0.7*(context.gbr_models[idx].predict(Xi)) + 0.3*(context.rfr_models[idx].predict(Xi))  # ndarray 0
            
            record(pred_price_change = pred_price_change)
            #record(X_price_change = np.array(X_price_change).mean())
            #record(X_vol_change = np.array(X_vol_change).mean())
            #record(mavg_fastslowdelta = mavg_fast-mavg_slow)
            #record(mavg_fast_slow = 2*(mavg_fast/mavg_slow-1))
            #record(mean_error_rfr = context.rfr_me[idx])
            #record(mean_error_gbr = context.gbr_me[idx])
            #record(mean_error_gbr_rfr = context.gbr_me[idx]/context.rfr_me[idx])
            #record(mean_error_gbr_gbrslow = context.gbr_me[idx]/context.gbr_me_slow[idx])
            #record(r2_score_gbr = context.gbr_r2[idx])

            # Distribute securities evenly accross securities
            weight = 1.0 / len(context.security_list)   

            # Go long if we predict the price will rise, short otherwise
            minpred = 1.0
            if pred_price_change > minpred:
                order_target_percent(security, +weight)
            elif pred_price_change < -minpred:
                order_target_percent(security, -weight/2) # short only with half
            else:
                order_target_percent(security, 0)


def record_close(context, data):
    if context.rfr_models and context.gbr_models: # Check if our model is generated
        for idx, security in enumerate(context.security_list):
            # Get close price of yesterday and close proce of today
            # Price is forward-filled, returning last known price, if there is one
            recent_prices = data.history(security, 'price', 2, '1d').values     # ndarray 0-1
            # Get price change
            price_change = np.diff(recent_prices).tolist()                      # list 1
            record(price_change = price_change[-1])

            breakpoint=1