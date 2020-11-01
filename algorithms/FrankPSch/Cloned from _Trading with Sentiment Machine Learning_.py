from datetime import timedelta
from pytz import timezone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
import talib
import statsmodels.api as sm
import numpy 
import pandas as pd


def preview(df):
    log.info(df.head())
    return df

def custom_split(string_list):
    """
        Parses a string and returns it in list format, without the '[' ']' 

        :params string_list: a list that's been made into string e.g. "[ hello, hello2]"
        :returns: a string that's been made into a list e.g. "[hello, hello2]" => [hello, hello2]
    """
    # Remove the '[' and ']'
    string_list = string_list[1:-1].split(',')
    # Convert to float
    string_list = [float(s) for s in string_list]
    return string_list

def get_day_delta(current_date):
    """
        Takes in the current date, checks it's day of week, and returns an appropriate date_delta
        E.g. if it's a Monday, the previous date should be Friday, not Sunday

        :params current_date: Pandas TimeStamp
        :returns: an int 
    """
    if current_date.isoweekday() == 1:
        return 3
    else:
        return 1

def fill_func(df, row, num_dates):
    """
        Should be applied to every row of a dataframe. Reaches for the past thirty days of each dataframe,
        appends the data to a string, returns the string which should be unpacked later on.

        :params df: The dataframe in it's totality
        :params row: The row of each dataframe, passed through the lambda function of Dataframe.apply(lambda row: row)
        :params num_dates: How many dates to go back (e.g. 30 = 30 days of past data)

        :returns: A list in the form of a string (containing past data) which should be unpacked later on
    """
    # Instantiate variables
    past_data = []
    # The current date is the name of the Series (row) being passed in 
    current_date = row.name
    # print ("current_date ", current_date)
    # Iterate through the number of dates from 0->num_dates
    for i in range(num_dates):
        # How many days to get back, calls get_day_delta for accurate delta assessment
        day_delta = get_day_delta(current_date)
        # print ("day delta ", day_delta)
        # Get the current_date and update the current_date to minus day_delta from the date
        # To get the appropriate past date
        current_date = current_date - timedelta(days=day_delta)
        #print ("changed current_date ", current_date)
        try:
            #: Get the price at the given current_date found by get_day_delta
            data = df.ix[current_date]['sentiment']
            # print ("current date ", current_date, "data " ,data)
            past_data.append(data)
            
            data = df.ix[current_date]['sentiment high']- df.ix[current_date]['sentiment low']
            past_data.append(data)
            
            data = df.ix[current_date]['news volume']
            past_data.append(data)
            
            data = df.ix[current_date]['news buzz']
            past_data.append(data)
            
            #print ("past data " ,past_data)
        except KeyError:
            #: No data for this date, pass
            pass
    # print str(past_data)
    
    # Return the a list made into a string
    return str(past_data)

def post_func(df): 
    df = pd.DataFrame(df)
    df['past_data'] = df.apply(lambda row: fill_func(df, row, 99), axis=1)
    #log.info(df.head())
    #log.info(df.ix[0:11,6])
   
    return df

def initialize(context):
    set_symbol_lookup_date('2017-08-09')
    ## Initialize list of securities we want to trade
    context.security_list = symbols('AAPL', 'BA', 'MRK', 'INTC', 'GOOG')
    ## Trailing stop loss
    context.stop_loss_pct = .995
    # We will weight each asset equally and leave a 5% cash
    # reserve. - actually this is sort of good idea
    context.weight = 0.95 / len(context.security_list)
    
    context.investment_size = (context.portfolio.cash*context.weight)
    
    fetch_csv("https://gist.githubusercontent.com/YUHefei/77133cc5437d7893dd15ab5bc56c4c2f/raw/ef2387158514783b4accd3d01093ac3717055bfb/Finsents_Apple", date_column = 'date', date_format = '%y-%m-%d', pre_func = preview, post_func = post_func)
    
    fetch_csv("https://gist.githubusercontent.com/YUHefei/564436cb1988478e075250208cb2c397/raw/94f8670af124c0b0c235aa35bf7cf31080e34878/Google", date_column = 'date', date_format = '%y-%m-%d', pre_func = preview, post_func = post_func)
    
    fetch_csv("https://gist.githubusercontent.com/YUHefei/3374b81b9235a15ad9d16f430c499ae7/raw/2e42ad76723d28163188e472c91e38a2f7696883/Boeing", date_column = 'date', date_format = '%y-%m-%d', pre_func = preview, post_func = post_func)
    
    fetch_csv("https://gist.githubusercontent.com/YUHefei/190e7dbb2fc08e0ef23a9a6c2ad5f9f5/raw/10d9defa1935dab4817167bb140666c35e7cd0ab/Intel", date_column = 'date', date_format = '%y-%m-%d', pre_func = preview, post_func = post_func)
        
    fetch_csv("https://gist.githubusercontent.com/YUHefei/846e8869b3da1c1a23ffa922ca1f4dc0/raw/c7dd8d7d7898fb757838edda0111aef95f2903f0/Merck", date_column = 'date', date_format = '%y-%m-%d', pre_func = preview, post_func = post_func)
                               

    context.historical_bars = 100
    context.feature_window = 3
    
    schedule_function(myfunc, date_rules.every_day(), 
        time_rules.market_open(hours=0, minutes=1))
       
def myfunc(context, data):
        price_history = data.history(context.security_list, fields="price", bar_count=100, frequency="1d")
        
        try: 
            # For loop for each stock traded everyday:
            for s in context.security_list:
                
                start_bar = context.feature_window
                price_list = price_history[s].tolist()
                past = data.current(s,'past_data')
                pastlist=custom_split(past)
                #print isinstance(past, str)
                #print isinstance(custom_split(past), list)
                 
                print pastlist 
                print len(past)
                print len(pastlist)
                print len(price_list)
                #print past[1:-1]
                
                X = []
                y = []
        
                bar= start_bar
                
                # Loop for each machine learning data set
                while bar < len(price_list)-1:
   
                # print s," price: ",data.history(s, 'price', 100 , "1d")
                    try: 
                        end_price = price_list[bar]
                        start_price = price_list[bar-1]
                
                        features = pastlist[(bar-3)*4: bar*4]
                        # Featuers are the attribute value used for machine learning.
                        #print(features)
                
                        if end_price > start_price:
                            label = 1
                        else:
                            label = -1
                        # Label is the indicator of whether this stock will rise or fall
                        bar +=1 
                
                        X.append(features)
                        y.append(label)
                    
                        #print X 
                        #print y
             
                    except Exception as e:
                
                        bar +=1
                        print(('feature creation', str(e)))
                
                print ('len(X1)',len(X))
                
                # Call the machined learning model
                clf1 = RandomForestClassifier(n_estimators=100)
                clf2 = LinearSVC()
                clf3 = NuSVC()
                clf4 = LogisticRegression()
                
                # Rrepare the attribute information for prediction
                current_features=pastlist[384:396]
                
                X.append(current_features)
                print ('len(X2)',len(X))
                
                # Rescall all the data
                X = preprocessing.scale(X)
        
                current_features = X[-1:]
                X = X[:-1]
                
                print current_features
                print ('len(X)',len(X))
                print ('len(y)',len(y))
                
                # Build the model
                clf1.fit(X,y)
                clf2.fit(X,y)
                clf3.fit(X,y)
                clf4.fit(X,y)
        
                # Predict the results 
                p1 = clf1.predict(current_features)[0]
                p2 = clf2.predict(current_features)[0]
                p3 = clf3.predict(current_features)[0]
                p4 = clf4.predict(current_features)[0]
         
                # If 3 out of 4 prediction votes for one same results, this results will be promted to be the one I will use. 
                if Counter([p1,p2,p3,p4]).most_common(1)[0][1] >= 3:
                    p = Counter([p1,p2,p3,p4]).most_common(1)[0][0]
            
                else: 
                    p = 0
            
                print(('Prediction',p))         
                
                current_price = data.current(s, 'price')
                current_position = context.portfolio.positions[s].amount
                cash = context.portfolio.cash
                
                # Add one more feature: moving average
                print('price_list', price_list)
                sma_50 = numpy.mean(price_list[-50:])
                sma_20 = numpy.mean(price_list[-20:])
                print('sma_20', sma_20)
                print('sma_50', sma_50)
                
                open_orders = get_open_orders()
                
                # Everyday's trading activities: 
                if (p == 1) or (sma_20 > sma_50):
                    if s not in open_orders:
                        order_target_percent(s, context.weight, style=StopOrder(context.stop_loss_pct*current_price))
                        cash-=context.investment_size
                elif (p == -1) or (sma_50 > sma_20):
                    if s not in open_orders:
                        order_target_percent(s,-context.weight)
       
        except Exception as e:
            print(str(e))    
    
def handle_data(context, data):
    #Plot variables at the end of each day.
    
    long_count = 0
    short_count = 0

    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        if position.amount < 0:
            short_count += 1
            
    record(num_long=long_count, num_short=short_count, leverage=context.account.leverage)
