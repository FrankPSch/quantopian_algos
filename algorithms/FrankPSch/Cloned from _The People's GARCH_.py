import cvxopt
from functools import partial
import math
import numpy as np
import scipy
from scipy import stats
import statsmodels as sm
from statsmodels.stats.stattools import jarque_bera

 
def initialize(context):
    #Some stocks that we will try to simulate the GARCH function on
    stock_ids = [symbol('AAPL'), symbol('AMZN'), symbol('ATVI'), symbol('AAMRQ'), symbol('CSCO'), symbol('DISH'), symbol('INTC'), symbol('LBTY_B'), symbol('MAT'), symbol('MSFT'), symbol('PYPL'), symbol('TXN'), symbol('VIAB'), symbol('SYMC')]
    
    #Simulating that we are trading on robinhood :)
    set_commission(commission.PerShare(cost=0.00, min_trade_cost=0.00))
    
    #Safer instruments to fall back on.
    context.gold = symbol('GLD')
    context.bond = symbol('QQQ')
    context.oil = symbol('OIL')
    
    # Setting constraints that will be used every time we run the maxmimum likelihood function
    context.cons = ({'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2},
                    {'type': 'ineq', 'fun': constraint3})
    
    # Guess values for alpha, beta and gamma will be put in the theta array
    context.initial_theta = [1, 0.5, 0.5]
    
    # An array to keep track of the stocks we are interested in
    context.avaliable_stocks = []
    
    #Creating a "Stock" class for each of the ids and putting into array above
    for i in range(len(stock_ids)):
        temp_s = Stock(stock_ids[i])
        context.avaliable_stocks.append(temp_s)
    
    #Variable used to keep track of the number of stock investments
    context.number_of_stocks = 0
    
    #We will simulate the GARCH assessment every week.
    schedule_function(weekly_assessment, date_rules.week_start(), time_rules.market_open(hours=0,minutes=5))

def weekly_assessment(context,data):
    # We will loop through our array of stocks and give the stock a score (predicted future return)
    for s in context.avaliable_stocks:
        s.score = update_garch(s.ident,context,data)        
    
    # We will sort an array based on future returns (largest first)
    t = context.avaliable_stocks
    temp_array  = sorted(t, key = by_score, reverse=True)
    context.number_of_stocks = 0
    
    for s in temp_array:
        print(s.score)
        if not data.can_trade(s.ident): 
            continue
            
        if s.score>0 and context.number_of_stocks<3:
            order_target_percent(s.ident,0.3)
            s.low_price = data.current(s.ident,"price")*0.98 #We will exit if the stock goes below 2% of buy price
            context.number_of_stocks+=1
        else:
            order_target_percent(s.ident,0)
      
    # Order a bond as a safety
    order_target_percent(context.bond, 0.1+(3-context.number_of_stocks)*0.3)

def handle_data(context,data):
    #record leverage
    lev = context.account.leverage
    record(l=lev)
                
def update_garch(equity, context, data):
    #Set input variables
    X=np.array(data.history(equity,'price',1000,'1d')[:-1])
    X=np.diff(np.log(X)) #we will use the log returns
        
    # Make our objective function by plugging X into our log likelihood function
    objective = partial(negative_log_likelihood, X)

    # Minimize (actually maximize) the likelihood function
    result = scipy.optimize.minimize(objective, context.initial_theta, method='COBYLA', constraints = context.cons)
    
    # Theta_mle will hold our estimated GARCH parameters from the MLE
    theta_mle = result.x

    # We can now calculate our previous sigmas using the GARCH function and our parameters
    sigma_hats = np.sqrt(compute_squared_sigmas(X, np.sqrt(np.mean(X**2)), theta_mle))
    
    # Setting parameters that we will use for simulating the new prices
    a0 = theta_mle[0]
    a1 = theta_mle[1] 
    b1 = theta_mle[2]
    sigma1 = sigma_hats[-1]
    
    # Finding current price
    current_stock_price=data.current(equity,"price")
    
    # Simulate the future price (using GARCH parameters, monte carlo, and the brownian motion of stock prices)
    future_price = mc_simulate(current_stock_price, a0, a1, b1, sigma1)

    return (future_price/current_stock_price-1)

#Using this function to sort stock arrays based on ha-scores    
def by_score(Stock):
      return Stock.score
      
# Define the constraints for our minimizer function
def constraint1(theta):
    return 1 - (theta[1] + theta[2]) #will set 1-a1-b1>0

def constraint2(theta):
    return theta[1] #will set a1>0

def constraint3(theta):
    return theta[2] #will set b1>0

def negative_log_likelihood(X, theta):
    # Estimate initial sigma squared
    initial_sigma = np.sqrt(np.mean(X ** 2))
    
    # Generate the squared sigma values with the GARCH function
    sigma2 = compute_squared_sigmas(X, initial_sigma, theta)
    
    # This is the maximum likelihood function for the probability distribution function
    # of the logaritmic returns. This is what we want to minimize (but actually maximize)
    logL = -((-np.log(sigma2) - X**2/sigma2).sum())

    return logL

# We will use this function to forecast future values of the log return and sigma
def forecast_GARCH(T, a0, a1, b1, sigma1):
    
    # Initialize our values to hold log returns
    X = np.ndarray(T)
    
    #Setting up starting values
    sigma = np.ndarray(T)
    sigma[0] = sigma1
    
    for t in range(1, T):
        # Draw the next return
        X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
   
        # Draw the next sigma_t
        var_temp = a0 + b1 * sigma[t - 1]**2 + a1 * X[t - 1]**2
        sigma[t] = math.sqrt(var_temp)
    
    #Last value
    X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)    
    
    return X, sigma

# This function will help us estimate sigmas with the parameters according to the GARCH function
def compute_squared_sigmas(X, initial_sigma, theta):
    
    #Setting GARCH parameters
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    
    T = len(X)
    sigma2 = np.ndarray(T)
    
    sigma2[0] = initial_sigma ** 2
    
    for t in range(1, T):
        # Here's where we apply the GARCH equation
        sigma2[t] = a0 + a1 * X[t-1]**2 + b1 * sigma2[t-1]
    
    return sigma2

#Monte-carlo simulate a stock movement with the help of GARCH parameters and a brownian motion
def mc_simulate(S0, a0, a1, b1, sigma1):
    price_forecast = []
    
    #Setting up time period for brownian motion
    T = 4
    dt = 1
    N = round(T/dt)
    
    #Standard brownian motion
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt) 
    
    #Start montecarlo simulation
    for i in range(1,100):
        S = S0
        for j in range(0,T-1):
            change_forecast, sigma_forecast = forecast_GARCH(T, a0, a1, b1, sigma1) #forecasting 
            X = (change_forecast[j]-0.5*sigma_forecast[j]**2)*j + sigma_forecast[j]*W 
            S = S*np.exp(X) ### geometric brownian motion ###
        
        price_forecast.append(S)
    
    #returning the average predicted stock price
    return np.mean(price_forecast)

#Keeping track of the stock
class Stock:
   def __init__(self, ident):
       self.ident = ident  
       self.score = 0
       self.low_price = 0