import numpy as np

#globals for get_avg batch transform decorator
R_P = 1 #refresh period in days
W_L = 5 #window length in days

def initialize(context):
    #['AMD', 'CERN', 'COST', 'DELL', 'GPS', 'INTC', 'MMM']
    context.stocks = [sid(351), sid(1419), sid(1787), sid(25317), sid(3321), sid(3951), sid(4922)]
    
    context.m = len(context.stocks)
    context.b_t = np.ones(context.m) / context.m
    context.eps = 1  #change epsilon here
    context.init = False
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
def handle_data(context, data):    
    
    # get prices
    prices = get_prices(data,context.stocks)
    if prices == None:
       return
    
    if not context.init:
        rebalance_portfolio(context, data, context.b_t)
        context.init = True
        return

    m = context.m

    x_tilde = np.zeros(m)

    b = np.zeros(m)

    # find relative moving average price for each security
    for i, stock in enumerate(context.stocks):
        x_tilde[i] = np.mean(prices[:,i])/prices[W_L-1,i]
    
    ###########################
    # Inside of OLMAR (algo 2)

    x_bar = x_tilde.mean()

    # Calculate terms for lambda (lam)
    dot_prod = np.dot(context.b_t, x_tilde)
    num = context.eps - dot_prod
    denom = (np.linalg.norm((x_tilde-x_bar)))**2

    # test for divide-by-zero case
    if denom == 0.0:
        lam = 0 # no portolio update
    else:     
        lam = max(0, num/denom)
    
    b = context.b_t + lam*(x_tilde-x_bar)

    b_norm = simplex_projection(b)

    rebalance_portfolio(context, data, b_norm)

    # update portfolio
    context.b_t = b_norm
    
    log.debug(b_norm)

    log.debug(np.sum(b_norm))
    
@batch_transform(refresh_period=R_P, window_length=W_L) # set globals R_P & W_L above

def get_prices(datapanel,sids):
    return datapanel['price'].as_matrix(sids)

def rebalance_portfolio(context, data, desired_port):
    #rebalance portfolio
    current_amount = np.zeros_like(desired_port)
    desired_amount = np.zeros_like(desired_port)
    
    if not context.init:
        positions_value = context.portfolio.starting_cash
    else:
        positions_value = context.portfolio.positions_value + context.portfolio.cash  
    
    for i, stock in enumerate(context.stocks):
        current_amount[i] = context.portfolio.positions[stock].amount
        desired_amount[i] = desired_port[i]*positions_value/data[stock].price

    diff_amount = desired_amount - current_amount

    for i, stock in enumerate(context.stocks):
        order(stock, diff_amount[i]) #order_stock

def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

Implemented according to the paper: Efficient projections onto the
l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
Optimization Problem: min_{w}\| w - v \|_{2}^{2}
s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
Output: Projection vector w

:Example:
>>> proj = simplex_projection([.4 ,.3, -.4, .5])
>>> print proj
array([ 0.33333333, 0.23333333, 0. , 0.43333333])
>>> print proj.sum()
1.0

Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
"""

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w