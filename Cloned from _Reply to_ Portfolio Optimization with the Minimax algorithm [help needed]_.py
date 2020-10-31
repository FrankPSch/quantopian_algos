import cvxpy as cvx  
import pandas as pd
import numpy as np
import quantopian.algorithm as algo
import quantopian.optimize as qopt

def get_minimax_weights(returns, target_loss=0.03):
    num_stocks = len(returns.columns)
    mean = returns.mean(axis=0).values
    A = returns.as_matrix()
    x = cvx.Variable(num_stocks)
    objective = cvx.Maximize(mean.T * x)

    constraints = [
        x >= 0,
        A * x >= -target_loss,
        cvx.sum_entries(x) == 1.0
    ]  
    prob = cvx.Problem(objective, constraints)  
    prob.solve(verbose=True)  
    print(prob.value)
    print(x.value)
    w = pd.Series(data=np.asarray(x.value).flatten(), index=returns.columns)
    w = w / w.abs().sum()
    return w

def get_RCK_weights(returns, minimalWealthFraction=0.7, confidence=0.3,max_expo=0.25):
    n = len(returns.columns)
    pi = np.array([1. / len(returns)] * len(returns))
    r = (returns+1.).as_matrix().T
    b_rck = cvx.Variable(n)
    lambda_rck = cvx.Parameter(sign='positive')
    lambda_rck.value = np.log(confidence) / np.log(minimalWealthFraction)
    growth_rate = pi.T * cvx.log(r.T * b_rck)
    risk_constraint = cvx.log_sum_exp(np.log(pi) - lambda_rck * cvx.log(r.T * b_rck)) <= 0
    constraints = [cvx.sum_entries(b_rck) == 1, b_rck >= 0, b_rck<=max_expo, risk_constraint] 
    rck = cvx.Problem(cvx.Maximize(growth_rate), constraints)
    rck.solve(verbose=False)
    #print rck.value
    #print b_rck.value
    w = pd.Series(data=np.asarray(b_rck.value).flatten(), index=returns.columns)
    w = w / w.abs().sum()
    return w


def initialize(context):
    context.securities=  [
        sid(25904), # VFH (Vanguard Financials ETF)
        sid(25906), # VHT (Vanguard Health Care ETF)
        sid(25905), # VGT (Vanguard Information Technology ETF)
        sid(26667), # VDE (Vanguard Energy ETF)
        sid(25902), # VCR (Vanguard Consumer Discretionary ETF)
        sid(22445), # IBB (iShares Nasdaq Biotechnology Index Fund)
        sid(39479), # IBB (iShares Nasdaq Biotechnology Index Fund)
        sid(22887), # EDV VANGUARD treasury
        sid(25899), # VB = Vanguard small cap
        sid(25898)  # VAW (Vanguard Materials ETF) 
    ]
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1))

def my_rebalance(context,data):
    returns = data.history(context.securities, 'price', 400, '1d').pct_change().dropna()
    try:
        w=get_RCK_weights(returns)
        order_ptf(w)
    except:
        pass

def order_ptf(weights):
    algo.order_optimal_portfolio(
        objective=qopt.TargetWeights(weights),
        constraints=[
            qopt.MaxGrossExposure(1.0),
        ]
    )