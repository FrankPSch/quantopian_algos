import math
import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.covariance import OAS, EmpiricalCovariance
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as smapi
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV

def getweights(params, cov, signal):
    cons = []
    (m,n) = np.shape(params)
    
    for i in range(0, n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.dot(x.T, params[:, i])})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -np.dot(x.T, params[:, i])})
    
    x0 = [1.] * m
    res = minimize(lambda x: 0., x0,
                   constraints = cons, method='cobyla',options={'maxiter':2000})
    return res.x

class Porfolio(object) :
    
    def __init__(self):
        self.hasPosition = False
        self.stocks = None
        self.positionStocks = None
        self.prices = None
        self.mean = 0
        self.std = 0
        self.posSign = 0
        self.signal = 0
        self.weights = None
        
    def set_stocks(self, stocks):
        self.stocks = stocks
        
    def trade(self, context, data):
        if not self.hasPosition:
            success, W, abspval, prices = self.compute(context, data, self.stocks)
            if success:
                self.openPos(-np.sign(self.signal), prices, W, abspval, data, context)
                     
    def compute(self, context, data, stocks):
        prices = data.history(stocks, "price", 200, "1d")
        prices = prices.dropna(axis=1)       
        returns = prices.pct_change().dropna().values
        returns = returns * 1000.
        cov = OAS().fit(returns).covariance_
        e, v = np.linalg.eig(cov)
        idx = e.argsort()
        comp = v[:, idx[-15:]]
        
        if comp[0, 0] < 0:
            comp *= -1
        
        sources = np.dot(returns, comp)
        betas = np.zeros((np.shape(returns)[1], np.shape(sources)[1]))

        for i in range(0, np.shape(returns)[1]):
            model = LassoCV().fit(sources, returns[:, i])
            betas[i, :] = model.coef_

        W = getweights(betas, cov, np.asarray([1.] * np.shape(returns)[1]))
        self.prices = prices.values[0, :]
        pvalue = np.dot(prices.values, W / self.prices)
        self.mean = np.mean(pvalue)
        self.std = np.std(pvalue)
        self.signal = (pvalue[-1] - self.mean) / self.std
        abspval = np.sum(np.abs(W))

        if abs(self.signal) < .5:
            return False, None, None, None
        return True, W, abspval, prices
        
    def openPos(self, side, prices, W, abspval, data, context):
        self.positionStocks = []
        for i, sid in enumerate(prices):
            order_target_value(sid, W[i] / abspval * context.portfolio.portfolio_value * side)
            self.positionStocks.append(sid)
        self.days = 0        
        self.posSign = side
        self.weights = W
        self.hasPosition = True

    def monitor(self, context, data):
        if not self.hasPosition:
            return
        self.days += 1
        prices = data.history(self.positionStocks, "price", 60, "1d")
        prices = prices.dropna(axis=1)       
        if len(prices.columns) <> len(self.positionStocks):
            self.closeAll(context, data, 0)
            return
        pvalue = np.dot(prices.values, self.weights / self.prices) 
        signal = (pvalue[-1] - self.mean) / self.std
        
        if self.posSign > 0 and (signal > 0 or signal < -4):
            self.closeAll(context, data, signal)
        elif self.posSign < 0 and (signal < 0 or signal > 4):
            self.closeAll(context, data, signal)
        elif self.days > 20:
            self.closeAll(context, data, signal)
        
    def closeAll(self, context, data, signal):
        self.positionStocks = None
        self.mean = 0
        self.std = 0
        self.posSign = 0
        self.weights = None
        self.hasPosition = False
        self.prices = None
        log.info(signal)
        for sid in context.portfolio.positions:
            order_target(sid, 0)
        
def initialize(context):
    
    context.portfolios = [Porfolio(), Porfolio(), Porfolio()]

    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=12))
    monitor1 = lambda context, data: context.portfolios[0].monitor(context, data)
    schedule_function(monitor1, date_rules.every_day(), time_rules.market_close(minutes=60))      
     
def before_trading_start(context, data):
    fundamental_df = get_fundamentals(query(fundamentals.valuation.market_cap)
        .filter(fundamentals.asset_classification.morningstar_sector_code == 309)
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") 
        .filter(fundamentals.share_class_reference.security_type == 'ST00000001') 
        .filter(~fundamentals.share_class_reference.symbol.contains('_WI')) 
        .filter(fundamentals.share_class_reference.is_primary_share == True) 
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) 
        .order_by(fundamentals.valuation.market_cap.desc())).T
    context.portfolios[0].set_stocks( fundamental_df[0:50].index )
    context.portfolios[1].set_stocks( fundamental_df[25:50].index )
    context.portfolios[2].set_stocks( fundamental_df[50:75].index )

def handle_data(context,data):
    pass

def trade(context, data):
    for p in context.portfolios:
        p.trade(context, data)
