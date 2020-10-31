from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.classifiers.fundamentals import Sector 
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline import CustomFactor  
from quantopian.pipeline.filters import QTradableStocksUS
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd


def initialize(context):

    # Rebalance on the first trading day of each month at 11AM.
    schedule_function(my_rebalance,
                      date_rules.month_start(),
                      time_rules.market_open(hours=1, minutes=30))

    # Create and attach our pipeline (dynamic stock selector), defined below.
    attach_pipeline(make_pipeline(context), 'pit')

    
def make_pipeline(context):
  # Get Latest Fundamentals 
  OCF = Fundamentals.operating_cash_flow.latest
  debt_to_asset = Fundamentals.debtto_assets.latest
  quick_ratio = Fundamentals.quick_ratio.latest
  outstanding_shares = Fundamentals.shares_outstanding.latest
  gross_margin = Fundamentals.gross_margin.latest
  assets_turnover = Fundamentals.assets_turnover.latest
  NI = Fundamentals.net_income_from_continuing_operations.latest
  cash_return = Fundamentals.cash_return.latest
  enterprise_value = Fundamentals.enterprise_value.latest
  total_revenue = Fundamentals.total_revenue.latest
  fcf_yield = Fundamentals.fcf_yield.latest

  #Get Last Year Fundamentals
  class Previous(CustomFactor):  
    # Returns value of input x trading days ago where x is the window_length  
    # Both the inputs and window_length must be specified as there are no defaults
      def compute(self, today, assets, out, inputs):  
          out[:] = inputs[0]
  window_length = 252
  debt_to_asset2 = Previous(inputs = [Fundamentals.debtto_assets], window_length = window_length)
  quick_ratio2 = Previous(inputs = [Fundamentals.quick_ratio], window_length = window_length)
  outstanding_shares2 = Previous(inputs = [Fundamentals.shares_outstanding], window_length = window_length)
  gross_margin2 = Previous(inputs = [Fundamentals.gross_margin], window_length = window_length)
  assets_turnover2 = Previous(inputs = [Fundamentals.assets_turnover], window_length = window_length)
  NI2 = Previous(inputs = [Fundamentals.net_income_from_continuing_operations], window_length = window_length)
  total_revenue2 = Previous(inputs = [Fundamentals.total_revenue], window_length = window_length)
  cash_return2 = Previous(inputs = [Fundamentals.cash_return], window_length = window_length)
  enterprise_value2 = Previous(inputs = [Fundamentals.enterprise_value], window_length = window_length)
  result = Pipeline(
    columns={
        'OCF':OCF,
        'debt_to_asset':debt_to_asset,
        'debt_to_asset2':debt_to_asset2,
        'quick_ratio':quick_ratio,
        'quick_ratio2':quick_ratio2,
        'outstanding_shares':outstanding_shares,
        'outstanding_shares2':outstanding_shares2,    
        'gross_margin':gross_margin,
        'gross_margin2':gross_margin2,
        'assets_turnover':assets_turnover,
        'assets_turnover2':assets_turnover2,
        'NI': NI,
        'NI2': NI2,
        'total_revenue': total_revenue,
        'total_revenue2': total_revenue2,
        'cash_return': cash_return,
        'cash_return2': cash_return2,
        'enterprise_value': enterprise_value,
        'enterprise_value2': enterprise_value2,
        'fcf_yield': fcf_yield
        }, screen = QTradableStocksUS()
  )
  return result


def before_trading_start(context, data):
    
    context.output = pipeline_output('pit')
    result = context.output.dropna(axis=0)
    result2 = context.output.dropna(axis=0)
    
    result.loc[:,('total_avg_assets')] = result.loc[:,('total_revenue')]/result.loc[:,('assets_turnover')]
    result.loc[:,('ROA')] = result.loc[:,('NI')]/result.loc[:,('total_avg_assets')]
    result.loc[:,('FCF')] = result.loc[:,('cash_return')]*result.loc[:,('enterprise_value')]
    result.loc[:,('FCFTA')] = result.loc[:,('FCF')]/result.loc[:,('total_avg_assets')]
    
    result.loc[:,('total_avg_assets2')] = result.loc[:,('total_revenue2')]/result.loc[:,('assets_turnover2')]
    result.loc[:,('ROA2')] = result.loc[:,('NI2')]/result.loc[:,('total_avg_assets2')]
    result.loc[:,('FCF2')] = result.loc[:,('cash_return2')]*result.loc[:,('enterprise_value2')]
    result.loc[:,('FCFTA2')] = result.loc[:,('FCF2')]/result.loc[:,('total_avg_assets2')]
    
#Current Profitability
#ROA > 0
    result.loc[:,('FS_ROA')] = result.loc[:,('ROA')] >0
#FCFTA > 0
    result.loc[:,('FS_FCFTA')] = result.loc[:,('FCFTA')] >0
#Accrual
    result.loc[:,('F_ACCRUAL')] = result.loc[:,('OCF')] > result.loc[:,('NI')]
    
#Stability
#Lever
    result.loc[:,('delta_lever')] = result.loc[:,('debt_to_asset')] - result.loc[:,('debt_to_asset2')] < 0
#Liquidity
    result.loc[:,('delta_quick_ratio')] = result.loc[:,('quick_ratio')] - result.loc[:,('quick_ratio2')] > 0
#Equity Repurchase
    result.loc[:,('delta_OS')] = result.loc[:,('outstanding_shares')] - result.loc[:,('outstanding_shares2')] <= 0
    
#Operational Improvement
#Increasing ROA
    result.loc[:,('delta_roa')] = result.loc[:,('ROA')] - result.loc[:,('ROA2')] > 0
#Increasing FCFTA
    result.loc[:,('delta_FCFTA')] = result.loc[:,('FCFTA')] - result.loc[:,('FCFTA2')] > 0
#Increasing gross margin
    result.loc[:,('delta_gross_margin')] = result.loc[:,('gross_margin')] - result.loc[:,('gross_margin2')] > 0
#Increasing assets turnover
    result.loc[:,('delta_assets_turnover')] = result.loc[:,('assets_turnover')] - result.loc[:,('assets_turnover2')] > 0
    
    result = result.drop(['OCF','assets_turnover','assets_turnover2','debt_to_asset',
                      'debt_to_asset2','gross_margin','gross_margin2','outstanding_shares',             'outstanding_shares2','quick_ratio','quick_ratio2','ROA','ROA2','NI','NI2','total_revenue','total_revenue2','cash_return', 'cash_return2', 'enterprise_value', 'enterprise_value2',
                     'total_avg_assets','FCF','FCFTA','total_avg_assets2','FCF2','FCFTA2','fcf_yield'], axis=1)
    result = result.astype(int)

  #Sum row to get the score
    result.loc[:,('score')] = result.sum(axis=1) 
    result.loc[:,('fcf_yield')] = result2.loc[:,('fcf_yield')]
    result.sort_values(by=['fcf_yield'])
    result10 = result[result['score'] == 10]
    context.securities = result10.head(10)
    log.info("This week's longs: "+", ".join([long_.symbol for long_ in context.securities.index]))
    #print len(context.securities)
    
    
def optimal_portfolio(returns):
    """
    Finds the Optimal Portfolio according to the Markowitz Mean-Variance Model
    """
    
    n = len(returns)
    returns = np.asmatrix(returns)
    
    # print type(returns)
    # print returns
    
    N = 200
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    #print pbar
  
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = 10 #= np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


def my_rebalance(context,data):
    """
     Called daily to rebalance the Portfolio according to the equity weights calulated by optimal_portfolio() 
    """
     
    prices = data.history(context.securities.index, 'price', bar_count=252, frequency='1d').dropna()
    returns = prices.pct_change().dropna()
    returns = returns.values
    
    # print type(prices)
    # print prices
    for stock in list(context.portfolio.positions.keys()):  
        if stock not in context.securities.index:  
             if data.can_trade(stock):  
                 order_target_percent(stock, 0) 
    
    try:
        # Calculate weights by method of choice
        weights, _, _ = optimal_portfolio(returns.T)
        #weights = equal_portfolio(returns.T)
        leverage = sum(abs(weights))
       
        # Rebalance portfolio accordingly
        for stock, weight in zip(prices.columns, weights):
            order_target_percent(stock, weight/leverage)

    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass
    pass