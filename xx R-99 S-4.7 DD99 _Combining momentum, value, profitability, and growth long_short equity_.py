"""
**PF: Parking lot of things to check later. List in no particular order.
- how to reduce drawdown spans (strategy can result in ~3y periods with no net gain)
- how to avoid occasional liquidity (partial order) problems
- evaluating possibility of nonuniform weighting
- implementing overlapping holding periods (maybe order every 2 days and hold for 10)
- eliminating use of the built-in market_cap() method that is not supported in live trading
- how to safely use leverage > 1.0 (see Guy Fleury posts)
- exploring alternative entry/exit logic (vs the simple fast/slow SMA)

**PF: that is the parking lot for now
"""
#
# import methods and data
#
from quantopian.algorithm import attach_pipeline, pipeline_output  
from quantopian.pipeline import Pipeline  
from quantopian.pipeline import CustomFactor  
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.data import morningstar 
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.factors import ExponentialWeightedMovingAverage, SimpleMovingAverage
from quantopian.pipeline.factors import RSI
import numpy as np
from collections import defaultdict
import pandas as pd
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits as st
from quantopian.pipeline.experimental import QTradableStocksUS
import quantopian.algorithm as algo
import quantopian.optimize as opt


#
# define custom classes
#
class simple_momentum(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 1  
     
   def compute(self, today, assets, out, close):  
     out[:] = close[-1]/close[0]
        
class daily_momentum(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 1
    
    def compute(self, today, assets, out, close, open):
        out[:] = open[-1]/close[0]
class Mean_Reversion_1M(CustomFactor):
    
    """
    1-Month Mean Reversion:
    1-month returns minus 12-month average of monthly returns over standard deviation
    of 12-month average of monthly returns.
    https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf
    Notes:
    High value suggests momentum (short term)
    Equivalent to analysis of returns (12-month window)
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
      ret_1M = (close[-1]/close[-20])
      ret_1Y_monthly = (close[-1]/close[0])/12
      out[:] = (ret_1M - np.nanmean(ret_1Y_monthly))/np.nanstd(ret_1Y_monthly)
    
class Downside_Risk(CustomFactor):
   """
   Downside Risk:
   Standard Deviation of 12-month monthly losses
   https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf
   Notes:
   High value suggests high risk of losses
   """
   inputs = [USEquityPricing.close]
   window_length = 252

   def compute(self, today, assets, out, close):
       stdevs = []
       # get monthly closes
       close = close[0::21, :]
       for col in close.T:
           col_ret = ((col - np.roll(col, 1)) / np.roll(col, 1))[1:]
           stdev = np.nanstd(col_ret[col_ret < 0])
           stdevs.append(stdev)
       out[:] = stdevs

class augmented_momentum(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 1  
     
   def compute(self, today, assets, out, close):  
     best = np.nanmax(np.diff(close,axis=0),axis=0)
     out[:] = (close[-1]/close[0]) + (best/close[0])
   
class price_vs_max(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 252  
     
   def compute(self, today, assets, out, close):
     out[:] = close[-1]/np.nanmax(close, axis=0) 
   
class market_cap(CustomFactor):    
   inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding]   
   window_length = 1  
     
   def compute(self, today, assets, out, close, shares):      
     out[:] = close[-1] * shares[-1]          
           
class get_fcf_per_share(CustomFactor):    
   inputs = [morningstar.valuation_ratios.fcf_per_share]   
   window_length = 1  
     
   def compute(self, today, assets, out, fcf_per_share):   
     out[:] = fcf_per_share
                   
class get_last_close(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 1  
     
   def compute(self, today, assets, out, close):  
     out[:] = close[-1]

class earning_yield(CustomFactor):    
   inputs = [morningstar.valuation_ratios.earning_yield]   
   window_length = 1  
   
   def compute(self, today, assets, out, earning_yield):   
     out[:] = earning_yield
    
class roic(CustomFactor):    
   inputs = [morningstar.operation_ratios.roic]   
   window_length = 1  
   
   def compute(self, today, assets, out, roic):   
     out[:] = roic

class cash_return(CustomFactor):    
   inputs = [morningstar.valuation_ratios.cash_return]   
   window_length = 1  
   
   def compute(self, today, assets, out, cash_return):   
     out[:] = cash_return
    
class ebitda(CustomFactor):
    inputs = [morningstar.income_statement.ebitda]
    window_length = 1
    
    def compute(self, today, assets, out, ebitda):
        out[:] = ebitda
       
     
class net_ppe(CustomFactor):
    inputs = [morningstar.balance_sheet.net_ppe]
    window_length = 1
    
    def compute(self, today, assets, out, net_ppe):
        out[:] = net_ppe
        
class fcf_yield(CustomFactor):
    inputs = [morningstar.valuation_ratios.fcf_yield]
    window_length = 1
    
    def compute(self, today, assets, out, fcf_yield):
        out[:] = fcf_yield
        
class current_ratio(CustomFactor):
    inputs = [morningstar.operation_ratios.current_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, current_ratio):
        out[:] = current_ratio
        
class capex(CustomFactor):
    inputs = [morningstar.cash_flow_statement.capital_expenditure, morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, capital_expenditure, market_cap):
        out[:] = capital_expenditure[-1]/market_cap[-1]
        
class ocf(CustomFactor):
    inputs = [morningstar.cash_flow_statement.operating_cash_flow]
    window_length = 1
    
    def compute(self, today, assets, out, ocf):
        out[:] = ocf
        
class total_assets(CustomFactor):
    inputs = [morningstar.balance_sheet.total_assets]
    window_length = 1
    
    def compute(self, today, assets, out, total_assets):
        out[:] = total_assets
        
class roa(CustomFactor):    
   inputs = [morningstar.operation_ratios.roa]   
   window_length = 1  
   
   def compute(self, today, assets, out, roa):   
     out[:] = roa
    
class working_capital(CustomFactor):
    inputs = [morningstar.balance_sheet.working_capital]
    window_length = 1
    
    def compute(self, today, assets, out, working_capital):
        out[:] = working_capital
        
class roe(CustomFactor):    
   inputs = [morningstar.operation_ratios.roe]   
   window_length = 1  
   
   def compute(self, today, assets, out, roe):   
     out[:] = roe
    
class Alpha41(CustomFactor):   
    inputs = [USEquityPricing.low, USEquityPricing.high] 
    window_length = 1
    
    def compute(self, today, assets, out, low, high):
        out[:] = low[0]*high[0]
        
class fcf(CustomFactor):
    inputs = [morningstar.cash_flow_statement.free_cash_flow]
    window_length = 1
    
    def compute(self, today, assets, out, fcf):
        out[:] = fcf   
        
class ebit(CustomFactor):
    inputs = [morningstar.income_statement.ebit]
    window_length = 1
    
    def compute(self, today, assets, out, ebit):
        out[:] = ebit
class ev_to_ebitda(CustomFactor):
    inputs = [morningstar.valuation_ratios.ev_to_ebitda]
    window_length = 1
    
    def compute(self, today, assets, out, ev_to_ebitda):
        out[:] = ev_to_ebitda
        
class Quality(CustomFactor):
    
    inputs = [morningstar.income_statement.gross_profit, morningstar.balance_sheet.total_assets]
    window_length = 1
    
    def compute(self, today, assets, out, gross_profit, total_assets):       
        out[:] = gross_profit[-1] / total_assets[-1]
        

class Sortino(CustomFactor):
    inputs = [USEquityPricing.close]   
    window_length = 1  
     
    
    def compute(self, today, assets, out, close):  
        out[:] = (close[-1]/close[0])/np.std(close[-1]/close[0])
        
class EPS_Growth_3M(CustomFactor):
        """
        3-month Earnings Per Share Growth:
        Increase in EPS over 3 months
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf
        Notes:
        High value represents large growth (long term)
        """
        inputs = [morningstar.income_statement.normalized_income]
        window_length = 60

        def compute(self, today, assets, out, income):
            out[:] = income[-1] / income[0]
            
class revenue_growth(CustomFactor):
    inputs = [morningstar.operation_ratios.revenue_growth]
    window_length = 60
    
    def compute(self, today, assets, out, revenue_growth):
        out[:] = revenue_growth
              
class Stochastic_Oscillator(CustomFactor):
        """
        20-day Stochastic Oscillator:
        K = (close price - 5-day low) / (5-day high - 5-day low)
        D = 100 * (average of past 3 K's)
        We use the slow-D period here (the D above)
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf
        Notes:
        High value suggests turning point in positive momentum (expected decrease)
        Low value suggests turning point in negative momentum (expected increase)
        """
        inputs = [USEquityPricing.close,
                  USEquityPricing.high, USEquityPricing.low]
        window_length = 30

        def compute(self, today, assets, out, close, high, low):

            stoch_list = []

            for col_c, col_h, col_l in zip(close.T, high.T, low.T):
                try:
                    _, slowd = talib.STOCH(col_h, col_l, col_c,
                                           fastk_period=5, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)
                    stoch_list.append(slowd[-1])
                # if error calculating
                except:
                    stoch_list.append(np.nan)

            out[:] = stoch_list

    
#class Sharpe(CustomFactor):
#    inputs = [USEquityPricing.close]
#    returns = 
#    window_safe = True
#    window_length = 1   
#    
#    def compute(self, today, assets, out, returns):
#        out [:] = np.nanmean(returns,axis=0) / np.nanstd(returns,axis=0)
#        
#    returns = Returns(window_length=2, mask=universe)
#    returns = returns.log1p() # dont know if you like to use log returns
#    sharpe = SharpeRatio(inputs=[returns], window_length=30, mask=universe)
    
class Volatility(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 252
    
    def compute(self, today, assets, out, close):  
        close = pd.DataFrame(data=close, columns=assets) 
        # Since we are going to rank largest is best we need to invert the sdev.
        out[:] = np.log(close).diff().std()

class EPS_Growth(CustomFactor):
    inputs = [morningstar.earnings_ratios.diluted_eps_growth]
    window_length = 1
    
    def compute(self, today, assets, out, diluted_eps_growth):
        out[:] = diluted_eps_growth
        
class Operating_Cashflows_To_Assets(CustomFactor):
        """
        Operating Cash Flows to Total Assets:
        Operating Cash Flows divided by Total Assets.
        https://www.pnc.com/content/dam/pnc-com/pdf/personal/wealth-investments/WhitePapers/FactorAnalysisFeb2014.pdf
        Notes:
        High value suggests good efficiency, as more cash being used for operations is being used to generate more assets
        """
        inputs = [morningstar.cash_flow_statement.operating_cash_flow,
                  morningstar.balance_sheet.total_assets]
        window_length = 1

        def compute(self, today, assets, out, cfo, tot_assets):
            out[:] = (cfo[-1] * 4) / tot_assets[-1]
            
class SGnA(CustomFactor):
    inputs = [morningstar.income_statement.selling_general_and_administration]
    window_length = 1
    
    def compute(self, today, assets, out, SGnA):
        out[:] = SGnA
        
class Sector(CustomFactor):
    inputs = [morningstar.asset_classification.morningstar_sector_code]
    window_length = 1
    def compute(self, today, assets, out, sector):
        out[:] = sector
        
class total_yield(CustomFactor):
    inputs = [morningstar.valuation_ratios.total_yield]
    window_length = 1
    
    def compute(self, today, assets, out, total_yield):
        out[:] = total_yield
        
class ps_ratio(CustomFactor):
    inputs = [morningstar.valuation_ratios.ps_ratio]
    window_length = 1
    
    def compute(self, today, assets, out, ps_ratio):
        out[:] = ps_ratio
        
class Liquidity(CustomFactor):   
    inputs = [USEquityPricing.volume, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    def compute(self, today, assets, out, volume, shares):       
        out[:] = volume[-1]/shares[-1]       
        
class Price_Oscillator(CustomFactor):
        """
        4/52-Week Price Oscillator:
        Average close prices over 4-weeks divided by average close
        prices over 52-weeks all less 1.
        https://www.math.nyu.edu/faculty/avellane/Lo13030.pdf
        Notes:
        High value suggests momentum
        """
        inputs = [USEquityPricing.close]
        window_length = 252

        def compute(self, today, assets, out, close):
            four_week_period = close[-20:]
            out[:] = (np.nanmean(four_week_period, axis=0) /
                      np.nanmean(close, axis=0)) - 1.
            
#class Piotroski9(CustomFactor):
#    #Profitability:
#    # ROA = NetIncome/Total Assets > 0, i.e. NetIncome> 0. score +1
#    # OpCF> 0, score +1.
#    # ROA > ROAprevYr. score +1
#    # CFOps > Net Inc. score +1
#    # Leverage & Liquidity:
#    # LTdebtRatio < prevYr, score +1
#    # CR > prevYr, score +1
#    # No new shares issued, score +1    
#    # Op Eff:
#    # Gross margin > prevYr
#    # AssetTurn > prevYr, score +1
#    inputs = [morningstar.operation_ratios.roa, morningstar.cash_flow_statement.cash_flow_from_continuing_operating_activities, morningstar.income_statement.net_income, morningstar.operation_ratios.long_term_debt_equity_ratio, morningstar.operation_ratios.current_ratio, morningstar.valuation.shares_outstanding, morningstar.operation_ratios.gross_margin, morningstar.operation_ratios.assets_turnover]
#    window_length = 252
#    def compute(self, today, assets, out, roa, cash_flow_from_continuing_operating_activities, net_income, long_term_debt_equity_ratio, current_ratio, shares_outstanding, gross_margin, assets_turnover):
#        out[:] =  np.sign(cash_flow_from_continuing_operating_activities[-1] - net_income[-1])  + np.sign(gross_margin[-1] - gross_margin[-252]) + np.sign(shares_outstanding[-252] - shares_outstanding[-1]) + np.sign(long_term_debt_equity_ratio[-252] - long_term_debt_equity_ratio[-1]) + np.sign(assets_turnover[-1] - assets_turnover[-252]) #+ np.sign(roa[-1]) + np.sign(cash_flow_from_continuing_operating_activities[-1]) #np.sign(roa[-1] - roa[-252]) 
## Yes, sure I know this is not the real Piotroski score!
#
        
class Piotroski2(CustomFactor):
    inputs = [morningstar.operation_ratios.assets_turnover]
    window_length = 252
    
    def compute(self, today, assets, out, assets_turnover):
        out[:] = assets_turnover[-1] - assets_turnover[-252]
        
class Piotroski3(CustomFactor):
    inputs = [morningstar.operation_ratios.current_ratio]
    window_length = 252
    
    def compute(self, today, assets, out, current_ratio):
        out[:] = current_ratio[-1] - current_ratio[-252]
        
class Piotroski4(CustomFactor):
    inputs = [morningstar.operation_ratios.operation_margin]
    window_length = 252
    
    def compute(self, today, assets, out, operation_margin):
        out[:] = operation_margin[-1]/operation_margin[-252]
#class AltmanZ(CustomFactor):  
#    # alt_A = WC / Total Assets 
#    # alt_B = Retained earnings / Total Assets 
#    # alt_C = EBIT  / Total Assets 
#    # alt_D = MktVal of Equity / Total Liabilities
#    # alt_E = Sales / Total Assets
#    # AltmanZ = 1.2*A + 1.4*B +3.3*C +0.6*D +1.0*E
#    inputs = [income_statement.total_assets, income_statement.working_capital, income_statement.retained_earnings, income_statement.ebit, valuation_ratios.market_cap, income_statement.total_liabilities, income_statement.total_revenue]
#    window_length = 252
#    def compute(self, today, assets, out, total_assets, working_capital, retained_earnings, ebit, market_cap, total_liabilities, total_revenue):  
#        out[:] = 1.2*(working_capital[-1]/total_assets[-1]) + 1.4*(retained_earnings[-1]/total_assets[-1]) + 3.3*(ebit[-1]/total_assets[-1]) + 0.6*(market_cap[-1]/total_liabilities[-1]) + 1.0*(total_revenue[-1]/total_assets[-1])
#        

class Momentum(CustomFactor):
    """
    Here we define a basic momentum factor using a CustomFactor. We take
    the momentum from the past year up until the beginning of this month
    and penalize it by the momentum over this month. We are tempering a 
    long-term trend with a short-term reversal in hopes that we get a
    better measure of momentum.
    """
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, prices):
        out[:] = ((prices[-21] - prices[-252])/prices[-252] -
                  (prices[-1] - prices[-21])/prices[-21])
    
class RnD_to_market(CustomFactor):
    
    inputs = [morningstar.income_statement.research_and_development, morningstar.valuation.market_cap]
    window_length = 1
    
    def compute(self, today, assets, out, research_and_development, market_cap):
        out[:] = research_and_development[-1]/market_cap[-1]
        
#class PredictionQuality(CustomFactor):
#    """
#    create a customized factor to calculate the prediction quality
#    for each stock in the universe.
#    
#    compares the percentage of predictions with the correct sign 
#    over a rolling window (3 weeks) for each stock
#   
#    """
#
#    # data used to create custom factor
#    inputs = [precog.predicted_five_day_log_return, USEquityPricing.close]
#
#    # change this to what you want
#    window_length = 15
#
#    def compute(self, today, assets, out, pred_ret, px_close):
#
#        # actual returns
#        px_close_df = pd.DataFrame(data=px_close)
#        pred_ret_df = pd.DataFrame(data=pred_ret)
#        log_ret5_df = np.log(px_close_df) - np.log(px_close_df.shift(5))
#
#        log_ret5_df = log_ret5_df.iloc[5:].reset_index(drop=True)
#        n = len(log_ret5_df)
#        
#        # predicted returns
#        pred_ret_df = pred_ret_df.iloc[:n]
#
#        # number of predictions with incorrect sign
#        err_df = (np.sign(log_ret5_df) - np.sign(pred_ret_df)).abs()/2.0
#
#        # custom quality measure
#        pred_quality = (1 - pd.ewma(err_df, min_periods=n, com=n)).iloc[-1].values
#        
#        out[:] = pred_quality
#
#        
#        
#class NormalizedReturn(CustomFactor):
#    """
#    Custom Factor to calculate the normalized forward return 
#       
#    scales the forward return expecation by the historical volatility
#    of returns
#    
#    """
#
#    # data used to create custom factor
#    inputs = [precog.predicted_five_day_log_return, USEquityPricing.close]
#    
#    # change this to what you want
#    window_length = 10
#
#    def compute(self, today, assets, out, pred_ret, px_close):
#        
#        # mean return 
#        avg_ret = np.nanmean(pred_ret[-1], axis =0)
#        
#        # standard deviation of returns
#        std_ret = np.nanstd(pred_ret[-1], axis=0)
#
#        # normalized returns
#        norm_ret = (pred_ret[-1] -avg_ret)/ std_ret
#
#        out[:] = norm_ret

class Reversion(CustomFactor):
    """
    Here we define a basic mean reversion factor using a CustomFactor. We
    take a ratio of the last close price to the average price over the
    last 60 days. A high ratio indicates a high price relative to the mean
    and a low ratio indicates a low price relative to the mean.
    """
    inputs = [USEquityPricing.close]
    window_length = 60

    def compute(self, today, assets, out, prices):
        out[:] = -prices[-1] / np.mean(prices, axis=0)

def initialize(context):  
    
    schedule_function(func=periodic_rebalance,
                      date_rule=date_rules.every_day(),#week_start(days_offset=1),
                      time_rule=time_rules.market_open(hours=.5)) 
    
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=0, minutes=1))
    #schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=1, minutes=1))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=2, minutes=1))
    #schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=3, minutes=1))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=4, minutes=1))
    #schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=5, minutes=1))
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(hours=6, minutes=1))
    
    #schedule_function(
    #    do_portfolio_construction,
    #    date_rule=algo.date_rules.week_start(),
    #    time_rule=algo.time_rules.market_open(minutes=30),
    #    half_days=False,
    #    )
        
    for hours_offset in range(7):
        schedule_function(
            my_rebalance, 
            date_rules.every_day(), 
            time_rules.market_open(hours=hours_offset, minutes=10),
            half_days = True)

#
# set portfolis parameters
#
    set_asset_restrictions(security_lists.restrict_leveraged_etfs)
    context.acc_leverage = 1.00
    context.min_holdings = 20
    context.s_min_holdings = 10
#
# set profit taking and stop loss parameters
#
    context.profit_taking_factor = 0.01
    context.profit_taking_target = 100.0 #set much larger than 1.0 to disable
    context.profit_target={}
    context.profit_taken={}
    context.stop_pct = 0.97    # set to 0.0 to disable
    context.stop_price = defaultdict(lambda:0)
#
# Set commission model to be used
#
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

# Define safe set (of bonds)
#sid(32268) #SH
    context.safe = [
      sid(23870), #IEF
      sid(23921), #TLT
      #sid(8554), #SPY
    ]
#
# Define proxy to be used as proxy for overall stock behavior
# set default position to be in safe set (context.buy_stocks = False)
#
    context.canary = sid(22739)#why not spy
    context.buy_stocks = False
#
# Establish pipeline
#
    pipe = Pipeline()  
    attach_pipeline(pipe, 'ranked_stocks') 
    
#
# Define the four momentum factors used in ranking stocks
#
    factor1 = simple_momentum(window_length=1)
    pipe.add(factor1, 'factor_1')  
    factor2 = simple_momentum(window_length=60)/Volatility(window_length=60)
    pipe.add(factor2, 'factor_2')  
    factor3 = simple_momentum(window_length=252)  
    pipe.add(factor3, 'factor_3')  
    factor4 = ((Momentum()/Volatility())+Momentum())#or Downside_Risk() 
    pipe.add(factor4, 'factor_4')
    factor8 = earning_yield()
    pipe.add(factor8, 'factor8')
    factor9 = roe()+roic()+roa()
    pipe.add(factor9, 'factor9')
    factor10 = cash_return()
    pipe.add(factor10, 'factor10')
    factor11 = fcf_yield()
    pipe.add(factor11, 'factor11')
    factor12 = current_ratio()
    pipe.add(factor12, 'factor12')
    factor13 = Quality()
    pipe.add(factor13, 'factor13')
    factor14 = market_cap()
    pipe.add(factor14, 'factor14')
    factor15 = RnD_to_market()+capex()
    pipe.add(factor15, 'factor15')
    factor18= EPS_Growth_3M()
    pipe.add(factor18, 'factor18')                      
    factor19 = Piotroski4()
    pipe.add(factor19, 'factor19')
    factor20 = capex()
    pipe.add(factor20, 'factor20')
#
# Define other factors that may be used in stock screening
#
    #factor5 = get_fcf_per_share()
    #pipe.add(factor5, 'factor_5')
    factor6 = AverageDollarVolume(window_length=60)
    pipe.add(factor6, 'factor_6')
    factor7 = get_last_close()
    pipe.add(factor7, 'factor_7')
    

    #factor_4_filter = factor4 > 1.03   # only consider stocks with positive 1y growth
    #factor_5_filter = factor5 > 0.0   # only  consider stocks with positive FCF
    factor_6_filter = factor6 > .5e6 # only consider stocks trading >$500k per day
    #factor_7_filter = factor7 > 3.00  # only consider stocks that close above this value 
    factor_12_filter = factor12 > .99
    #factor_8_filter = factor8 > 0
    #factor_15_filter = factor15 > factor6
    #factor_1_filter = factor1 > 1.1
    #factor_2_filter = factor2 > 1
    #factor_20_filter = factor20 > 0
    utilities_filter = Sector()!=207
    materials_filter = Sector()!=101
    energy_filter = Sector()!=309
    industrial_filter = Sector()!=310
    health_filter = Sector()!=206
    staples_filter = Sector()!=205
    real_estate_filter = Sector()!=104
    #sentiment_filter = ((0.5*st.bull_scored_messages.latest)>(st.bear_scored_messages.latest)) & (st.bear_scored_messages.latest > 10)
    consumer_cyclical_filter = Sector()!=102
    financial_filter = Sector()!=103
    communication_filter = Sector()!=308
    technology_filter = Sector!=311
    
    #Basic_Materials = context.output[context.output.sector == 101]
    #Consumer_Cyclical = context.output[context.output.sector == 102]
    #Financial_Services = context.output[context.output.sector == 103]
    #Real_Estate = context.output[context.output.sector == 104]
    #Consumer_Defensive = context.output[context.output.sector == 205]
    #Healthcare = context.output[context.output.sector == 206]
    #Utilities = context.output[context.output.sector == 207]
    #Communication_Services = context.output[context.output.sector == 308]
    #Energy = context.output[context.output.sector == 309]
    #Industrials = context.output[context.output.sector == 310]
    #Technology = context.output[context.output.sector == 311]
#
# Establish screen used to establish candidate stock list
#
    mkt_screen = market_cap()
    cash_flow = factor10+factor11
    price = factor14
    profitability = factor9
    #earning_quality = factor15
    stocks = QTradableStocksUS()  #mkt_screen.top(3500)&profitability.top(3500)&factor19.top(2000)#&factor8.top(2000)#&price.top(2000)#&factor15.top(3000)#
    total_filter = (stocks
                    & factor_6_filter  
                    #& factor_15_filter
                    #& factor_8_filter
                    #& factor_9_filter
                    #& factor_1_filter
                    #& factor_20_filter
                    #& communication_filter
                    #& consumer_cyclical_filter
                    #& financial_filter
                    #& staples_filter
                    #& materials_filter
                    #& industrial_filter
                    #& factor_12_filter
                    #& technology_filter
                    )

    pipe.set_screen(total_filter)
#
# Establish ranked stock list
#
    factor1_rank = factor1.rank(mask=total_filter, ascending=False)  
    pipe.add(factor1_rank, 'f1_rank')  
    factor2_rank = factor2.rank(mask=total_filter, ascending=False)  
    pipe.add(factor2_rank, 'f2_rank')  
    factor3_rank = factor3.rank(mask=total_filter, ascending=False) #significant effect 
    pipe.add(factor3_rank, 'f3_rank')  
    factor4_rank = factor4.rank(mask=total_filter, ascending=False) #significant effect 
    pipe.add(factor4_rank, 'f4_rank')
    factor8_rank = factor8.rank(mask=total_filter, ascending=False) #significant effect
    pipe.add(factor8_rank, 'f8_rank')  
    factor9_rank = factor9.rank(mask=total_filter, ascending=False) #very big effect  
    pipe.add(factor9_rank, 'f9_rank') 
    factor10_rank = factor10.rank(mask=total_filter, ascending=False)
    pipe.add(factor10_rank, 'f10_rank')
    factor11_rank = factor11.rank(mask=total_filter, ascending=False)
    pipe.add(factor11_rank, 'f11_rank')
    factor13_rank = factor13.rank(mask=total_filter, ascending=False)#may want to remove
    pipe.add(factor13_rank, 'f13_rank')
    factor14_rank = factor14.rank(mask=total_filter, ascending=True)
    pipe.add(factor14_rank, 'f14_rank')
    factor15_rank = factor15.rank(mask=total_filter, ascending=False) 
    pipe.add(factor15_rank, 'f15_rank')
    factor18_rank = factor18.rank(mask=total_filter, ascending=False) 
    pipe.add(factor18_rank, 'f18_rank')
    factor19_rank = factor19.rank(mask=total_filter, ascending=False)
    pipe.add(factor19_rank, 'f19_rank')
    factor20_rank = factor20.rank(mask=total_filter, ascending=False)  
    pipe.add(factor20_rank, 'f20_rank')
    
    combo_raw = (factor8_rank+factor18_rank+factor1_rank+factor4_rank+factor10_rank+factor11_rank+factor15_rank+factor9_rank+factor19_rank)#+factor14_rank*.5)
    pipe.add(combo_raw, 'combo_raw')   
    pipe.add(combo_raw.rank(mask=total_filter), 'combo_rank')
    
def gather_data(data):   
    # Gathers data for an arbitrary number of days and refreshes
    # Also serves to make sure that data exists 
    return data
 
''' 
    Adjusts a list of past prices to the number of periods you want 
    So if you want the nummber of prices in the last forty days, set period = 40
'''
def gather_prices(context, data, sid, period):
    context.past_prices.append(data[sid].price)
    if len(context.past_prices) > period:
        context.past_prices.pop(0)
    return
    
'''
    Hurst exponent helps test whether the time series is:
    (1) A Random Walk (H ~ 0.5)
    (2) Trending (H > 0.5)
    (3) Mean reverting (H < 0.5)
'''
def hurst(context, data, sid):
    # Gathers all the prices that you need
    gather_prices(context, data, sid, 40)
    # Checks whether data exists
    data_gathered = gather_data(data)
    if data_gathered is None:
        return    
    
    tau, lagvec = [], []
    # Step through the different lags
    for lag in range(2,20):  
        # Produce price different with lag
        pp = np.subtract(context.past_prices[lag:],context.past_prices[:-lag])
        # Write the different lags into a vector
        lagvec.append(lag)
        # Calculate the variance of the difference
        tau.append(np.sqrt(np.std(pp)))
    # Linear fit to a double-log graph to get power
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)
    # Calculate hurst
    hurst = m[0]*2
    
    return hurst
        

def before_trading_start(context, data):  
#
# Calculate maximum number of stocks to buy
#
    n_30 = int(context.portfolio.portfolio_value/50e3)
    context.holdings = max(context.min_holdings, n_30)
    context.s_holdings = max(context.s_min_holdings, n_30)
#
# Screen to find the current top stocks
#
    context.output = pipeline_output('ranked_stocks')     
    ranked_stocks = context.output
    #context.s_output = pipeline_output('ranked_stocks')     
    #s_ranked_stocks = context.s_output
    context.stock_factors = ranked_stocks.sort(['combo_rank'], ascending=True).iloc[:context.holdings]                     
    context.s_stock_factors = ranked_stocks.sort(['combo_rank'], ascending=False).iloc[:context.s_holdings]
    context.stock_list = context.stock_factors.index 
    context.s_stock_list = context.s_stock_factors.index   
#
# Use fast/slow SMA test of proxy to determine whether to be in stocks vs safe
#
    Canary = data.history(context.canary, 'price', 80, '1d')
    Canary_fast = Canary[-10:].mean()
    Canary_slow = Canary.mean()    
    context.buy_stocks = False
    if Canary_fast > Canary_slow:context.buy_stocks = True
    
    context.pipeline_data = algo.pipeline_output('ranked_stocks')

def my_rebalance(context, data):    
#
# Do daily maintenance
#    a) sell obsolete positions
#    b) implement stop loss
#    c) implement profit taking
#    d) record values for backtest display
#  
#
# Sell any holdings that are not in context.this_periods_list
#

    
    for stock in context.portfolio.positions:
        if data.can_trade(stock):
            if stock not in context.this_periods_list: #context.stock_list or context.s_stock_list:
                order_target(stock, 0)
#
# update stop loss limits and sell any stocks that are below their limits
#
    for stock in context.portfolio.positions:
        if data.can_trade(stock):
            price = data.current(stock, 'price')
            context.stop_price[stock] = max(context.stop_price[stock], 
                                            context.stop_pct * price)
            if price < context.stop_price[stock]:
                order_target(stock, 0)
                context.stop_price[stock] = 0
                log.info("%s stop loss"%stock)
#
# Profit take if profit target is met
# Skip this for safe set assets
#
    takes = 0
    for stock in context.portfolio.positions:
        if stock not in context.safe:
            if data.can_trade(stock) and data.current(stock, 'close') > context.profit_target[stock]:
                context.profit_target[stock] = data.current(stock, 'close')*1.1 #profit target
                profit_taking_amount = context.portfolio.positions[stock].amount * context.profit_taking_factor
                takes += 1
                log.info(profit_taking_amount)
                order_target(stock, profit_taking_amount) 
#
# Record parameters
#
    n100 = len(context.output)/100
    record(leverage=context.account.leverage, 
           positions=len(context.portfolio.positions), 
           #t=takes,
           candidates=n100)  
           
def periodic_rebalance(context,data):    
#
# rebalance portfolio based on most recent context.buy_stocks signal
#
# rebalance portfolio in stocks
#
    if context.buy_stocks:
        context.this_periods_list = np.concatenate((context.stock_list,context.s_stock_list,context.safe))
    #    context.this_periods_list = [context.stock_list, context.s_stock_list]
    #    context.s_this_periods_list = context.s_stock_list
    #
    # sell any holdings not in this period's stock list
    #
        for stock in context.portfolio.positions:  
            if data.can_trade(stock):
                if stock not in context.this_periods_list: #context.stock_list or context.s_stock_list:
                    order_target(stock, 0)
                    
        #order_target_percent(sid(8554), 1)
    # equally weight portfolio over assets that can trade
    # set profit_target threshold based on recent close
    #                   
        weight = context.acc_leverage / len(context.stock_list)
        p_tgt = context.profit_taking_target
        for stock in context.stock_list:  
            if stock in security_lists.leveraged_etf_list:
                continue 
            if data.can_trade(stock): #can remove the factor_1
                order_target_percent(stock, weight*.8)  
                context.profit_target[stock] = data.current(stock, 'close')*p_tgt
                print(stock)
                                
        s_weight = context.acc_leverage / len(context.s_stock_list)
        s_p_tgt = context.profit_taking_target
        for stock in context.s_stock_list:  
            if stock in security_lists.leveraged_etf_list:
                continue 
            if data.can_trade(stock): #can remove the factor_1
                order_target_percent(stock, -s_weight*.1)  
                context.profit_target[stock] = data.current(stock, 'close')*s_p_tgt
                print(stock)
               
               
        n = 0
        for stock in context.safe:
            if data.can_trade(stock): n += 1
        if n > 0:
            weight = 1.0/n
            for stock in context.safe:
                if data.can_trade(stock): 
                    order_target_percent(stock, weight*.1)
#
# otherwise put portfolio into safe set
#
    else:                
        context.this_periods_list = np.concatenate((context.stock_list,context.s_stock_list,context.safe)) #contex.safe
    #
    # sell any holdings not in safe set
    #
        for stock in context.portfolio.positions:  
            if data.can_trade(stock):
                if stock not in context.this_periods_list:
                    order_target(stock, 0)
    
     #equally weight portfolio over safe assets that can trade
    
        weight = context.acc_leverage / len(context.stock_list)
        p_tgt = context.profit_taking_target
        for stock in context.stock_list:  
            if stock in security_lists.leveraged_etf_list:
                continue 
            if data.can_trade(stock): #can remove the factor_1
                order_target_percent(stock, weight*.15)  
                context.profit_target[stock] = data.current(stock, 'close')*p_tgt
                print(stock)
                                 
        s_weight = context.acc_leverage / len(context.s_stock_list)
        s_p_tgt = context.profit_taking_target
        for stock in context.s_stock_list:  
            if stock in security_lists.leveraged_etf_list:
                continue 
            if data.can_trade(stock): #can remove the factor_1
                order_target_percent(stock, -s_weight*.55)  
                context.profit_target[stock] = data.current(stock, 'close')*s_p_tgt
                print(stock)
        n = 0
        for stock in context.safe:
            if data.can_trade(stock): n += 1
        if n > 0:
            weight = 1.0/n
            for stock in context.safe:
                if data.can_trade(stock): 
                    order_target_percent(stock, weight*.3)
#
#def do_portfolio_construction(context, data):
#    pipeline_data = context.pipeline_data
#
#    # Objective
#    # ---------
#    # For our objective, we simply use our naive ranks as an alpha coefficient
#    # and try to maximize that alpha.
#    # 
#    # This is a **very** naive model. Since our alphas are so widely spread out,
#    # we should expect to always allocate the maximum amount of long/short
#    # capital to assets with high/low ranks.
#    #
#    # A more sophisticated model would apply some re-scaling here to try to generate
#    # more meaningful predictions of future returns.
#    #objective = opt.MaximizeAlpha(pipeline_data.combo_rank)
#    # Constrain ourselve to have a net leverage of 0.0 in each sector.
#    sector_neutral = opt.NetGroupExposure.with_equal_bounds(
#        labels=pipeline_data.sector,
#        min=-0.001,
#        max=0.001,
#    )
#
#    # Run the optimization. This will calculate new portfolio weights and
#    # manage moving our portfolio toward the target.
#    algo.order_optimal_portfolio(
#        constraints=[
#            sector_neutral,
#        ],
#    )