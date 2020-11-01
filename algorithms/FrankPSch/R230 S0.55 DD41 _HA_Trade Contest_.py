from quantopian.algorithm import attach_pipeline, pipeline_output  
from quantopian.pipeline import Pipeline  
from quantopian.pipeline import CustomFactor  
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.data import morningstar 
import numpy as np
from collections import defaultdict

class MarketEquity(CustomFactor):
    window_length = 21
    inputs = [morningstar.valuation.market_cap]
    def compute(self,today,assets,out,mcap):
        out[:] = mcap[0]

class BookEquity(CustomFactor):
    window_length = 21
    inputs = [morningstar.balance_sheet.tangible_book_value]
    def compute(self,today,assets,out,book):
        out[:] = book[0]

class momentum_factor_1(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 20  
     
   def compute(self, today, assets, out, close):      
     out[:] = close[-1]/close[0]      
   
class momentum_factor_2(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 60  
     
   def compute(self, today, assets, out, close):      
     out[:] = close[-1]/close[0]   
   
class momentum_factor_3(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 125  
     
   def compute(self, today, assets, out, close):      
     out[:] = close[-1]/close[0]  
   
class momentum_factor_4(CustomFactor):    
   inputs = [USEquityPricing.close]   
   window_length = 252  
     
   def compute(self, today, assets, out, close):      
     out[:] = close[-1]/close[0]  
   
class market_cap(CustomFactor):    
   inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding]   
   window_length = 1  
     
   def compute(self, today, assets, out, close, shares):      
     out[:] = close[-1] * shares[-1]        
        
class efficiency_ratio(CustomFactor):    
   inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low]   
   window_length = 252
     
   def compute(self, today, assets, out, close, high, low):
       lb = self.window_length
       e_r = np.zeros(len(assets), dtype=np.float64)
       a=np.array(([high[1:(lb):1]-low[1:(lb):1],abs(high[1:(lb):1]-close[0:(lb-1):1]),abs(low[1:(lb):1]-close[0:(lb-1):1])]))      
       b=a.T.max(axis=1)
       c=b.sum(axis=1)
       e_r=abs(close[-1]-close[0]) /c  
       out[:] = e_r
        
def initialize(context):  
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    schedule_function(func=monthly_rebalance, date_rule=date_rules.month_start(days_offset=5), time_rule=time_rules.market_open(), half_days=True)  
    
    set_do_not_order_list(security_lists.leveraged_etf_list)
    context.acc_leverage = 1.00 
    context.holdings =10
    context.profit_taking_factor = 0.01
    context.profit_target={}
    context.profit_taken={}
    context.entry_date={}
    context.stop_pct = 0.75
    context.stop_price = defaultdict(lambda:0)
       
    pipe = Pipeline()  
    attach_pipeline(pipe, 'ranked_stocks')  
     
    factor1 = momentum_factor_1()  
    pipe.add(factor1, 'factor_1')   
    factor2 = momentum_factor_2()  
    pipe.add(factor2, 'factor_2')  
    factor3 = momentum_factor_3()  
    pipe.add(factor3, 'factor_3')  
    factor4 = momentum_factor_4()  
    pipe.add(factor4, 'factor_4') 
    factor5=efficiency_ratio()
    pipe.add(factor5, 'factor_5')

    mkt_cap = MarketEquity()
    pipe.add(mkt_cap,'market_cap')
    
    book_equity = BookEquity()
    # book equity over market equity
    be_me = book_equity/mkt_cap
    pipe.add(be_me,'be_me')    
  
    mkt_screen = market_cap()    
    stocks = mkt_screen.top(3000) 
    factor_5_filter = factor5 > 0.03
    total_filter = (stocks& factor_5_filter)
    pipe.set_screen(total_filter)  
     
    be_me_rank = be_me.rank(mask=total_filter, ascending=False)  
    pipe.add(be_me_rank, 'be_me_rank')         
    factor1_rank = factor1.rank(mask=total_filter, ascending=False)  
    pipe.add(factor1_rank, 'f1_rank')  
    factor2_rank = factor2.rank(mask=total_filter, ascending=False)  
    pipe.add(factor2_rank, 'f2_rank')  
    factor3_rank = factor3.rank(mask=total_filter, ascending=False)   
    pipe.add(factor3_rank, 'f3_rank')  
    factor4_rank = factor4.rank(mask=total_filter, ascending=False)  
    pipe.add(factor4_rank, 'f4_rank')  
    
    combo_raw = (factor1_rank+factor2_rank+factor3_rank+factor4_rank+1.5*be_me_rank)/5.5  
    pipe.add(combo_raw, 'combo_raw')   
    pipe.add(combo_raw.rank(mask=total_filter), 'combo_rank')       
         
def before_trading_start(context, data):  
    context.output = pipeline_output('ranked_stocks')  
   
    # Only consider stocks with a positive efficiency rating
    ranked_stocks = context.output[context.output.factor_5 > 0]
    
    # We are interested in the top 10 stocks ranked by combo_rank
    context.stock_factors = ranked_stocks.sort(['combo_rank'], ascending=True).iloc[:context.holdings]  
     
    context.stock_list = context.stock_factors.index  
           
def monthly_rebalance(context,data):
    
    # used to calculate order weights
    positions = set()
    
    for stock in context.stock_list:
        positions.add(stock)
    for stock in context.portfolio.positions:
        positions.add(stock)
        
    weight = context.acc_leverage / len(context.stock_list)    
    
    for stock in context.stock_list:  
        if stock in security_lists.leveraged_etf_list:
            continue
        if context.stock_factors.factor_1[stock] > 1:
            order_target_percent(stock, weight)  
            context.profit_target[stock] = data.current(stock, 'close')*1.25
     
    for stock in context.portfolio.positions:  
        if data.can_trade(stock) not in context.stock_list or context.stock_factors.factor_1[stock]<=1:  
            order_target(stock, 0)