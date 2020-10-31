# From 
"""
Adapted from "A simple momentum rotation system for stocks"
https://www.quantopian.com/posts/a-simple-momentum-rotation-system-for-stocks

The unmodified performance of this algorithm is remarkable from 1/4/2003 to 11/30/2015
Total Returns 1287%    Benchmark 192.5%    Max Drawdown    50.4%
Alpha    0.87    Beta    0.85    Sharpe    3.25    Volatility    0.30
Method outline is:
Buy and hold best 10 of 3000 stocks each month
During the month sell big losers (stop loss) and big winners (profit taking)
Selection considers
- Four momentum factors over 20, 60, 125 and 252 days
- Efficiency threshold = 0.031 based on 252 day return vs sum of daily High minus Low

The original post successfully outlines a method outline of use to the community. There was no attempt to make this a production-ready algorithm, so some problems and uncertain features exist. A few of these impacted my ability to understand what was happening so I tried to resolve them (perhaps only to myself):
1) although this is nominally a long-only algo the daily rebalance can result in shorting
2) liquidity problem even when starting with only $100k. Leverage is roughly 35% to 110%. The number of assets is roughtly 4 to 15 vs defined top-10.
3) the algorithm lacks any logic to exit stocks during prolonged drawdown periods. Real investors would have exited a few times from 2003 through 2015.
4) the utility of the efficiency factor test is not clear. The threshold of 0.031 appears to be oddly low as efficiency could easily be much larger.

Below is a summary of what I did to resolve/improve issues 1-3 and my finding that the utility of the efficiency function can be had more simply by requiring the 252-day return (factor_4) to be > 0.0.
I tried to leave the rest of the algorithm as is. Performance is evaluated over the same 1/4/2003 to 11/30/2015 period as the original posting. Another tester might investigate other interesting features of Garner's algorithm (ranking periods, profit taking logic, ...)

Shorting issue (resolved in one change)
I modified the code to issue sell orders for obsolete postions before issuing buy orders for new positions.
This has resolved the problem and improved overall return as the stocks being shorted were probably not good shorting candidates.

Liquidity problems (resolved in three changes)
Leverage often exceeds 1.0 due to an inability to sell obsolete positions in a single trading session.
Leverage 1: Add a function to daily rebalance to continue sales of these positions
This did drive the leverage down to 1.0 quickly in all but a few cases.
As expected the total return also dropped as the average leverage was reduced and more trade fees were paid.
Total Returns 1163%    Benchmark 192.5%    Max Drawdown    52.1%
Alpha    0.78    Beta    0.88    Sharpe    2.85    Volatility    0.31

Leverage 2: Add Average Daily Dollar Volume (ADDV) as a filter factor.
Consider only stocks with ADDV > $500k over the past 20 days
This nearly eliminated the need to sell obsolete stocks on multiple days until portfolio size got much bigger ~ $500k
This did improve overall returns
Total Returns 1314%    Benchmark 192.5%    Max Drawdown    48.1%
Alpha    0.88    Beta    0.89    Sharpe    3.17    Volatility    0.31

Leverage 3: Allow the number of equities to increase with portfolio value
Try context.holdings = max(10, int( portfolio_value/30e3 )
As expected this reduced volatility. It also had some benefit to overall return
Total Returns 1356%    Benchmark 192.5%    Max Drawdown    48.5%
Alpha    0.91    Beta    0.91    Sharpe    3.72    Volatility    0.28

Drawdown protection (improved to acceptable level)
Add a simple drawdown protection based on simple moving averages of SPY
If SPY_SMA_fast < SPY_SMA_slow, then go to cash; else use the algorithm
Fast period should be on the order of the shortest momentum filter (20 days)
Since SMA filter is slower than EMA a period less than 20 days is desired.
Slow period should be several multiples of the fast period, but not slower than the overall algo.
The geometric average of the four periods (20,60,125,252) is 78 days
A 15/80 day test provided good drawdown reduction (26% vs 48%) with about 10% loss in total return
15/80 Cash  Total return 1204%    Alpha 0.85    Sharpe 4.00    Max DD 26%

Most asset allocation models would exit to bonds vs cash, so that was tried as well
Bond set = [TLT, IEF, AGG]
15/80 Bonds   Total return 1790%    Alpha 1.32    Sharpe 6.07    Max DD 20%
This is a nice result. A somewhat better result might be had by allowing rotation between stocks, bonds, cash, or some combination of stocks/bonds, but that is beyond my current purpose.

What is effect of the ADDV limit?
ADDV limit. $30k per holding and $100k initial investment.
Exiting to bonds when indicated by 15/80 SMA test
$0.2M:  Total return 1810%    Alpha 1.34    Sharpe 6.15    Max DD 20%
$0.5M:  Total return 1790%    Alpha 1.32    Sharpe 6.07    Max DD 20%
$1.5M:  Total return 1546%    Alpha 1.13    Sharpe 5.00    Max DD 20%

What is the effect of the efficiency threshold?
I tried several values as shown below
Any limit > 0.0 has a good result until some point above 0.5.
Garner's 0.031 recommendedation for his top 10 algorithm looks good.
My finding is for a variable and larger set of equities (10 to 60 in any trial).

Intermediate is the return reported for week of 1/3/2010 (near midpoint)
Limit  0.0      total return 1815%    intermediate    848%    Sharpe    6.15
Limit  0.031    total return 1790%    intermediate    836%    Sharpe    6.07
Limit  0.1      total return 1786%    intermediate    818%    Sharpe    6.05
Limit  0.2      total return 1784%    intermediate    813%    Sharpe    6.03
Limit  0.4      total return 1799%    intermediate    791%    Sharpe    6.09
Limit  0.5      total return 1764%    intermediate    809%    Sharpe    5.97
Limit  0.7      total return 1550%    intermediate    739%    Sharpe    5.20
==> might as well use a limit of 0.0
==> This is equivalent to stating factor_4 > 0.0 which is easier to implement.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output  
from quantopian.pipeline import Pipeline  
from quantopian.pipeline import CustomFactor  
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.data import morningstar 
from quantopian.pipeline.factors import AverageDollarVolume, Returns
import numpy as np
import pandas as pd
import talib
from collections import defaultdict

fct_window_length_1 = 20
fct_window_length_2 = 60
fct_window_length_3 = 125
fct_window_length_4 = 252

class MomentumRanking(CustomFactor):
    inputs = [USEquityPricing.close]   
    window_length = 252 
     
    def compute(self, today, assets, out, close): 
#        close[close <= 5] = np.nan # Get ride of penny stock
        value_table = pd.DataFrame(index=assets)
        value_table['mom1'] = close[-1] / close[-fct_window_length_1] - 1                   
        value_table['mom2'] = close[-1] / close[-fct_window_length_2] - 1                   
        value_table['mom3'] = close[-1] / close[-fct_window_length_3] - 1                    
        value_table['mom4'] = close[-1] / close[0] - 1   
        
        out[:] = value_table.rank(ascending = False).mean(axis=1) 

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
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.00))
    schedule_function(func=monthly_rebalance, date_rule=date_rules.month_start(days_offset=5), time_rule=time_rules.market_open(), half_days=True)  
    schedule_function(func=daily_rebalance, date_rule=date_rules.every_day(), time_rule=time_rules.market_close(hours=1))
    
    set_do_not_order_list(security_lists.leveraged_etf_list)
    context.canary = sid(8554)
    context.acc_leverage = 1.00 
    context.min_holdings = 10
    context.profit_taking_factor = 0.01
    context.profit_target={}
    context.profit_taken={}
    context.entry_date={}
    context.stop_pct = 0.75
    context.stop_price = defaultdict(lambda:0)
    context.no_trade_yet = True
    context.buy_stocks = False
#    context.fct_window_length_1 = 20
#    context.fct_window_length_2 = 60
#    context.fct_window_length_3 = 125
#    context.fct_window_length_4 = 252
    
    context.safe = [
      sid(23870), #IEF
      sid(23921), #TLT
      sid(25485)  #AGG
    ]
       
    pipe = Pipeline()  
    attach_pipeline(pipe, 'ranked_stocks')  
    
#    factor1 = momentum_factor_1()  
    factor1 = Returns(window_length=fct_window_length_1)  
    pipe.add(factor1, 'factor_1')   
    '''
#    factor2 = momentum_factor_2()  
    factor2 = Returns(window_length=context.fct_window_length_2)  
    pipe.add(factor2, 'factor_2')  
#    factor3 = momentum_factor_3()  
    factor3 = Returns(window_length=context.fct_window_length_3)  
    pipe.add(factor3, 'factor_3')  
#    factor4 = momentum_factor_4()  
    factor4 = Returns(window_length=context.fct_window_length_4)  
    pipe.add(factor4, 'factor_4') 
    '''    
    factor5=efficiency_ratio()
    pipe.add(factor5, 'factor_5')
    factor6 = AverageDollarVolume(window_length=20)
    pipe.add(factor6, 'factor_6')
        
    mkt_screen = market_cap()    
    stocks = mkt_screen.top(3000) 
    factor_5_filter = factor5 > 0.0
    factor_6_filter = factor6 > 0.5e6 # only consider stocks trading >$500k per day
    total_filter = (stocks & factor_5_filter & factor_6_filter)
    pipe.set_screen(total_filter)  
     
    '''        
    factor1_rank = factor1.rank(mask=total_filter, ascending=False)  
#    pipe.add(factor1_rank, 'f1_rank')  
    factor2_rank = factor2.rank(mask=total_filter, ascending=False)  
#    pipe.add(factor2_rank, 'f2_rank')  
    factor3_rank = factor3.rank(mask=total_filter, ascending=False)   
#    pipe.add(factor3_rank, 'f3_rank')  
    factor4_rank = factor4.rank(mask=total_filter, ascending=False)  
#    pipe.add(factor4_rank, 'f4_rank')  
    '''   
#    combo_raw = (factor1_rank+factor2_rank+factor3_rank+factor4_rank)/4  
#    pipe.add(combo_raw, 'combo_raw')   
    combo_rank = MomentumRanking(mask=total_filter)
    pipe.add(combo_rank, 'combo_rank')
#    pipe.add(combo_raw.rank(mask=total_filter), 'combo_rank')       
#    pipe.add(mom_rank.rank(mask=total_filter), 'combo_rank')       
         
def before_trading_start(context, data):  
    context.output = pipeline_output('ranked_stocks').dropna()  
    log.info("Original DF:\n%s" %context.output.head(3))
    
    n_30 = int(context.portfolio.portfolio_value/30e3)
    context.holdings = max(context.min_holdings, n_30)
   
    # Only consider stocks with a efficiency rating > threshold
    ranked_stocks = context.output[context.output.factor_5 > 0.0]

    
    # We are interested in the top 10 stocks ranked by combo_rank
    context.stock_factors = ranked_stocks.sort(['combo_rank'], ascending=True).iloc[:context.holdings]  
     
    context.stock_list = context.stock_factors.index   

#
# Entry/exit logic using slow/fast SMA
#
    Canary = data.history(context.canary, 'price', 80, '1d')
    Canary_fast = Canary[-15:].mean()
    Canary_slow = Canary.mean()
    
    context.buy_stocks = False
    if Canary_fast > Canary_slow: context.buy_stocks = True

def daily_rebalance(context, data):

    for stock in context.portfolio.positions:
        if data.can_trade(stock):
            if stock not in context.this_months_list:
                order_target(stock, 0)
                
    for stock in context.portfolio.positions:
        if data.can_trade(stock):
            # Set/update stop price
            price = data.current(stock, 'price')
            context.stop_price[stock] = max(context.stop_price[stock], context.stop_pct * price)
            
            # Check stop price, sell if price is below it
            if price < context.stop_price[stock]:
                order_target(stock, 0)
                context.stop_price[stock] = 0
                log.info("%s stop loss"%stock)
                
# Increase our position in stocks that are performing better than their target and reset the target
    takes = 0
    for stock in context.portfolio.positions:
        if stock not in context.safe: # don't profit take on bonds
            if data.can_trade(stock) and data.current(stock, 'close') > context.profit_target[stock]:
                context.profit_target[stock] = data.current(stock, 'close')*1.25
                profit_taking_amount = context.portfolio.positions[stock].amount * context.profit_taking_factor
                takes += 1
                order_target(stock, profit_taking_amount) 
    
    # Record leverage and number of positions held
    record(leverage=context.account.leverage, positions=len(context.portfolio.positions), t=takes)  
           
def monthly_rebalance(context,data):    
    
    if context.buy_stocks == False:
        current_year = get_datetime('US/Eastern').year
        context.this_months_list = context.safe
        for stock in context.portfolio.positions:  
            if data.can_trade(stock):
                if stock not in context.safe:
                    order_target(stock, 0)
#        if current_year > 2003:
        n = 0
        for stock in context.safe:
            if data.can_trade(stock): n += 1
        if n > 0:
            weight = 1.0/n
            for stock in context.safe:
                if data.can_trade(stock): 
                    order_target_percent(stock, weight)
    else:
        context.this_months_list = context.stock_list
        # used to calculate order weights
        positions = set()

        for stock in context.stock_list:
            positions.add(stock)
        for stock in context.portfolio.positions:    #TBD = why is this logged?
            positions.add(stock)

        weight = context.acc_leverage / len(context.stock_list)

        cssf1 = context.stock_factors.factor_1
        for stock in context.portfolio.positions:  
            if data.can_trade(stock):
#                if stock not in context.stock_list or cssf1[stock]<=1:
                if stock not in context.stock_list or cssf1[stock]<=0:
                    order_target(stock, 0)

        for stock in context.stock_list:  
            if stock in security_lists.leveraged_etf_list:
                continue
#            if context.stock_factors.factor_1[stock] > 1:
            if context.stock_factors.factor_1[stock] > 0:
                order_target_percent(stock, weight)  
                context.profit_target[stock] = data.current(stock, 'close')*1.25