import pandas as pd
import numpy as np
import quantopian.algorithm as algo
import quantopian.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import builtin, Fundamentals, psychsignal
from quantopian.pipeline.factors import AverageDollarVolume, SimpleBeta
from quantopian.pipeline.factors.fundamentals import MarketCap
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.experimental import QTradableStocksUS, risk_loading_pipeline

from quantopian.pipeline.factors import CustomFactor, RollingPearsonOfReturns, RollingSpearmanOfReturns
# Algorithm Parameters
# --------------------
UNIVERSE_SIZE = 1000
LIQUIDITY_LOOKBACK_LENGTH = 100

MINUTES_AFTER_OPEN_TO_TRADE = 60

MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.01  # 1%
MAX_LONG_POSITION_SIZE = 0.01   # 1%

# 5 BPS Fixed Slippage Model    
class FixedBasisPointsSlippage(slippage.SlippageModel):
    """
    Model slippage as a fixed percentage of fill price. Executes the full
    order immediately.
    Orders to buy will be filled at: `price + (price * basis_points * 0.0001)`.
    Orders to sell will be filled at:
        `price - (price * basis_points * 0.0001)`.
    Parameters
    ----------
    basis_points : float, optional
        Number of basis points of slippage to apply on each execution.
    volume_limit : float, optional
        fraction of the trading volume that can be filled each minute.
    """
    def __init__(self, basis_points=5, volume_limit=0.1):
        slippage.SlippageModel.__init__(self)
        self.basis_points = basis_points
        self.percentage = self.basis_points / 10000.0
        self.volume_limit = volume_limit

    def process_order(self, data, order):

        price = data.current(order.asset, "close")
        
        volume = data.current(order.asset, "volume")

        max_volume = self.volume_limit * volume

        remaining_volume = max_volume - self.volume_for_bar
        if remaining_volume < 1:
            # We can't fill any more transactions.
            raise LiquidityExceeded()

        # The current order amount will be the min of the
        # volume available in the bar or the open amount.
        cur_volume = int(min(remaining_volume, abs(order.open_amount)))

        if cur_volume < 1:
            return None, None

        return (
            price + price * (self.percentage * order.direction),
            cur_volume * order.direction
        )
    

def initialize(context):
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))  
    #set_slippage(slippage.VolumeShareSlippage(volume_limit=1, price_impact=0))  
    #set_slippage(FixedBasisPointsSlippage())
    #set_slippage(slippage.FixedBasisPointsSlippage())
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

    #Fetch SP500 DNN predictions
    fetch_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTHhEDVJC3WLC2tl_7JcJeW_X7okfxm5v6Dpzk4gRFpEQDaK_Br5Xx1dP5ZZ1z0eInt2-nueuHRc9_3/pub?output=csv', 
               date_column = 'Date',
               date_format = '%m/%d/%y') 
    context.stock = symbol('SPY')
    
    # Universe Selection
    # ------------------
    base_universe = QTradableStocksUS()

    # From what remains, each month, take the top UNIVERSE_SIZE stocks by average dollar
    # volume traded.
    monthly_top_volume = (
        AverageDollarVolume(window_length=LIQUIDITY_LOOKBACK_LENGTH)
        .top(UNIVERSE_SIZE, mask=base_universe)
        .downsample('week_start')
    )
    # The final universe is the monthly top volume &-ed with the original base universe.
    # &-ing these is necessary because the top volume universe is calculated at the start 
    # of each month, and an asset might fall out of the base universe during that month.
    universe = monthly_top_volume & base_universe
    
    # Market Beta Factor
    # ------------------
    stock_beta = SimpleBeta(
                    target=sid(8554),
                    regression_length=21,
                    )    
    # Alpha Generation
    # ----------------
    # Compute Z-scores of free cash flow yield and earnings yield. 
    # Both of these are fundamental value measures.
    #fcf_zscore = Fundamentals.fcf_yield.latest.zscore(mask=universe)
    #yield_zscore = Fundamentals.earning_yield.latest.zscore(mask=universe)
    #sentiment_zscore = psychsignal.stocktwits.bull_minus_bear.latest.zscore(mask=universe)
    b = stock_beta.rank(mask=universe)
    # Alpha Combination
    # -----------------
    # Assign every asset a combined rank and center the values at 0.
    # For UNIVERSE_SIZE=500, the range of values should be roughly -250 to 250.
    #combined_alpha = (fcf_zscore + yield_zscore + sentiment_zscore).rank().demean()
    alpha = b.top(500)


    # Schedule Tasks
    # --------------
    # Create and register a pipeline computing our combined alpha and a sector
    # code for every stock in our universe. We'll use these values in our 
    # optimization below.
    pipe = Pipeline(
        columns={
            'alpha': alpha,
        },
        # combined_alpha will be NaN for all stocks not in our universe,
        # but we also want to make sure that we have a sector code for everything
        # we trade.
        screen=alpha,
    )
    algo.attach_pipeline(pipe, 'pipe')

    # Schedule a function, 'do_portfolio_construction', to run twice a week
    # ten minutes after market open.
    algo.schedule_function(
        do_portfolio_construction,
        date_rule=algo.date_rules.every_day(),
        time_rule=algo.time_rules.market_open(minutes=MINUTES_AFTER_OPEN_TO_TRADE),
        half_days=False,
    )


def before_trading_start(context, data):
    # Call pipeline_output in before_trading_start so that pipeline
    # computations happen in the 5 minute timeout of BTS instead of the 1
    # minute timeout of handle_data/scheduled functions.
    context.pipeline_data = algo.pipeline_output('pipe')
    context.Perfect = data.current(context.stock, 'Perfect') 

# Portfolio Construction
# ----------------------
def do_portfolio_construction(context, data):
    pipeline_data = context.pipeline_data
    perf = context.Perfect
    # Objective
    # ---------
    # For our objective, we simply use our naive ranks as an alpha coefficient
    # and try to maximize that alpha.
    # 
    # This is a **very** naive model. Since our alphas are so widely spread out,
    # we should expect to always allocate the maximum amount of long/short
    # capital to assets with high/low ranks.
    #
    # A more sophisticated model would apply some re-scaling here to try to generate
    # more meaningful predictions of future returns.
    objective = opt.MaximizeAlpha(pipeline_data.alpha * perf)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossExposure(MAX_GROSS_LEVERAGE)
    
    # Constrain individual position size to no more than a fixed percentage 
    # of our portfolio. Because our alphas are so widely distributed, we 
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
        ],
    )