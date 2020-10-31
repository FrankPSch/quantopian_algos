import pandas as pd
import numpy as np
import quantopian.algorithm as algo
import quantopian.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import builtin, Fundamentals, psychsignal
from quantopian.pipeline.factors import AverageDollarVolume, SimpleBeta, Returns, Factor
from quantopian.pipeline.factors.fundamentals import MarketCap
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.experimental import QTradableStocksUS, risk_loading_pipeline

from quantopian.pipeline.factors import CustomFactor, RollingPearsonOfReturns, RollingSpearmanOfReturns
# Algorithm Parameters
# --------------------
UNIVERSE_SIZE = 1000
LIQUIDITY_LOOKBACK_LENGTH = 100

MINUTES_AFTER_OPEN_TO_TRADE = 1

MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.01  # 1%
MAX_LONG_POSITION_SIZE = 0.01   # 1%

def vectorized_beta(spy, assets):
    """Calculate beta between every column of ``assets`` and ``spy``.
    
    Parameters
    ----------
    spy : np.array
        An (n x 1) array of returns for SPY.
    assets : np.array
        An (n x m) array of returns for m assets.
    """
    assert len(spy.shape) == 2 and spy.shape[1] == 1, "Expected a column vector for spy."

    asset_residuals = assets - assets.mean(axis=0)
    spy_residuals = spy - spy.mean()

    covariances = (asset_residuals * spy_residuals).sum(axis=0)
    spy_variance = (spy_residuals ** 2).sum()
    return covariances / spy_variance

daily_returns = Returns(window_length=2)
daily_log_returns = daily_returns.log1p()
SPY_asset = sid(8554) #symbols('SPY')
class MyBeta(CustomFactor):
    # Get daily returns for every asset in existence, plus the daily returns for just SPY
    # as a column vector.
    inputs = [daily_log_returns, daily_log_returns[SPY_asset]]
    # Set a default window length of 2 years.
    window_length = 126
    
    def compute(self, today, assets, out, all_returns, spy_returns):
        out[:] = vectorized_beta(spy_returns, all_returns)



def initialize(context):
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    #Fetch SP500 DNN predictions
    fetch_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQAtxWEdzLnzdmU6zfTxk7RYgOIieg27z9em6sSd9Mm2mBpK46CXOY7EFrYjnZ3Vy9L8vYxBSICyEFz/pub?output=csv', 
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
    stock_beta = MyBeta() #SimpleBeta(
                    #target=sid(8554),
                    #regression_length=21,
                    #)    
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
    alpha = b.top(100)


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

    schedule_function(func=record_vars,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB

def before_trading_start(context, data):
    # Call pipeline_output in before_trading_start so that pipeline
    # computations happen in the 5 minute timeout of BTS instead of the 1
    # minute timeout of handle_data/scheduled functions.
    context.pipeline_data = algo.pipeline_output('pipe')
    context.Predict = data.current(context.stock, 'Predicted') 

# Portfolio Construction
# ----------------------
def do_portfolio_construction(context, data):
    pipeline_data = context.pipeline_data
    perf = context.Predict
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
    
def record_vars(context, data):
    """
    This function is called at the end of each day and plots certain variables.
    """

    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1

    # Record and plot the leverage of our portfolio over time as well as the
    # number of long and short positions. Even in minute mode, only the end-of-day
    # leverage is plotted.
    record(leverage = context.account.leverage, long_count=longs, short_count=shorts)
    
def handle_data(context, data):    
    record(Predict = data.current(context.stock, 'Predicted'))