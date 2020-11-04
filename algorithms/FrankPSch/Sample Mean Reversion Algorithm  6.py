"""
This is a sample mean-reversion algorithm on Quantopian for you to test and adapt.
This example uses a dynamic stock selector called Pipeline to select stocks to trade. 
It orders stocks from the QTradableStocksUS, which is a dynamic universe of over 2000
liquid stocks. 
(https://www.quantopian.com/help#quantopian_pipeline_filters_QTradableStocksUS)

Algorithm investment thesis:
Top-performing stocks from last week will do worse this week, and vice-versa.

Every Monday, we rank stocks from the QTradableStocksUS based on their previous 5-day 
returns. We enter long positions in the 10% of stocks with the WORST returns over the 
past 5 days. We enter short positions in the 10% of stocks with the BEST returns over 
the past 5 days.
"""

# Import the libraries we will use here.
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.filters import QTradableStocksUS

# Define static variables that can be accessed in the rest of the algorithm.

# Controls the maximum leverage of the algorithm. A value of 1.0 means the algorithm
# should spend no more than its starting capital (doesn't borrow money).
MAX_GROSS_EXPOSURE = 1.0

# Controls the maximum percentage of the portfolio that can be invested in any one
# security. A value of 0.02 means the portfolio will invest a maximum of 2% of its
# portfolio in any one stock.
MAX_POSITION_CONCENTRATION = 0.001

# Controls the lookback window length of the Returns factor used by this algorithm
# to rank stocks.
RETURNS_LOOKBACK_DAYS = 5


def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.
    Use this function for initializing state or other bookkeeping.
    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in 
        your algorithm. context is automatically passed to initialize, 
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information 
        about current positions.
    """
    # Rebalance on the first trading day of each week at 11AM.
    algo.schedule_function(
        rebalance,
        algo.date_rules.week_start(days_offset=0),
        algo.time_rules.market_open(hours=1, minutes=30)
    )

    # Create and attach our pipeline (dynamic stock selector), defined below.
    algo.attach_pipeline(make_pipeline(context), 'mean_reversion_example')


def make_pipeline(context):
    """
    A function that creates and returns our pipeline.
    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation. In particular, this function can be
    copy/pasted into research and run by itself.
    Parameters
    -------
    context : AlgorithmContext
        See description above.
    Returns
    -------
    pipe : Pipeline
        Represents computation we would like to perform on the assets that make
        it through the pipeline screen.
    """

    # Filter for stocks in the QTradableStocksUS universe. For more detail, see 
    # the documentation:
    # https://www.quantopian.com/help#quantopian_pipeline_filters_QTradableStocksUS
    universe = QTradableStocksUS()
    
    # Create a Returns factor with a 5-day lookback window for all securities
    # in our QTradableStocksUS Filter.
    recent_returns = Returns(
        window_length=RETURNS_LOOKBACK_DAYS, 
        mask=universe
    )
    
    # Turn our recent_returns factor into a z-score factor to normalize the results.
    recent_returns_zscore = recent_returns.zscore()

    # Define high and low returns filters to be the bottom 10% and top 10% of
    # securities in the QTradableStocksUS.
    low_returns = recent_returns_zscore.percentile_between(0,10)
    high_returns = recent_returns_zscore.percentile_between(90,100)

    # Add a filter to the pipeline such that only high-return and low-return
    # securities are kept.
    securities_to_trade = (low_returns | high_returns)

    # Create a pipeline object to computes the recent_returns_zscore for securities
    # in the top 10% and bottom 10% (ranked by recent_returns_zscore) every day.
    pipe = Pipeline(
        columns={
            'recent_returns_zscore': recent_returns_zscore
        },
        screen=securities_to_trade
    )

    return pipe

def before_trading_start(context, data):
    """
    Optional core function called automatically before the open of each market day.
    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        An object that provides methods to get price and volume data, check
        whether a security exists, and check the last time a security traded.
    """

    # pipeline_output returns a pandas DataFrame with the results of our factors
    # and filters.
    context.output = algo.pipeline_output('mean_reversion_example')

    # Sets the list of securities we want to long as the securities with a 'True'
    # value in the low_returns column.
    context.recent_returns_zscore = context.output['recent_returns_zscore']


def rebalance(context, data):
    """
    A function scheduled to run once at the start of the week an hour 
    and half after market open to order an optimal portfolio of assets.
    Parameters
    ----------
    context : AlgorithmContext
        See description above.
    data : BarData
        See description above.
    """

    # Each day, we will enter and exit positions by defining a portfolio optimization
    # problem. To do that, we need to set an objective for our portfolio as well
    # as a series of constraints. You can learn more about portfolio optimization here:
    # https://www.quantopian.com/help#optimize-title
    
    # Our objective is to maximize alpha, where 'alpha' is defined by the negative of
    # recent_returns_zscore factor.
    objective = opt.MaximizeAlpha(-context.recent_returns_zscore)
    
    # We want to constrain our portfolio to invest a maximum total amount of money
    # (defined by MAX_GROSS_EXPOSURE).
    max_gross_exposure = opt.MaxGrossExposure(MAX_GROSS_EXPOSURE)
    
    # We want to constrain our portfolio to invest a limited amount in any one 
    # position. To do this, we constrain the position to be between +/- 
    # MAX_POSITION_CONCENTRATION (on Quantopian, a negative weight corresponds to 
    # a short position).
    max_position_concentration = opt.PositionConcentration.with_equal_bounds(
        -MAX_POSITION_CONCENTRATION,
        MAX_POSITION_CONCENTRATION
    )
    
    # We want to constraint our portfolio to be dollar neutral (equal amount invested in
    # long and short positions).
    dollar_neutral = opt.DollarNeutral()
    
    # Stores all of our constraints in a list.
    constraints = [
        max_gross_exposure,
        max_position_concentration,
        dollar_neutral,
    ]

    algo.order_optimal_portfolio(objective, constraints)
