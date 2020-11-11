import numpy as np

from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.factors import CustomFactor, AverageDollarVolume
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.filters.morningstar import IsPrimaryShare

# Premium version available at
# https://www.quantopian.com/data/zacks/earnings_surprises
from quantopian.pipeline.data.zacks import EarningsSurprises

# Premium version available at
# https://www.quantopian.com/data/eventvestor/earnings_calendar
from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysSincePreviousEarnings as days_since_earnings
)

# Premium version availabe at
# https://www.quantopian.com/data/accern/alphaone
# from quantopian.pipeline.data.accern import alphaone as alphaone
from quantopian.pipeline.data.accern import alphaone_free as alphaone

def make_pipeline(context):
    # Create our pipeline  
    pipe = Pipeline()  

    # Instantiating our factors  
    factor = EarningsSurprises.eps_pct_diff_surp.latest
    # Time of day that the earnings report happened
    # BTO - before the open, DTM - during the market,
    # AMC - after market close
    time_of_day = EarningsSurprises.act_rpt_code.latest

    # Filter down to stocks in the top/bottom according to
    # the earnings surprise
    longs = (factor >= context.min_surprise) & (factor <= context.max_surprise)
    shorts = (factor <= -context.min_surprise) & (factor >= -context.max_surprise)

    # Set our pipeline screens  
    # Filter down stocks using sentiment  
    article_sentiment = alphaone.article_sentiment.latest
    top_universe = universe_filters() & longs & article_sentiment.notnan() \
        & (article_sentiment > .30)
    bottom_universe = universe_filters() & shorts & article_sentiment.notnan() \
        & (article_sentiment < -.50)

    # Add long/shorts to the pipeline  
    pipe.add(top_universe, "longs")
    pipe.add(bottom_universe, "shorts")
    pipe.add(days_since_earnings(), 'pe')
    pipe.add(time_of_day, 'time_of_day')
    pipe.set_screen(factor.notnan())
    return pipe  
        
def initialize(context):
    #: Set commissions and slippage to 0 to determine pure alpha
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))

    #: Declaring the days to hold, change this to what you want
    context.days_to_hold = 3
    #: Declares which stocks we currently held and how many days we've held them dict[stock:days_held]
    context.stocks_held = {}

    #: Declares the minimum magnitude of percent surprise
    context.min_surprise = .00
    context.max_surprise = .05

    #: OPTIONAL - Initialize our Hedge
    # See order_positions for hedging logic
    # context.spy = sid(8554)
    
    # Make our pipeline
    attach_pipeline(make_pipeline(context), 'earnings')

    
    # Log our positions at 10:00AM
    schedule_function(func=log_positions,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))
    # Order our positions
    schedule_function(func=order_positions,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open())

def before_trading_start(context, data):
    # Screen for securities that only have an earnings release
    # 1 business day previous and separate out the earnings surprises into
    # positive and negative 
    results = pipeline_output('earnings')
    results = results[results['pe'] == 1]
    assets_in_universe = results.index
    context.positive_surprise = assets_in_universe[results.longs]
    context.negative_surprise = assets_in_universe[results.shorts]

def log_positions(context, data):
    #: Get all positions  
    if len(context.portfolio.positions) > 0:  
        all_positions = "Current positions for %s : " % (str(get_datetime()))  
        for pos in context.portfolio.positions:  
            if context.portfolio.positions[pos].amount != 0:  
                all_positions += "%s at %s shares, " % (pos.symbol, context.portfolio.positions[pos].amount)  
        log.info(all_positions)  
        
def order_positions(context, data):
    """
    Main ordering conditions to always order an equal percentage in each position
    so it does a rolling rebalance by looking at the stocks to order today and the stocks
    we currently hold in our portfolio.
    """
    port = context.portfolio.positions
    record(leverage=context.account.leverage)

    # Check our positions for loss or profit and exit if necessary
    check_positions_for_loss_or_profit(context, data)
    
    # Check if we've exited our positions and if we haven't, exit the remaining securities
    # that we have left
    for security in port:  
        if data.can_trade(security):  
            if context.stocks_held.get(security) is not None:  
                context.stocks_held[security] += 1  
                if context.stocks_held[security] >= context.days_to_hold:  
                    order_target_percent(security, 0)  
                    del context.stocks_held[security]  
            # If we've deleted it but it still hasn't been exited. Try exiting again  
            else:  
                log.info("Haven't yet exited %s, ordering again" % security.symbol)  
                order_target_percent(security, 0)  

    # Check our current positions
    current_positive_pos = [pos for pos in port if (port[pos].amount > 0 and pos in context.stocks_held)]
    current_negative_pos = [pos for pos in port if (port[pos].amount < 0 and pos in context.stocks_held)]
    negative_stocks = context.negative_surprise.tolist() + current_negative_pos
    positive_stocks = context.positive_surprise.tolist() + current_positive_pos
    
    # Rebalance our negative surprise securities (existing + new)
    for security in negative_stocks:
        can_trade = context.stocks_held.get(security) <= context.days_to_hold or \
                    context.stocks_held.get(security) is None
        if data.can_trade(security) and can_trade:
            order_target_percent(security, -1.0 / len(negative_stocks))
            if context.stocks_held.get(security) is None:
                context.stocks_held[security] = 0

    # Rebalance our positive surprise securities (existing + new)                
    for security in positive_stocks:
        can_trade = context.stocks_held.get(security) <= context.days_to_hold or \
                    context.stocks_held.get(security) is None
        if data.can_trade(security) and can_trade:
            order_target_percent(security, 1.0 / len(positive_stocks))
            if context.stocks_held.get(security) is None:
                context.stocks_held[security] = 0

    #: Get the total amount ordered for the day
    # amount_ordered = 0 
    # for order in get_open_orders():
    #     for oo in get_open_orders()[order]:
    #         amount_ordered += oo.amount * data.current(oo.sid, 'price')

    #: Order our hedge
    # order_target_value(context.spy, -amount_ordered)
    # context.stocks_held[context.spy] = 0
    # log.info("We currently have a net order of $%0.2f and will hedge with SPY by ordering $%0.2f" % (amount_ordered, -amount_ordered))
    
def check_positions_for_loss_or_profit(context, data):
    # Sell our positions on longs/shorts for profit or loss
    for security in context.portfolio.positions:
        is_stock_held = context.stocks_held.get(security) >= 0
        if data.can_trade(security) and is_stock_held and not get_open_orders(security):
            current_position = context.portfolio.positions[security].amount  
            cost_basis = context.portfolio.positions[security].cost_basis  
            price = data.current(security, 'price')
            # On Long & Profit
            if price >= cost_basis * 1.10 and current_position > 0:  
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold Long for Profit')  
                del context.stocks_held[security]  
            # On Short & Profit
            if price <= cost_basis* 0.90 and current_position < 0:
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold Short for Profit')  
                del context.stocks_held[security]
            # On Long & Loss
            if price <= cost_basis * 0.90 and current_position > 0:  
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold Long for Loss')  
                del context.stocks_held[security]  
            # On Short & Loss
            if price >= cost_basis * 1.10 and current_position < 0:  
                order_target_percent(security, 0)  
                log.info( str(security) + ' Sold Short for Loss')  
                del context.stocks_held[security]  
                
# Constants that need to be global
COMMON_STOCK= 'ST00000001'

SECTOR_NAMES = {
 101: 'Basic Materials',
 102: 'Consumer Cyclical',
 103: 'Financial Services',
 104: 'Real Estate',
 205: 'Consumer Defensive',
 206: 'Healthcare',
 207: 'Utilities',
 308: 'Communication Services',
 309: 'Energy',
 310: 'Industrials',
 311: 'Technology' ,
}

# Average Dollar Volume without nanmean, so that recent IPOs are truly removed
class ADV_adj(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 252
    
    def compute(self, today, assets, out, close, volume):
        close[np.isnan(close)] = 0
        out[:] = np.mean(close * volume, 0)
                
def universe_filters():
    """
    Create a Pipeline producing Filters implementing common acceptance criteria.
    
    Returns
    -------
    zipline.Filter
        Filter to control tradeablility
    """

    # Equities with an average daily volume greater than 750000.
    high_volume = (AverageDollarVolume(window_length=252) > 750000)
    
    # Not Misc. sector:
    sector_check = Sector().notnull()
    
    # Equities that morningstar lists as primary shares.
    # NOTE: This will return False for stocks not in the morningstar database.
    primary_share = IsPrimaryShare()
    
    # Equities for which morningstar's most recent Market Cap value is above $300m.
    have_market_cap = mstar.valuation.market_cap.latest > 300000000
    
    # Equities not listed as depositary receipts by morningstar.
    # Note the inversion operator, `~`, at the start of the expression.
    not_depositary = ~mstar.share_class_reference.is_depositary_receipt.latest
    
    # Equities that listed as common stock (as opposed to, say, preferred stock).
    # This is our first string column. The .eq method used here produces a Filter returning
    # True for all asset/date pairs where security_type produced a value of 'ST00000001'.
    common_stock = mstar.share_class_reference.security_type.latest.eq(COMMON_STOCK)
    
    # Equities whose exchange id does not start with OTC (Over The Counter).
    # startswith() is a new method available only on string-dtype Classifiers.
    # It returns a Filter.
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    
    # Equities whose symbol (according to morningstar) ends with .WI
    # This generally indicates a "When Issued" offering.
    # endswith() works similarly to startswith().
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    
    # Equities whose company name ends with 'LP' or a similar string.
    # The .matches() method uses the standard library `re` module to match
    # against a regular expression.
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    
    # Equities with a null entry for the balance_sheet.limited_partnership field.
    # This is an alternative way of checking for LPs.
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    
    # Highly liquid assets only. Also eliminates IPOs in the past 12 months
    # Use new average dollar volume so that unrecorded days are given value 0
    # and not skipped over
    # S&P Criterion
    liquid = ADV_adj() > 250000
    
    # Add logic when global markets supported
    # S&P Criterion
    domicile = True
    
    # Keep it to liquid securities
    ranked_liquid = ADV_adj().rank(ascending=False) < 1500
    
    universe_filter = (high_volume & primary_share & have_market_cap & not_depositary &
                      common_stock & not_otc & not_wi & not_lp_name & not_lp_balance_sheet &
                    liquid & domicile & sector_check & liquid & ranked_liquid)
    
    return universe_filter
