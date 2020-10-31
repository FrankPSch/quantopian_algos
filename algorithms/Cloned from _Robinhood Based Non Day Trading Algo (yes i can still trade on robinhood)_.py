from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.filters.morningstar import IsPrimaryShare

import numpy as np #needed for NaN handling
import math #ceil and floor are useful for rounding

from itertools import cycle

def initialize(context):
    #set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.50))
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=0.80)) # TradeStation: $0.01 comission per share, $1 minimum comission per trade
    #set_slippage(slippage.FixedSlippage(spread=0.00))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_long_only()

    context.MaxCandidates=100 # was 100
    context.MaxBuyOrdersAtOnce=12 # was 30
    context.MyLeastPrice=4.00 # was 3
    context.MyMostPrice=20.00 # was 20
    context.MyFireSalePrice=context.MyLeastPrice
    context.MyFireSaleAge=15 # After 6 days, sell the stock either way

    # over simplistic tracking of position age
    context.age={}
    print len(context.portfolio.positions)

    # Rebalance every x minutes within trading day hours
    EveryThisManyMinutes=360
    TradingDayHours=6.5
    TradingDayMinutes=int(TradingDayHours*60)
    for minutez in xrange(
        1, 
        TradingDayMinutes, 
        EveryThisManyMinutes
    ):
        schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open(minutes=minutez))

    # Prevent excessive logging of canceled orders at market close.
    schedule_function(cancel_open_orders, date_rules.every_day(), time_rules.market_close(hours=0, minutes=1))

    # Record variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline(context)
    attach_pipeline(my_pipe, 'my_pipeline')

    # Define set of securities (collection of elements where no element is repeated).
    context.myFullStockSet = set() 

#---------------------------------------------------------------------------------------------------------------
class SidInList(CustomFilter):
    """
    Filter returns True for any SID included in parameter tuple passed at creation.
    Usage: my_filter = SidInList(sid_list=(23911, 46631))
    https://www.quantopian.com/posts/using-a-specific-list-of-securities-in-pipeline
    """    
    inputs = []
    window_length = 1
    params = ('sid_list',)

    def compute(self, today, assets, out, sid_list):
        out[:] = np.in1d(assets, sid_list)

    #How to create a new DataSet like USEquityPricing
    #https://groups.google.com/forum/#!topic/zipline/MwBgOFl0KYA
        
def make_pipeline(context):
    """
    Create our pipeline.
    https://www.quantopian.com/help#masking-factors
    A Pipeline is an object that represents computation we would like to perform every day.
    A freshly-constructed pipeline is empty, which means it doesn't yet know how to compute anything, and it won't produce any values if we ask for its outputs.
    Just constructing a new pipeline doesn't do anything by itself: we have to tell our algorithm to use the new pipeline, and we have to provide a name for the pipeline so that we can identify it later when we ask for results.
    The attach_pipeline() function from quantopian.algorithm accomplishes both of these tasks.
    """

    # User Input
    LowVar=context.MyLeastPrice # was 6
    HighVar=context.MyMostPrice # was 40
    log.info('\nAlgorithm initialized variables:\n context.MaxCandidates %s \n LowVar %s \n HighVar %s'
        % (context.MaxCandidates, LowVar, HighVar)
    )

    #set(['WSTG', 'AGTC', 'IPDN', 'ASTC', 'COGT', 'JOB', 'MYO', 'NVUS', 'YUME', 'OBLN', 'AWRE', 'CETX', 'ADMA', 'XBIO', 'NBEV', 'STDY', 'XBIT', 'TGEN', 'ZFGN', 'CHFS', 'AHC', 'YOGA', 'BWEN', 'NTRP', 'CDOR', 'AKTS', 'IVAC', 'WILC',
    #'FUV', 'LPTX', 'ZKIN', 'METC', 'LPCN', 'PFNX', 'SREV', 'DTEA', 'DXYN', 'TCS', 'JMBA', 'CCUR', 'INVE', 'XRM', 'UTI', 'EVOL', 'MAGS', 'AGFS', 'JAKK', 'AXSM', 'GLMD', 'BBOX', 'ATXI', 'TCON', 'YGYI', 'AQXP', 'FBIO', 'HTM', 'MNTX', 'RVLT', 'BTN', 'SSNT', 'PIXY', 'PLXP', 'DAVE', 'EDUC', 'CLRO', 'GEMP', 'ASV', 'AST', 'OVID', 'OMED', 'PME', 'OMEX', 'ATTO', 'QBAK', 'XELA', 'CDTX', 'FLGT', 'HWCC', 'DRAD', 'AZRX', 'BBW', 'CUI', 'CMFN', 'VIVE', 'CVO', 'SRTS', 'AIRG', 'HEBT', 'CLBS', 'HNRG', 'LEU', 'MDWD', 'SCKT', 'OXBR', 'ALQA', 'BCLI', 'CBMG', 'RTIX', 'CKPT', 'ERA', 'CSSE', 'BXC', 'XPLR', 'SPNE', 'SANW', 'FLKS', 'MGEN', 'ISSC', 'HAIR', 'NGVC', 'GECC', 'TZOO', 'TNDM', 'PCYG', 'NTWK', 'AUTO', 'SKIS', 'SMED', 'TACT', 'ORPN', 'BRQS', 'HPJ', 'RESN', 'ATOM', 'GNUS', 'MYND', 'INTX', 'PW', 'MXE', 'RVEN'

    # Always a good idea to set lookup date when referencing securities by symbol
    set_symbol_lookup_date('2017-10-01')
    
    # Custom filter to return only sids in the list
    my_sid_filter = SidInList(
        sid_list = (
            symbol('WSTG').sid, 
            symbol('AGTC').sid,
            symbol('IPDN').sid,
            symbol('ASTC').sid, 
            symbol('COGT').sid,
            symbol('JOB').sid,
            symbol('MYO').sid,
            symbol('NVUS').sid,
            symbol('YUME').sid,
            symbol('OBLN').sid,
            symbol('AWRE').sid,
            symbol('CETX').sid,
            symbol('ADMA').sid,
            symbol('XBIO').sid,
            symbol('NBEV').sid,
            symbol('STDY').sid,
            symbol('XBIT').sid,
            symbol('TGEN').sid,
            symbol('ZFGN').sid,
            symbol('CHFS').sid,
            symbol('AHC').sid,
            symbol('YOGA').sid,
            symbol('BWEN').sid,
            symbol('NTRP').sid,
            symbol('CDOR').sid,
            symbol('AKTS').sid,
            symbol('IVAC').sid,
            symbol('WILC').sid,
            symbol('FUV').sid, 
            symbol('LPTX').sid,
            symbol('ZKIN').sid,
                )
            )

    # Filter for primary share equities. IsPrimaryShare is a built-in filter.
    primary_share = IsPrimaryShare()

    # Equities listed as common stock (as opposed to, say, preferred stock).
    # 'ST00000001' indicates common stock.
    common_stock = morningstar.share_class_reference.security_type.latest.eq('ST00000001')

    # Non-depositary receipts. Recall that the ~ operator inverts filters,
    # turning Trues into Falses and vice versa
    not_depositary = ~morningstar.share_class_reference.is_depositary_receipt.latest

    # Equities not trading over-the-counter.
    not_otc = ~morningstar.share_class_reference.exchange_id.latest.startswith('OTC')

    # Not when-issued equities.
    not_wi = ~morningstar.share_class_reference.symbol.latest.endswith('.WI')

    # Equities without LP in their name, .matches does a match using a regular
    # expression
    not_lp_name = ~morningstar.company_reference.standard_name.latest.matches('.* L[. ]?P.?$')

    # Equities with a null value in the limited_partnership Morningstar
    # fundamental field.
    not_lp_balance_sheet = morningstar.balance_sheet.limited_partnership.latest.isnull()

    # Equities whose most recent Morningstar market cap is not null have
    # fundamental data and therefore are not ETFs.
    have_market_cap = morningstar.valuation.market_cap.latest.notnull()

    # At least a certain price
    price = USEquityPricing.close.latest
    AtLeastPrice   = (price >= context.MyLeastPrice)
    AtMostPrice    = (price <= context.MyMostPrice)

    # Filter for stocks that pass all of our previous filters.
    tradeable_stocks = (
        # my_sid_filter &
        primary_share
        & common_stock
        & not_depositary
        & not_otc
        & not_wi
        & not_lp_name
        & not_lp_balance_sheet
        & have_market_cap
        & AtLeastPrice
        & AtMostPrice
    )

    ### Filter between low and high dollar amount
    # Dollar volume filter.
    base_universe = AverageDollarVolume(
        window_length=20,
        mask=tradeable_stocks
    ).percentile_between(LowVar, HighVar)
    #print base_universe

    ### Filter for most undervaluated stocks (stocks_worst) where ShortAvg/LongAvg is smallest
    # Short close price average.
    ShortAvg = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=3,
        mask=base_universe
    )

    # Long close price average.
    LongAvg = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=45,
        mask=base_universe
    )

    #percent_difference = (ShortAvg - LongAvg) / LongAvg
    percent_difference = ShortAvg/LongAvg

    # Filter to select securities to long.
    stocks_worst = percent_difference.bottom(context.MaxCandidates)
    securities_to_trade = (stocks_worst)

    # Create a pipeline object.
    # https://www.quantopian.com/posts/masking-pipeline-factors-new-feature
    pipe = Pipeline(
        columns={
            'stocks_worst': stocks_worst
        },
        screen=(securities_to_trade),
    )
    return pipe

#---------------------------------------------------------------------------------------------------------------
def my_compute_weights(context):
    """
    Compute ordering weights.
    """
    # Compute even target weights for our long positions and short positions.
    div = len(context.stocks_worst)
    if div != 0: # FSC
        stocks_worst_weight = 1.00/len(context.stocks_worst)
    else: # FSC
        stocks_worst_weight = 1.00 # FSC

    return stocks_worst_weight

#---------------------------------------------------------------------------------------------------------------
def before_trading_start(context, data):
    # Gets our pipeline output every day.
    # ask for the results of the pipeline attached to our algorithm https://www.quantopian.com/help#masking-factors
    context.output = pipeline_output('my_pipeline')

    context.stocks_worst = context.output[context.output['stocks_worst']].index.tolist()
    context.stocks_worst_weight = my_compute_weights(context)
    context.MyCandidate = cycle(context.stocks_worst)
    
    context.LowestPrice=context.MyLeastPrice #reset to MyLeastPrice at beginning of day
    
    log.info("len(context.portfolio.positions): " + str(len(context.portfolio.positions)))
    for stock in context.portfolio.positions:
        #log.info(stock.symbol)
        # collect all portfolio symbols
        context.myFullStockSet.add(str(stock.symbol))
        # find LowestPrice
        CurrPrice = float(data.current([stock], 'price'))
        if CurrPrice<context.LowestPrice:
            context.LowestPrice = CurrPrice
        # before_trading_start, increase age by 1
        if stock in context.age:
            context.age[stock] += 1
        else:
            context.age[stock] = 1
    log.info(context.myFullStockSet)
            
    for stock in context.age:
        if stock not in context.portfolio.positions:
            context.age[stock] = 0
        message = 'stock.symbol: {symbol}  :  age: {age}'
        log.info(message.format(symbol=stock.symbol, age=context.age[stock]))
    pass

    #https://www.quantopian.com/posts/calling-stocks-from-pipeline-output
    #https://www.quantopian.com/posts/how-to-get-the-values-from-the-columns-of-a-pipeline
    #context.output= pipeline_output('my_pipeline')
    #for stock in context.output.index:  
    #    log.info(stock.symbol)
    #log.info(context.output.axes)
    #log.info(context.output.index)
    #log.info(context.output.head(3))
         
#---------------------------------------------------------------------------------------------------------------
def my_rebalance(context, data):
    BuyFactor=.99
    SellFactor=1.01
    cash=context.portfolio.cash

    # Cancelling all open BUY orders
    cancel_open_buy_orders(context, data)

    # Place buy order with leverage of x
    WeightThisBuyOrder=float(1.00/context.MaxBuyOrdersAtOnce) # weight every buy order equally
    for ThisBuyOrder in range(context.MaxBuyOrdersAtOnce): # TODO should this be range(context.MaxBuyOrdersAtOnce-1)
        stock = context.MyCandidate.next()
        PH = data.history([stock], 'price', 20, '1d')
        PH_Avg = float(PH.mean())
        CurrPrice = float(data.current([stock], 'price'))
        if np.isnan(CurrPrice):
            pass # probably best to wait until nan goes away
        else:
            if CurrPrice > float(1.25*PH_Avg): # if current price is 25% greater than the average of the last 20 days
                BuyPrice=float(CurrPrice) # buy at 100% of current price
            else:
                BuyPrice=float(CurrPrice*BuyFactor) # buy at 99% of current price
            # Buy the stock
            BuyPrice=float(make_div_by_05(BuyPrice, buy=True))
            StockShares = int(WeightThisBuyOrder*cash/BuyPrice)
            order(stock, StockShares,
                style=LimitOrder(BuyPrice)
                )

    # Order sell at profit target in hope that somebody actually buys it
    for stock in context.portfolio.positions:
        if not get_open_orders(stock): # if there are curently no open orders (BUY or SELL)
            StockShares = context.portfolio.positions[stock].amount
            CurrPrice = float(data.current([stock], 'price'))
            CostBasis = float(context.portfolio.positions[stock].cost_basis)
            SellPrice = float(make_div_by_05(CostBasis*SellFactor, buy=False))
            
            
            if np.isnan(SellPrice) :
                pass # probably best to wait until nan goes away
            elif (stock in context.age and context.age[stock] == 1) :
                pass
            elif (
                stock in context.age 
                and context.MyFireSaleAge<=context.age[stock] 
                and (
                    context.MyFireSalePrice>CurrPrice
                    or CostBasis>CurrPrice
                )
            ):
                if (stock in context.age and context.age[stock] < 2) :
                    pass
                elif stock not in context.age:
                    context.age[stock] = 1
                else:
                    SellPrice = float(make_div_by_05(.95*CurrPrice, buy=False))
                    order(stock, -StockShares,
                        style=LimitOrder(SellPrice)
                    )
            else:
                if (stock in context.age and context.age[stock] < 2) :
                    pass
                elif stock not in context.age:
                    context.age[stock] = 1
                else:
                
                    order(stock, -StockShares,
                        style=LimitOrder(SellPrice)
                    )

#---------------------------------------------------------------------------------------------------------------

#if cents not divisible by .05, round down if buy, round up if sell
def make_div_by_05(s, buy=False):
    s *= 20.00
    s =  math.floor(s) if buy else math.ceil(s)
    s /= 20.00
    return s

def my_record_vars(context, data):
    """
    Record variables at the end of each day.
    """

    # Record our variables.
    #https://www.quantopian.com/posts/quantopian-open-example-algorithm-to-control-leverage
    record(leverage=context.account.leverage) # Track the algorithm's leverage, and put it on the custom graph
    record(positions=len(context.portfolio.positions))
    if 0<len(context.age):
        MaxAge=context.age[max(context.age.keys(), key=(lambda k: context.age[k]))]
        print MaxAge
        record(MaxAge=MaxAge)
    record(LowestPrice=context.LowestPrice)

def log_open_order(StockToLog):
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        if stock == StockToLog:
            for order in orders:
                message = 'Found open order for {amount} shares in {stock}'
                log.info(message.format(amount=order.amount, stock=stock))

def log_open_orders():
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        for order in orders:
            message = 'Found open order for {amount} shares in {stock}'
            log.info(message.format(amount=order.amount, stock=stock))

def cancel_open_buy_orders(context, data):
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        for order in orders:
            #message = 'Canceling order of {amount} shares in {stock}'
            #log.info(message.format(amount=order.amount, stock=stock))
            if 0<order.amount: #it is a buy order
                cancel_order(order)

def cancel_open_orders(context, data):
    oo = get_open_orders()
    if len(oo) == 0:
        return
    for stock, orders in oo.iteritems():
        for order in orders:
            #message = 'Canceling order of {amount} shares in {stock}'
            #log.info(message.format(amount=order.amount, stock=stock))
            cancel_order(order)

# This is the every minute stuff
def handle_data(context, data):
    pass
