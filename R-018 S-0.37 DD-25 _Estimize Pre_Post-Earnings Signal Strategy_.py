import numpy as np
import re
import pandas as pd

from datetime import timedelta
from pandas import isnull
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, CustomFactor
from quantopian.pipeline.factors.morningstar import MarketCap
from quantopian.pipeline.filters.morningstar import default_us_equity_universe_mask, Q500US, Q1500US, Q3000US

# For testing, use `signal.sample.csv` as the file in the URL
# For yearly data use df2012.csv, df2013.csv, df2014.csv, df2015.csv, df2016.csv
# For full, six year data use signal-2017q2.csv
SIGNAL_URL = 'https://s3.amazonaws.com/com.estimize.production.data/quantopian/df2016.csv'
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'
TIMEZONE = 'US/Eastern'
POST_ONLY = False
MIN_STOCKS = 10
PERCENTILE = 0.10 # If None then MIN_STOCKS is used
DEBUG = True

def initialize(context):
    context.aapl_signals = None
    
    set_slippage(IdealSlippage())
    set_commission(commission.PerTrade(cost=0.00))  
    
    attach_pipeline(make_pipeline(), name="pipeline")
    
    # Fetch and process the signal data csv
    fetch_csv(
        SIGNAL_URL,
        pre_func=pre_func,
        post_func=post_func,
        date_column='date',
        date_format=DATE_FORMAT
    )
    
    # Schedule morning trading
    schedule_function(
        func=trade,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open()
    )
    
    # Schedule afternoon trading
    schedule_function(
        func=trade,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close(minutes=1)
    )
    
    # Schedule function to print AAPL signal results
    schedule_function(
        func=print_results,
        date_rule=date_rules.month_end(),
        time_rule=time_rules.market_close()
    )
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
def pre_func(df):
    if POST_ONLY:
        df = df[df['type'] == 'post']
    
    # Drop unused columns and rename ticker -> symbol
    df.drop(['cusip', 'reports_at', 'type'], axis=1, inplace=True)
    df.rename(columns={'ticker': 'symbol', 'as_of': 'date'}, inplace=True)
    
    # Fix invalid tickers
    fix_missing_symbols(df)
    
    if DEBUG:
        aapl = df[df['symbol'] == 'AAPL']
        log.info(aapl.head(10))
    
    return df
    
def post_func(df):
    # Reset the index so we can manipulate
    df.reset_index(inplace=True)
    
    # Group morning and afternoon into single row
    df['dt'] = df['dt'].dt.tz_convert(TIMEZONE)
    df['signal_time'] = df.apply(lambda row: 'am' if row['dt'].hour == 7 else 'pm', axis=1)
    df['dt'] = df['dt'].dt.normalize().dt.tz_convert('UTC')
    df = pd.pivot_table(df, values='signal', index=['dt', 'sid', 'fiscal_date'], columns=['signal_time'], aggfunc=np.max)
    df.reset_index(inplace=True)
    
    if DEBUG:
        log.info(df.dtypes)
        log.info(df.head(10))
    
    # Group rows to find last day of release signal
    zdf = df.groupby(['sid', 'fiscal_date']).agg('max')
    zdf.reset_index(inplace=True)
    
    # Add day to last day of release signal and set signal to 0
    zdf['dt'] = zdf['dt'] + timedelta(days=1)
    zdf['am'] = np.nan
    zdf['pm'] = np.nan
    
    # Drop unused columns
    df.drop(['fiscal_date'], axis=1, inplace=True)
    zdf.drop(['fiscal_date'], axis=1, inplace=True)
    
    # Concatenate to original DataFrame
    df = pd.concat([df, zdf], copy=False)
    
    # Set the index back
    df.set_index('dt', inplace=True)
    
    if DEBUG:
        aapl = df[df['sid'] == sid(24)]
        log.info(aapl.head(10))
    
    return df

def fix_missing_symbols(df):
    symbol_map = {
        'ACIIQ': 'ACI',
        'AGN-defunct': 'AGN',
        'ATK': 'ATKWI',
        'AROPQ': 'ARO',
        'BIN': None,
        'CMCSA': 'CMCS_A',
        'DISCA': 'DISC_A',
        'ENR': 'ENR',
        'ETP-Defunct': 'ETP',
        'EXXIQ': 'EXXI',
        'GCI': 'GCI',
        'GLFM': 'GLF',
        'GOOGL': 'GOOG_L',
        'JCI-Defunct': 'JCI',
        'KELYA': 'KELY_A',
        'LBTYA': 'LBTY_A',
        'LGF-A': 'LGF',
        'MRKT': None,
        'MCPIQ': 'MCP',
        'MHRCQ': None,
        'MSG': 'MSG',
        'NU': None,
        'NYLD': 'NYLD_A',
        'PVAH': 'PVAC',
        'RHNOD': 'RNO',
        'ROVI': None,
        'RUSHA': 'RUSH_A',
        'SDOC': 'SD',
        'STRZA': 'STRZ_A',
        'SUNEQ': 'SUNE',
        'SWFT': 'SWFT',
        'TAL-Defunct': 'TAL',
        'TLLP': None,
        'TRNX': None,
        'TSO': 'TSO_WI',
        'UNISQ': 'UNIS',
        'UPLMQ': 'UPL',
        'WLTGQ': 'WLT',
        'ZQKSQ': 'ZQK'
    }
    
    for k, v in symbol_map.items():
        if v:
            df.loc[df['symbol'] == k, 'symbol'] = v

def before_trading_start(context, data):
    # Add pipeline to context
    context.pipeline = pipeline_output('pipeline')
    
def setup_positions(context, data):
    now = get_datetime().tz_convert(TIMEZONE)
    signal_column = 'am' if now.hour < 12 else 'pm'
    
    # Fetch entire universe of pipeline and fetched assets
    results = data.current(context.pipeline.index, [signal_column])
    results.rename(columns={signal_column: 'signal'}, inplace=True)
    
    # Drop assets where there is no signal data
    results.dropna(inplace=True)
    
    collect_aapl_signals(context, results)
    record_data(results, symbols('AMZN', 'AAPL'))
    record(num_assets=len(results))
    
    # Add longs/shorts to context:
    if PERCENTILE is not None:
        top_quantile = results.signal.quantile(1.0 - PERCENTILE)
        bottom_quantile = results.signal.quantile(PERCENTILE)
        
        context.longs = results[results.signal >= top_quantile].index.values
        context.shorts = results[results.signal <= bottom_quantile].index.values
        
    elif len(results) >= (MIN_STOCKS * 2):
        context.longs = results.nlargest(MIN_STOCKS, 'signal').index.values
        context.shorts = results.nsmallest(MIN_STOCKS, 'signal').index.values
        
    else:
        context.longs = []
        context.shorts = []
    
    record(num_longs=len(context.longs))
    record(num_shorts=len(context.shorts))
    
    # Close
    context.close = list(set(context.portfolio.positions.keys()) - (set(context.longs) | set(context.shorts)))
    
def trade(context, data):
    cancel_open_orders(context)
    setup_positions(context, data)
    
    if can_trade(context):
        rebalance(context)
    else:
        close_all_positions(context)
        
def can_trade(context):
    return len(context.shorts) >= MIN_STOCKS and len(context.longs) >= MIN_STOCKS
    
def rebalance(context):
    long_allocation = 1.0 / len(context.longs)
    short_allocation = -1.0 / len(context.shorts)
    
    for asset in context.close:
        order_target_percent(asset, 0.0)

    for asset in context.longs:
        order_target_percent(asset, long_allocation)

    for asset in context.shorts:
        order_target_percent(asset, short_allocation)
        
def close_all_positions(context):
    for asset in context.portfolio.positions.keys():
        order_target_percent(asset, 0.0)

def cancel_open_orders(context):
    all_open_orders = get_open_orders()
    
    if all_open_orders:
        for security, oo_for_sid in all_open_orders.iteritems():
            for order_obj in oo_for_sid:
                cancel_order(order_obj)

def record_data(results, stocks):
    for stock in stocks:
        asset = results[results.index == stock]
        signal_series = '{}_signal'.format(stock.symbol)
    
        if not asset.empty:
            record(signal_series, asset.iloc[0]['signal'])
        else:
            record(signal_series, 0)

def print_results(context, data):
    print(context.aapl_signals)
            
def collect_aapl_signals(context, results):
    adf = results[results.index == sid(24)]
    
    if not adf.empty:
        adf['dt'] = get_datetime()
        
        if context.aapl_signals is None:
            context.aapl_signals = adf
        else:
            context.aapl_signals = pd.concat([context.aapl_signals, adf], copy=False)            

def make_pipeline():
    universe = default_us_equity_universe_mask(minimum_market_cap=100000000)
    # market_cap = MarketCap()
    dv = AverageDollarVolume(window_length=20)
    price = USEquityPricing.close.latest
    
    # min_market_cap = market_cap >= 100e6
    min_dv = dv >= 1e6
    min_price = price >= 4.0
    
    screen = (universe & min_dv & min_price)
    
    return Pipeline(screen=screen)

class IdealSlippage(slippage.SlippageModel):
    
    def simulate(self, data, asset, orders_for_asset):
        # Make sure price exsists
        price = data.current(asset, 'close')

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END
        
        dt = data.current_dt

        for order in orders_for_asset:
            if order.open_amount == 0:
                continue

            order.check_triggers(price, dt)
            if not order.triggered:
                continue
                
            txn = slippage.create_transaction(
                order,
                data.current_dt,
                price,
                order.amount
            )
            
            yield order, txn

    def process_order(self, data, order):
        pass