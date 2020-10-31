import pandas

def rename_col(df):
    df = df.rename(columns={'New York 15:00': 'price'})
    df = df.rename(columns={'Value': 'price'})
    df = df.fillna(method='ffill')
    df = df[['price', 'sid']]
    # Correct look-ahead bias in mapping data to times   
    df = df.tshift(1, freq='b')
    log.info(' \n %s ' % df.head())
    return df

def preview(df):
    log.info(' \n %s ' % df.head())
    return df
    
def initialize(context):
    # import the external data
    fetch_csv('https://www.quandl.com/api/v1/datasets/JOHNMATT/PALL.csv?trim_start=2012-01-01',
        date_column='Date',
        symbol='palladium',
        pre_func = preview,
        post_func=rename_col,
        date_format='%Y-%m-%d')

    fetch_csv('https://www.quandl.com/api/v1/datasets/BUNDESBANK/BBK01_WT5511.csv?trim_start=2012-01-01',
        date_column='Date',
        symbol='gold',
        pre_func = preview,
        post_func=rename_col,
        date_format='%Y-%m-%d')
    
    # Tiffany
    context.stock = sid(7447)
    
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())

 

def rebalance(context, data):
    # Invest 100% of the portfolio in Tiffany stock when the price of gold is low.
    # Decrease the Tiffany position to 50% of portfolio when the price of gold is high.
    current_gold_price = data.current('gold', 'price')
    if (current_gold_price < 1600):
       order_target_percent(context.stock, 1.00)
    if (current_gold_price > 1750):
       order_target_percent(context.stock, 0.50)

    # Current prices of palladium (from .csv file) and TIF
    current_pd_price = data.current('palladium', 'price')
    current_tif_price = data.current(context.stock, 'price')
    
    # Record the variables
    record(palladium=current_pd_price, gold=current_gold_price, tif=current_tif_price)
