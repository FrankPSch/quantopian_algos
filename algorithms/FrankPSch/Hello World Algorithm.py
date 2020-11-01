# Called once at the start of the simulation.
def initialize(context):
    # Reference to the AAPL security.
    context.aapl = sid(24)

    # Rebalance every day, one hour and a half after market open.
    schedule_function(my_rebalance, 
        date_rules.every_day(), 
        time_rules.market_open(hours=1, minutes=30))

# This function was scheduled to run once per day at 11AM ET.
def my_rebalance(context, data):
    
    # Take a 100% long position in AAPL. Readjusts each day to 
    # account for price fluctuations.
    if data.can_trade(context.aapl):
        order_target_percent(context.aapl, 1.00)
