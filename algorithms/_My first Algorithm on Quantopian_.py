def initialize(context):  
    context.fcau = sid(47888)  
    schedule_function(open_positions, date_rules.week_start(), time_rules.market_open())  
    schedule_function(close_positions, date_rules.week_end(), time_rules.market_close(minutes=30))

def open_positions(context, data):  
    order_target_percent(context.fcau, 0.50)

def close_positions(context, data):  
    order_target_percent(context.fcau, 0)