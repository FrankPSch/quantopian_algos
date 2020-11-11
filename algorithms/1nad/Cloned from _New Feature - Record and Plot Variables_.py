def initialize(context):
    context.sid = sid(26578)
    context.invested = False

def handle_data(context, data):
    context.price = data[context.sid].price
    short = data[context.sid].mavg(20)
    long = data[context.sid].mavg(60)
    
    if (short > long) and not context.invested:
        order(context.sid, 500)
        context.invested = True
    elif (short < long) and context.invested:
        order(context.sid, -500)
        context.invested = False

    record(short_mavg = short,
        long_mavg = long,
        goog_price = context.price)