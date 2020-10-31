# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
import numpy as np

def initialize(context):
    context.asset1 =symbol('ABGB')
    context.asset2 = symbol('FSLR')
    context.price1 = 0
    context.price2 = 0
    context.pos = 0
    context.deltas=[]
    
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    # Implement your algorithm logic here.

    # data[sid(X)] holds the trade event data for that security.
    # context.portfolio holds the current portfolio state.

    # Place orders with the order(SID, amount) method.

    # TODO: implement your own logic here.
    curp1 = data.current(context.asset1, 'price')
    curp2 = data.current(context.asset2, 'price')
    #print(curp1, curp2)
    if context.price1 == 0 or context.price2 == 0:
        context.price1 = curp1
        context.price2 = curp2
        return
    delta = curp1 - context.price1 - (curp2 - context.price2)
    context.deltas.append(delta)
    #hedge ratio
    
    #ratio = curp1 / curp2
    if context.pos == 0: 
        if delta > 5: 
            #share1 = int(context.portfolio.cash*0.4 / curp1)
            #share2 = int(context.portfolio.cash*0.4 / curp2)
            share = int(context.portfolio.cash * 0.8 / (curp1 + curp2))
            order(context.asset1, -share)
            order(context.asset2, share)
            context.pos = 1
            log.info('Open: sell {} shares of {}, buy {} shares of {}'.format(share, context.asset1, share, context.asset2))
        elif delta < -5:
            #share1 = int(context.portfolio.cash*0.4 / curp1)
            #share2 = int(context.portfolio.cash*0.4 / curp2)
            share = int(context.portfolio.cash * 0.8 / (curp1 + curp2))
            order(context.asset1, share)
            order(context.asset2, -share)
            context.pos = -1
            log.info('Open: buy {} shares of {}, sell {} shares of {}'.format(share, context.asset1, share, context.asset2))
    else:
        if context.pos == 1 and delta < 0:
            order_target(context.asset1, 0)
            order_target(context.asset2, 0)
            context.pos = 0
            log.info('Close: buy {}, sell {}'.format(context.asset1, context.asset2))
        elif context.pos == -1 and delta > 0:
            order_target(context.asset1, 0)
            order_target(context.asset2, 0)
            context.pos = 0
            log.info('Close: sell {}, buy {}'.format(context.asset1, context.asset2))
                    
    record(delta = delta)
    #record(assets = context.portfolio.portfolio_value)
    #arr = np.array(context.deltas)
    #log.info('mean: {}, std: {}'.format(arr.mean(), arr.std()))