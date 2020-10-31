import numpy as np

def initialize(context):
    """
    Called once at the start of the algorithm.
    """   
    context.universe = [sid(8554), sid(22972), sid(22446), sid(23870), sid(26807)]
    schedule_function(my_rebalance, date_rules.month_end(), time_rules.market_close(hours=1))
         
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


def my_rebalance(context,data):
    closings = data.history(context.universe, fields = "price", bar_count = 61, frequency = "1d")
    return_20days = closings.ix[-2] / closings.ix[-21] - 1
    return_60days = closings.ix[-2] / closings.ix[-61] - 1
    return_daily = closings.pct_change(1)
    roll_vol_20days = return_daily.ix[-21:-1].std(axis = 0)
    rank_20days_ret = return_20days.rank(ascending=False)
    rank_60days_ret = return_60days.rank(ascending=False)
    rank_20days_roll_vol = roll_vol_20days.rank(ascending=True)
    # 20 days x 0.3, 60 days x 0.4, and 20 days vol x 0.3
    weighted_rank = rank_20days_ret * 0.3 + rank_60days_ret * 0.4 + rank_20days_roll_vol * 0.3
    context.to_buy = weighted_rank.sort_values(ascending = True).index[:2].tolist()
    print(context.to_buy)
    for security in context.universe:
        if (security not in context.to_buy) & (data.can_trade(security)):
            order_target_percent(security,0)
        elif (security in context.to_buy) & (data.can_trade(security)):
            order_target_percent(security, 0.5)
            
    #record("Leverage", context.account.leverage)
    #record("SPY", data.current(context.universe[0],"price"))
    #record("EFA", data.current(context.universe[1],"price"))
    #record("ICF", data.current(context.universe[2],"price"))
    #record("IEF", data.current(context.universe[3],"price"))
    #record("GLD", data.current(context.universe[4],"price"))