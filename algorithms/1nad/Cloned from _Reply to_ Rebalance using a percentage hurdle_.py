PercentEquities = 70.0

def initialize(context):
    set_symbol_lookup_date('2015-01-01')
    context.Equities    = symbols("JKL","FAB","FVD","FTA","EZY","EPS","DEF","PRF","PWV","FDL","IWD")
    context.FixedIncome = symbols("TLO","VGSH","SCHO","PLW","EDV","GOVT","IEI","SHY","IEF","TLT")
    schedule_function(Rebalance, date_rule=date_rules.month_start())

def handle_data(context, data):  
    record(Leverage     = context.account.leverage)
    record(AccountValue = context.portfolio.positions_value + context.portfolio.cash)
    
def Rebalance(context, data):  
    accountValue = context.portfolio.positions_value + context.portfolio.cash
    
    pctEquities = PercentEquities / 100.0
    pctFixedInc = 1.0 - pctEquities
        
    #
    # Equities rebalance
    #    
    eligible = []
    accruedValue = 0.0
    for stock in context.Equities:    
        if (stock in data):
            eligible.append(stock)
            accruedValue += context.portfolio.positions[stock].amount * data[stock].close_price
            
    eligibleCount = float(len(eligible))
    for stock in eligible:
        order_target_percent(stock, pctEquities / eligibleCount)

    pctOfAccount = accruedValue / accountValue * 100.0
    record(EquityPct = pctOfAccount)
    print("Equities % of account {0:>5.2f}".format(pctOfAccount))
    
    #
    # Fixed Income rebalance
    #    
    eligible = []
    accruedValue = 0.0
    for stock in context.FixedIncome:
        if (stock in data):
            eligible.append(stock)
            accruedValue += context.portfolio.positions[stock].amount * data[stock].close_price
            
    eligibleCount = float(len(eligible))
    for stock in eligible:
        order_target_percent(stock, pctFixedInc / eligibleCount)
        
    pctOfAccount = accruedValue / accountValue * 100.0
    record(FixedIncomePct = pctOfAccount)
    print("FixedInc % of account {0:>5.2f}".format(pctOfAccount))
