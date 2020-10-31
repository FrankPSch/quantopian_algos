# Здесь мы используем стратегию, в которой работает фактор "Моментум", # т.е. заходим вдолгую в те активы, которые за прошедший период времени # росли наиболее сильно.  
# Код для сортировки "успешных" акций был позаимствован отсюда: # https://www.quantopian.com/posts/quick-and-dirty-momentum-strategy
from operator import itemgetter

def initialize(context):
    set_long_only()
    context.topMom = 1
    context.rebal_int = 3
    context.lookback = 250
    set_symbol_lookup_date('2015-01-01')
    #В качестве выборки взяты компании из индекса Доу-Джонса, которые там находились в 2006 году
    context.stocks = symbols('GE', 'XOM', 'PG', 'UTX', 'MMM', 'IBM', 'MRK', 'AXP', 'MSD', 'BA', 'KO', 'CAT', 'JPM', 'DIS', 'JNJ', 'WMT', 'HD', 'INTC', 'MSFT', 'PFT', 'VZ', 'GM')  
    schedule_function(rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open())
    

def rebalance(context, data):
    #Создали список акций по моментуму
    MomList = GenerateMomentumList(context, data)
    
    
    for stock in context.portfolio.positions:
        order_target(stock, 0)
    #Это наш бенчмарк, который мы также рисуем на графике для наглядности
    spy = symbol('SPY')
    
    # weight = /context.topMom
    weight = 1

    #Идем в длинную по самому первому в рейтинге MomList, то есть выбираем самый быстрорастущий!
    for l in MomList:
        stock = l[0]
        if stock in data and data[stock].close_price > data[stock].mavg(200):
            order_percent(stock, weight)
    pass

  
# Тут мы генерируем рейтинг акций, ранжируя их по скорости месячного роста
def GenerateMomentumList(context, data):
    
    MomList = []
    price_history = history(bar_count=context.lookback, frequency="1d", field='price')
#Здесь old - месяц назад
#Добавляем для каждой акции ее прирост за месяц
    for stock in context.stocks:
        now = price_history[stock].ix[-1]
        old = price_history[stock].ix[0]
        pct_change = (now - old) / old
        MomList.append([stock, pct_change, price_history[stock].ix[0]])

    
    MomList = sorted(MomList, key=itemgetter(1), reverse=True)
    #Отсортировали, сделали так, чтобы показывался 
    MomList = MomList[0:context.topMom]
#Выбираем наилучший!
    print (MomList[0][0].symbol)

    return MomList


def change(one, two):
    return(( two - one)/one)
    
def handle_data(context, data):
    pass