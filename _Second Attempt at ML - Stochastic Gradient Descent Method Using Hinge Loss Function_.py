import numpy as np
import random

def initialize(context):
    set_universe(universe.DollarVolumeUniverse(97, 99))
    context.bet_amount = 100000.0
    context.max_notional=1000000.1
    context.min_notional=-100000.0

    
def handle_data(context, data):

    for stock in data:
           
        if caltheta(data,stock,5) is None:
            continue

        (theta,historicalPrices) = caltheta(data,stock,5)
        indicator=np.dot(theta,historicalPrices)
        # normalize
        hlen=sum([k*k for k in historicalPrices])
        tlen=sum([j*j for j in theta])
        indicator/=float(hlen*tlen) #now indicator lies between [-1,1]
        
        current_Prices = data[stock].price
        notional = context.portfolio.positions[stock].amount * current_Prices

            
        if  indicator>=0 and notional<context.max_notional:
            order(stock,indicator*context.bet_amount)
            log.info("%f shares of %s bought." %(context.bet_amount*indicator,stock))
                
        if  indicator<0 and notional>context.min_notional:
            order(stock,indicator*context.bet_amount)
            log.info("%f shares of %s sold." %(context.bet_amount*indicator,stock))

   

@batch_transform(refresh_period=1,window_length=60)
def caltheta(datapanel, sid, num):
    prices=datapanel['price'][sid]
    for i in range(len(prices)):
        if prices[i] is None:
            return None
    testX=[[prices[i] for i in range(j,j+4)] for j in range(0,60,5)]
    avg=[np.average(testX[k]) for k in range(len(testX))]
    testY=[np.sign(prices[5*i+4]-avg[i]) for i in range(len(testX))]
    theta=hlsgdA(testX, testY, 0.01, randomIndex, num)
    priceh=prices[-4:] #get historical data for the last four days
    return (theta,priceh)


# stochastic gradient descent using hinge loss function  
def hlsgdA(X, Y, l, nextIndex, numberOfIterations):
    theta=np.zeros(len(X[0]))
    best=np.zeros(len(X[0]))
    e=0
    omega=1.0/(2*len(Y))
    while e<numberOfIterations:
        ita=1.0/(1+e)
        for i in range(len(Y)):
            index=nextIndex(len(Y))
            k=np.dot(ita,(np.dot(l,np.append([0],[k for k in theta[1:]]))-np.dot((sgn(1-Y[index]*np.dot(theta,X[index]))*Y[index]),X[index])))
            theta-=k
            best=(1-omega)*best+omega*theta  #a recency-weighted average of theta: an average that exponentially decays the influence of older theta values
        e+=1
    return best

# sign operations to identify mistakes
def sgn(k):
    if k<=0:
        return 0
    else:
        return 1

def randomIndex(n):
    return random.randint(0, n-1)