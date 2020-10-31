import math
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.covariance import OAS
import statsmodels.api as smapi
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.filters.eventvestor import IsAnnouncedAcqTarget
import quantopian.optimize as opt

from quantopian.pipeline.data.eventvestor import EarningsCalendar
from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings
)

def make_pipeline():
    minprice = USEquityPricing.close.latest > 5
    not_announced_acq_target = ~IsAnnouncedAcqTarget()
    pipe = Pipeline(screen=Q1500US() & minprice & not_announced_acq_target)
    
    sectors = Sector()
    pipe.add(sectors, 'sector')
    pipe.add(BusinessDaysSincePreviousEarnings(), 'PE')
    return pipe
    
def initialize(context):
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB
    
    context.sectorStocks = {}
    context.stocks = None
    context.alphas = None
    context.betas = None
    
    context.sector_ids = [ Sector.BASIC_MATERIALS,
                           Sector.CONSUMER_CYCLICAL,
                           Sector.FINANCIAL_SERVICES,
                           Sector.REAL_ESTATE,
                           Sector.CONSUMER_DEFENSIVE,
                           Sector.HEALTHCARE,
                           Sector.UTILITIES,
                           Sector.COMMUNICATION_SERVICES,
                           Sector.ENERGY,
                           Sector.INDUSTRIALS,
                           Sector.TECHNOLOGY ]
    
    context.leverage = 1.
    context.days = 45
    context.counter = 2
    schedule_function(trade_sectors, 
                      date_rules.every_day(), 
                      time_rules.market_open(minutes=60))
    
    schedule_function(close_all, 
                      date_rules.every_day(), 
                      time_rules.market_close(minutes=30))
    
    schedule_function(update_chart, 
                      date_rules.every_day(), 
                      time_rules.market_close(minutes=1))
    
    attach_pipeline(make_pipeline(), "Q1500")
    
def handle_data(context, data):
    pass

def close_all(context, data):
    os = get_open_orders()
    
    for ol in os.values():
        for o in ol:
            cancel_order(o)
    
    for sid in context.portfolio.positions:
        order_target(sid, 0)

def before_trading_start(context, data):
    context.screener = pipeline_output("Q1500")
    context.screener = context.screener[context.screener['PE'] > 2].index
    
    if context.days < 45:
        context.days += 1
        return
    context.days = 0
    context.output = pipeline_output("Q1500")
    context.sectorStocks.clear()
    
    context.sectorStocks[Sector.BASIC_MATERIALS] = get_cluster(context, data, context.output[context.output.sector == Sector.BASIC_MATERIALS].index)    
    context.sectorStocks[Sector.CONSUMER_CYCLICAL]= get_cluster(context, data, context.output[context.output.sector == Sector.CONSUMER_CYCLICAL].index)    
    context.sectorStocks[Sector.CONSUMER_DEFENSIVE]= get_cluster(context, data, context.output[context.output.sector == Sector.CONSUMER_DEFENSIVE].index)
    context.sectorStocks[Sector.FINANCIAL_SERVICES]= get_cluster(context, data, context.output[context.output.sector == Sector.FINANCIAL_SERVICES].index)
    context.sectorStocks[Sector.REAL_ESTATE] = get_cluster(context, data, context.output[context.output.sector == Sector.REAL_ESTATE].index)
    context.sectorStocks[Sector.HEALTHCARE] = get_cluster(context, data, context.output[context.output.sector == Sector.HEALTHCARE].index)
    context.sectorStocks[Sector.UTILITIES] = get_cluster(context, data, context.output[context.output.sector == Sector.UTILITIES].index)
    context.sectorStocks[Sector.COMMUNICATION_SERVICES] = get_cluster(context, data, context.output[context.output.sector == Sector.COMMUNICATION_SERVICES].index)
    context.sectorStocks[Sector.ENERGY] = get_cluster(context, data, context.output[context.output.sector == Sector.ENERGY].index)
    context.sectorStocks[Sector.INDUSTRIALS]= get_cluster(context, data, context.output[context.output.sector == Sector.INDUSTRIALS].index)
    context.sectorStocks[Sector.TECHNOLOGY] = get_cluster(context, data, context.output[context.output.sector == Sector.TECHNOLOGY].index)
    
    
def get_cluster(context, data, stocks):
    return stocks
    
def trade_sectors(context, data):
    context.stocks = None
    context.alphas = None
    context.betas = None
    context.sectors = {}
    for sector_id in context.sector_ids:
        if sector_id not in context.sectorStocks or len(context.sectorStocks[sector_id]) < 30:
            continue
        stocks, alphas, betas = find_weights(context, data, context.sectorStocks[sector_id])
        
        if stocks is None:
            continue
            
        if context.stocks is None:
            context.stocks = stocks
            context.alphas = alphas
            context.betas = betas
        else:
            context.stocks = np.hstack((context.stocks, stocks))
            context.alphas = np.hstack((context.alphas, alphas))
            zero1 = np.zeros((context.betas.shape[0], betas.shape[1]))
            zero2 = np.zeros((betas.shape[0], context.betas.shape[1]))
            context.betas = np.hstack((context.betas, zero1))
            betas = np.hstack((zero2, betas))
            context.betas = np.vstack((context.betas, betas))
            
        for sid in context.stocks:
            context.sectors[sid] = sector_id
        
    if context.stocks is None:
        return
    
    todays_universe = context.stocks
    N = context.betas.shape[1]
    M = context.betas.shape[0]
    names = [str(i) for i in range(0, N)]
    risk_factor_exposures = pd.DataFrame(context.betas, index=todays_universe, columns=names)
    objective = opt.MaximizeAlpha(pd.Series(-context.alphas, index=todays_universe))
    
    constraints = []

    constraints.append(opt.MaxGrossLeverage(1.0))
    constraints.append(opt.DollarNeutral(0.0001))
    neutralize_risk_factors = opt.WeightedExposure(
        loadings=risk_factor_exposures,
        min_exposures=pd.Series([-0.01] * N, index=names),
        max_exposures=pd.Series([0.01] * N, index=names))
    constraints.append(neutralize_risk_factors)
    sector_neutral = opt.NetPartitionExposure.with_equal_bounds(labels=context.sectors, min=-0.0001, max=0.0001)
    constraints.append(sector_neutral)
    constraints.append(opt.PositionConcentration.with_equal_bounds(min=-10./M, max=10./M))
    order_optimal_portfolio(objective=objective, constraints=constraints)
                
def find_weights(context, data, stocks):
    prices = data.history(stocks, "price", 90, "1d")
    prices = prices.dropna(axis=1)
    
    dropsids = []
    
    for sid in prices:
        if sid not in context.screener:
            dropsids.append(sid)
    
    prices = prices.drop(dropsids, axis=1)
    
    logP = np.log(prices.values)
    diff = np.diff(logP, axis=0)
    factors = PCA(0.9,whiten=False).fit_transform(diff)
    model = smapi.OLS(diff, smapi.add_constant(factors)).fit()
    betas = model.params.T[:, 1:]
    model = smapi.GLS(diff[-1, :], betas, weights=1. / np.var(diff, axis=0)).fit()
    return prices.columns.values, sp.stats.zscore(model.resid), betas

def update_chart(context,data):
    record(leverage = context.account.leverage)

    longs = shorts = 0
    
    for position in context.portfolio.positions.itervalues():        
        if position.amount > 0:
            longs += 1
        if position.amount < 0:
            shorts += 1
            
    record(long_lever=longs, short_lever=shorts)