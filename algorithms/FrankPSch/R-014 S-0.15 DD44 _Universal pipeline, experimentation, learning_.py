'''
https://www.quantopian.com/posts/universal-pipeline-for-experimentation-and-learning
'''
from quantopian.pipeline              import Pipeline
from quantopian.pipeline.data         import Fundamentals
from quantopian.algorithm             import attach_pipeline, pipeline_output
from quantopian.pipeline.filters      import Q500US, Q1500US, Q3000US
from quantopian.pipeline.factors      import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.classifiers.morningstar import Sector
import quantopian.optimize as opt
import statsmodels.api as sm
import numpy  as np
import pandas as pd

def make_pipeline(c):
    m = VolumeMin(window_length=42).top(400) & Sector().notnull() & Q1500US()  # mask
    #                            Adding to mask, excluding values around the middle, ~ means not.
    a = EBITPerEV      (mask=m); m &= ~(a.percentile_between(40, 60))
    #b = Momentum       (mask=m); m &= ~(b.percentile_between(40, 60))
    #c = ROE1           (mask=m); m &= ~(c.percentile_between(40, 60))
    d = Div_Yield      (mask=m); m &= ~(d.percentile_between(40, 60))
    #e = Slope          (mask=m); m &= ~(e.percentile_between(40, 60))
    #f = Price_Earnings (mask=m); m &= ~(f.percentile_between(40, 60))
    return Pipeline(
        screen  = m,
        columns = {
            'a': a,
            #'b': b,
            #'c': c,
            'd': -d,   # sometimes just tossing a minus sign in front makes for higher outuput
            #'e': e,
            #'f': f,
            'sector': Sector(mask=m),
        })

def initialize(context):
    c = context
    c.long_shrt_num = 100
    c.nullzone      = .2
    c.headroom      = 0
    c.log_universe  = 2     # Number of days to log universe in before_trading_start.
    c.cannot_hold   = []    # Securities you want optimize to not order or hold onto.
    #set_commission(commission.PerShare(cost=0.001, min_trade_cost=0)) # default now?
    attach_pipeline(make_pipeline(c), 'p')

    use_optimize = 1
    if use_optimize:
        schedule_function(do_opt,   date_rules.week_start(2), time_rules.market_open(minutes=1))
    else:
        schedule_function(do_shrts,   date_rules.month_start(), time_rules.market_open(minutes=1 ))
        schedule_function(cancel_oos, date_rules.month_start(), time_rules.market_open(minutes=9))
        schedule_function(do_longs,   date_rules.month_start(), time_rules.market_open(minutes=10))

    schedule_function(cancel_oos, date_rules.every_day(), time_rules.market_close())

    # This is included for the indication of profit per dollar invested since
    #   different factors might not always invest the same amount. Apples-to-apples comparison.
    for i in range(1, 391):
        #break
        schedule_function(pvr, date_rules.every_day(), time_rules.market_open(minutes=i))
        
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)) # Default
    set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0)) # FSC for IB


def before_trading_start(context, data):
    c = context
    o = pipeline_output('p').dropna().drop(c.cannot_hold, errors='ignore')
    c.ori = o
    if not len(o): return
    num = int(min(c.long_shrt_num, len(o)/2)) ; nullzone = c.nullzone
    if 'score' in o.columns: o['score_ori'] = o['score']
    score = 0 ; valid = 0
    for col in o.columns:
        if not np.issubdtype(o[col][0].dtype, np.number): continue  # skip True/False columns
        if col == 'sector': continue
        o[col] += abs(o[col].min())         # shift to positive
        valid = 1
    if not valid:
        log.info('Found no columns with numbers in before_trading_start() pipe')
        return
    for col in o.columns:
        if not np.issubdtype(o[col][0].dtype, np.number): continue
        if col == 'sector': continue
        o[col] /= o[col].sum()              # normalize
        if not score: o['score']  = o[col]  # combine values
        else:         o['score'] += o[col]
        score = 1
    o['score']  = o['score'].rank()
    o['score'] /= o['score'].sum()
    o['score'] -= o['score'].mean()
    o   = o.dropna()
    mid = o.score.rank().mean(); sliver = (nullzone * mid)
    longs   = o[(o.score.rank() > mid + sliver)].head(num)
    shrts   = o[(o.score.rank() < mid - sliver)].tail(len(longs))
    c.longs =  longs['score'] / longs['score'].sum()
    c.shrts = -shrts['score'] / shrts['score'].sum()

    # Log pipe length & some long, short details a number of times.
    if c.log_universe >= 0:
        lng = c.longs.sort_values(ascending=False)
        shs = c.shrts.sort_values(ascending=False)
        log.info('pipe len {}'.format(len(c.ori)))
        log.info('lng {}  top {} {}  bottom {} {}'.format(len(lng),
            lng.index[0].symbol, '%5f' % lng[0], lng.index[-1].symbol, '%5f' % lng[-1]))
        log.info('shs {}  top {} {}  bottom {} {}'.format(len(shs),
            shs.index[0].symbol, '%5f' % shs[0], shs.index[-1].symbol, '%5f' % shs[-1]))
        c.log_universe -= 1

    c.actives = c.longs.index.union(c.shrts.index)
    c.pipe    = longs.append(shrts)

def do_opt(context, data):
    order_optimal_portfolio(
        # For objective, simply use naive ranks as an alpha coefficient
        # and try to maximize that alpha.
        #
        # This is a **very** naive model. Since alphas are so widely spread out,
        # should expect to always allocate the maximum amount of long/short
        # capital to assets with high/low ranks.
        #
        # A more sophisticated model would apply some re-scaling here to try to generate
        # more meaningful predictions of future returns.
        objective = opt.MaximizeAlpha(context.pipe.score),
        constraints=[
            # Constrain gross leverage to 1.0 or less. This means that the absolute
            #   value of long and short positions should not exceed the value of portfolio.
            opt.MaxGrossExposure(1.0),
            # Constrain individual position size to no more than a fixed percentage
            # of portfolio. Because alphas are so widely distributed,
            # should expect to end up hitting this max for every stock in universe.
            opt.PositionConcentration.with_equal_bounds(-.015, .015),
            opt.DollarNeutral(),    # Same amount of capital to long and short positions.
            opt.NetGroupExposure.with_equal_bounds(   # Net leverage in each sector.
                labels = context.pipe.sector,
                min = -0.0001,
                max =  0.0001,
            ),
            opt.CannotHold(context.cannot_hold)
        ],
    )

def close(context, data):
    for s in context.portfolio.positions:
        if s in context.actives:  continue
        if not data.can_trade(s): continue
        order_target(s, 0)

def do_shrts(context, data):
    c = context
    cancel_oos(context, data)
    c.headroom = max(0, .5 * c.portfolio.cash)
    for s in c.shrts.index:
        if get_open_orders(s):    continue
        if not data.can_trade(s): continue
        order_target_value(s, c.shrts[s] * c.headroom)

def do_longs(context, data):
    c = context
    cancel_oos(c, data)
    for s in c.longs.index:
        if get_open_orders(s):    continue
        if not data.can_trade(s): continue
        order_target_value(s, c.longs[s] * c.headroom)

def cancel_oos(context, data):    # Primarily to prevent the logging of unfilled orders at end of day
    oo = get_open_orders()        #   Can also be use at any time to limit partial fills.
    for s in oo:
        for o in oo[s]:
            # Next line can be beneficial if midday cancel_oos() in use.
            #if cls_opn_crs(c, o) in [0, 2]: continue  # closing, leave it alone
            cancel_order(o.id)

def cls_opn_crs(c, o):      # c = context    o = order object
    # Whether order is closing, opening or crossover (short to long or reverse)
    #   https://www.quantopian.com/posts/order-state-on-partial-fills-close-open-or-crossover
    if c.portfolio.positions[o.sid].amount * o.amount < 0:   # close or crossover
        if abs(c.portfolio.positions[o.sid].amount) < abs(o.amount - o.filled):
            if abs(c.portfolio.positions[o.sid].amount) - abs(o.filled) < 0:
                  return 3  # crossed 0 shares and now opening
            else: return 2  # cross closing
        else:     return 0  # closing
    else:         return 1  # opening

'''
    Fundamentals, Factors ...
'''

def nanfill(_in):    # From https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    # Includes a way to count nans on webpage at
    #   https://www.quantopian.com/posts/forward-filling-nans-in-pipeline

    #return _in            # uncomment to not run the code below
    mask = np.isnan(_in)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    _in[mask] = _in[np.nonzero(mask)[0], idx[mask]]
    return _in

def beta(ts, benchmark, benchmark_var):
    return np.cov(ts, benchmark)[0, 1] / benchmark_var
def slope(in_):   # Slope of regression line. Make sure input has no nans or screen its output later
    # https://www.quantopian.com/posts/slope-calculation
    return sm.OLS(in_, sm.add_constant(range(-len(in_) + 1, 1))).fit().params[-1]  # slope
def curve(_in):   # ndarray   see https://www.quantopian.com/posts/curve-calculation
    return sm.OLS(_in[-len(_in)/2:], sm.add_constant(range(-len(_in[-len(_in)/2:]) + 1, 1))).fit().params[-1] - sm.OLS(_in[0:len(_in)/2], sm.add_constant(range(-len(_in[0:len(_in)/2]) + 1, 1))).fit().params[-1]

class AvgDailyDollarVolumeTraded(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume] ; window_length = 42
    def compute(self, today, assets, out, close, volume):
        volume = nanfill(volume)
        close  = nanfill(close)
        out[:] = np.mean(close * volume, axis=0)
class ATR(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.high,USEquityPricing.low]
    window_length = 21
    def compute(self, today, assets, out, close, high, low):
        close  = nanfill(close)
        high   = nanfill(high)
        low    = nanfill(low)
        hml    = high - low
        hmpc   = np.abs(high - np.roll(close, 1, axis=0))
        lmpc   = np.abs(low - np.roll(close, 1, axis=0))
        tr     = np.maximum(hml, np.maximum(hmpc, lmpc))
        atr    = np.mean(tr[1:], axis=0)
        out[:] = atr
class Beta(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 60
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        returns = pd.DataFrame(close, columns=assets).pct_change()[1:]
        spy_returns = returns[sid(8554)]
        spy_returns_var = np.var(spy_returns)
        out[:] = returns.apply(beta, args=(spy_returns,spy_returns_var,))
class CashReturn(CustomFactor):
    inputs = [Fundamentals.cash_return] ; window_length = 42
    def compute(self, today, assets, out, cash_return):
        cash_return = nanfill(cash_return)
        out[:] = np.mean(cash_return, axis=0)
class CashReturnSlope(CustomFactor):
    inputs = [Fundamentals.cash_return] ; window_length = 5
    def compute(self, today, assets, out, cash_return):
        cash_return = nanfill(cash_return)
        out[:] = slope(cash_return)
class CrossSectionalMomentum(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 252
    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        closes = pd.DataFrame(closes)
        R = (closes / closes.shift(100))
        out[:] = (R.T - R.T.mean()).T.mean()
class Curve(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 6
    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        out[:] = curve(closes)
class Div_Yield(CustomFactor):
    inputs = [Fundamentals.trailing_dividend_yield]; window_length = 12
    def compute(self, today, assets, out, d_y):
        d_y = nanfill(d_y)
        out[:] = d_y[-1]
class Downward(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 5
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        ratio_avg = (close[-1] / np.mean(close, axis=0))
        out[:] = ((close[-1] / close[0]) + ratio_avg)
class EBITPerEV(CustomFactor):
    inputs = [Fundamentals.ebit, Fundamentals.enterprise_value]; window_length = 12
    def compute(self, today, assets, out, ebit, ev):
        ebit = nanfill(ebit)
        ev = nanfill(ev)
        out[:] = ebit[-1] / ev[-1]
class Liquidity(CustomFactor):
    inputs = [USEquityPricing.volume, Fundamentals.shares_outstanding] ; window_length = 12
    def compute(self, today, assets, out, volume, shares):
        volume = nanfill(volume)
        shares = nanfill(shares)
        out[:] = volume[-1] / shares[-1]
class MACD(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 60
    def ema(self, data, window):      # Initial value for EMA is taken as trialing SMA
        import numpy as np
        c = 2.0 / (window + 1)
        ema = np.mean(data[-(2*window)+1:-window+1], axis=0)
        for value in data[-window+1:]:
            ema = (c * value) + ((1 - c) * ema)
        return ema
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        fema = self.ema(close, 12)
        sema = self.ema(close, 26)
        macd_line = fema - sema
        macd = []
        macd.insert(0, self.ema(close,12) - self.ema(close,26))
        for i in range(1,15, 1):
            macd.insert(0, self.ema(close[:-i],12) - self.ema(close[:-i],26))
        signal = self.ema(macd,9)
        out[:] = macd_line - signal
class MaxGap(CustomFactor): # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close] ; window_length = 90
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = max_gap
class MedianValue(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 42
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        out[:] = np.nanmedian(close, axis=0)
class Momentum(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 20
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        out[:] = close[-1] / close[0]
class Price_Earnings(CustomFactor):
    inputs = [Fundamentals.pe_ratio] ; window_length = 3
    def compute(self, today, assets, out, pe):
        pe = nanfill(pe)
        out[:] = pe[-1]
class Price_to_TTM_Cashflows(CustomFactor):
    inputs = [Fundamentals.pcf_ratio] ; window_length = 12
    def compute(self, today, assets, out, pcf):
        pcf = nanfill(pcf)
        out[:] = -pcf[-1]
class Price_to_TTM_Sales(CustomFactor):
    inputs = [Fundamentals.ps_ratio] ; window_length = 12
    def compute(self, today, assets, out, ps):
        ps = nanfill(ps)
        out[:] = -ps[-1]
class PriceChange(CustomFactor):      # Possible missed split in data
    inputs = [USEquityPricing.close] ; window_length = 2
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        out[:] = close[-1] / close[0]
class PriceToBook(CustomFactor):
    inputs = [Fundamentals.pb_ratio] ; window_length = 12
    def compute(self, today, assets, out, ptb):
        ptb = nanfill(ptb)
        out[:] = -ptb[-1]
class ProfitPerAssets(CustomFactor):
    inputs = [Fundamentals.gross_profit, Fundamentals.total_assets]; window_length = 12
    def compute(self, today, assets, out, gross_profit, total_assets):
        gross_profit = nanfill(gross_profit)
        total_assets = nanfill(total_assets)
        out[:] = gross_profit[-1] / total_assets[-1]
class Quality1(CustomFactor):
    inputs = [Fundamentals.gross_profit, Fundamentals.total_assets]; window_length = 12
    def compute(self, today, assets, out, gross_profit, total_assets):
        gross_profit = nanfill(gross_profit)
        total_assets = nanfill(total_assets)
        out[:] = gross_profit[-1] / total_assets[-1]
class Quality2(CustomFactor):
    inputs = [Fundamentals.gross_profit, Fundamentals.total_assets]
    window_length = 24
    def compute(self, today, assets, out, gross_profit, total_assets):
        norm = gross_profit / total_assets
        norm = nanfill(norm)
        out[:] = (norm[-1] - np.mean(norm, axis=0)) / np.std(norm, axis=0)
class Revenue(CustomFactor):
    inputs = [Fundamentals.total_revenue] ; window_length = 12
    def compute(self, today, assets, out, revenue):
        revenue = nanfill(revenue)
        out[:] = revenue[-1]
class ROE1(CustomFactor):
    inputs = [Fundamentals.roe] ; window_length = 77
    def compute(self, today, assets, out, roe):
        roe = nanfill(roe)
        out[:] = np.mean(roe[-5:], axis=0) - np.mean(roe, axis=0)
class ROE2(CustomFactor):
    inputs = [Fundamentals.roe] ; window_length = 11
    def compute(self, today, assets, out, roe):
        roe = nanfill(roe)
        out[:] = roe[-1]
class ROIC(CustomFactor):
    inputs = [Fundamentals.roic] ; window_length = 12
    def compute(self, today, assets, out, roic):
        roic = nanfill(roic)
        out[:] = roic[-1]
class Slope(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 10
    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        out[:] = slope(closes)
class Value1(CustomFactor):
    inputs = [Fundamentals.ebit, Fundamentals.enterprise_value]; window_length = 12
    def compute(self, today, assets, out, ebit, ev):
        ebit = nanfill(ebit)
        ev = nanfill(ev)
        out[:] = ebit[-1] / ev[-1]
class Value2(CustomFactor):
    inputs = [Fundamentals.book_value_yield,
              Fundamentals.sales_yield,
              Fundamentals.fcf_yield]
    window_length = 12
    def compute(self, today, assets, out, book_value, sales, fcf):
        book_value = nanfill(book_value)
        sales = nanfill(sales)
        fcf = nanfill(fcf)
        value_table = pd.DataFrame(index=assets)
        value_table['book_value'] = book_value[-1]
        value_table['sales']      = sales[-1]
        value_table['fcf']        = fcf[-1]
        out[:] = value_table.rank().mean(axis=1)
class Volatility1(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 252
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        close = pd.DataFrame(data=close, columns=assets)
        # Rank largest is best, need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()
class Volatility2(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 252
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        close = pd.DataFrame(data=close, columns=assets)
        # Rank largest is best, need to invert the sdev.
        out[:] = np.log(close).diff().std()
class Volatility3(CustomFactor):
    inputs = [USEquityPricing.close] ; window_length = 122
    def compute(self, today, assets, out, close):
        close = nanfill(close)
        # 6-month volatility, starting before the five-day mean reversion period
        daily_returns = np.log(close[1:-6]) - np.log(close[0:-7])
        out[:] = daily_returns.std(axis = 0)
class VolumeMinimum(CustomFactor):
    inputs = [USEquityPricing.volume] ; window_length = 42
    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.min(np.array(volume), axis=0)  #.astype(int)
class VolumeMin(CustomFactor):
    inputs = [USEquityPricing.volume] ; window_length = 42
    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.min(volume, axis=0)    # & VolumeMin().top(200)
class VolumeMax(CustomFactor):
    inputs = [USEquityPricing.volume] ; window_length = 42
    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.max(volume, axis=0)
class VolumeMean(CustomFactor):
    inputs = [USEquityPricing.volume] ; window_length = 42
    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.mean(volume, axis=0)


'''
Extracted from https://www.quantopian.com/help/fundamentals (count: 929)

accession_number    accession_number    accounts_payable    accounts_receivable    accretion_on_preferred_stock    accrued_interest_receivable    accrued_investment_income    accrued_liabilities_total    accrued_preferred_stock_dividends    accruedand_deferred_income    accruedand_deferred_income_current    accruedand_deferred_income_non_current    accumulated_depreciation    acquired_in_process_rn_d    acquired_in_process_rn_d_income    acquiredin_process_rn_d_income_banks    acquisition_expense    additional_paid_in_capital    adjusted_geography_segment_data    adjustmentsfor_undistributed_profitsof_associates    administrative_expense    advance_from_federal_home_loan_banks    advancesfrom_central_banks    agency_fees    agency_fees_and_commissions    allowance_for_doubtful_accounts_receivable    allowance_for_funds_construction    allowance_for_loans_and_lease_losses    allowance_for_notes_receivable    allowances_for_construction    amortization    amortization    amortization_of_deferred_acquisition_costs    amortization_of_financing_costs_and_discounts    amortization_of_intangibles    amortization_of_intangibles    amortization_of_securities    asset_impairment_charge    assets_held_for_sale    assets_of_discontinued_operations    assets_turnover    available_for_sale_securities    average_dilution_earn
bank_acceptance_executed_and_outstanding    bank_indebtedness    bank_loan    bank_loans_current    bank_loans_non_current    bank_loans_total    bank_owned_life_insurance    basic_accounting_change    basic_average_shares    basic_continuous_operations    basic_discontinuous_operations    basic_eps    basic_eps_other_gains_losses    basic_extraordinary    beginning_cash_position    book_value_per_share    book_value_yield    buildings_and_improvements    business_country_id    buy_back_yield
calls_maturities_of_maturity_securities    cannaics    cap_ex_reported    capital_expenditure    capital_lease_obligations    capital_stock    capitaln_business_taxes    cash    cash_advancesand_loans_madeto_other_parties    cash_and_cash_equivalents    cash_and_due_from_banks    cash_cash_equivalents_and_federal_funds_sold    cash_cash_equivalents_and_marketable_securities    cash_conversion_cycle    cash_dividends_paid    cash_equivalents    cash_flow_from_continuing_financing_activities    cash_flow_from_continuing_investing_activities    cash_flow_from_continuing_operating_activities    cash_flow_from_discontinued_operation    cash_flowsfromusedin_operating_activities_direct    cash_from_discontinued_financing_activities    cash_from_discontinued_investing_activities    cash_from_discontinued_operating_activities    cash_receiptsfrom_paymentsfor_financial_derivative_contracts    cash_receiptsfrom_repaymentof_advancesand_loans_madeto_other_parties    cash_return    cash_value_of_life_insurance    cashand_balanceswith_central_banks    casualty_claims    ceded_premiums    ceded_unearned_premiums    cf_yield    cfo_per_share    change_in_account_payable    change_in_accrued_expense    change_in_accrued_investment_income    change_in_deferred_acquisition_costs    change_in_deferred_charges    change_in_dividend_payable    change_in_federal_funds_and_securities_sold_for_repurchase    change_in_funds_withheld    change_in_income_tax_payable    change_in_interest_payable    change_in_inventory    change_in_loans    change_in_loss_and_loss_adjustment_expense_reserves    change_in_other_current_assets    change_in_other_current_liabilities    change_in_other_working_capital    change_in_payable    change_in_payables_and_accrued_expense    change_in_premiums_receivable    change_in_prepaid_assets    change_in_prepaid_reinsurance_premiums    change_in_receivables    change_in_reinsurance_receivable_on_paid_losses    change_in_reinsurance_recoverable_on_paid_and_unpaid_losses    change_in_reinsurance_recoverable_on_unpaid_losses    change_in_restricted_cash    change_in_tax_payable    change_in_trading_account_securities    change_in_unearned_premiums    change_in_unearned_premiums_ceded    change_in_working_capital    changein_accrued_income    changein_deferred_income    changein_insurance_contract_assets    changein_investment_contract    changein_reinsurance_receivables    changes_in_account_receivables    changes_in_cash    changesin_inventoriesof_finished_goodsand_workin_progress    cik    claims_outstanding    claimsand_paid_incurred    classesof_cash_payments    classesof_cash_receiptsfrom_operating_activities    com_tre_sha_num    commercial_loan    commercial_paper    commission_expenses    commission_revenue    common_equity_to_assets    common_stock    common_stock_dividend_paid    common_stock_equity    common_stock_issuance    common_stock_payments    common_stocks_available_for_sale    common_utility_plant    company_status    construction_grants    construction_in_progress    consumer_loan    contact_email    continuing_and_discontinued_basic_eps    continuing_and_discontinued_diluted_eps    convertible_loans_current    cost_of_revenue    country_id    credit_card    credit_losses_provision    credit_risk_provisions    cumulative_effect_of_accounting_change    cumulative_effect_of_accounting_change    currency_id    current_accrued_expenses    current_assets    current_capital_lease_obligation    current_debt    current_debt_and_capital_lease_obligation    current_deferred_assets    current_deferred_liabilities    current_deferred_revenue    current_deferred_taxes_assets    current_deferred_taxes_liabilities    current_liabilities    current_notes_payable    current_provisions    current_ratio    customer_acceptances    customer_accounts
days_in_inventory    days_in_payment    days_in_sales    debt_securities    debt_securitiesin_issue    debt_total    debtto_assets    decreasein_interest_bearing_depositsin_bank    deferred_acquisition_costs    deferred_assets    deferred_cost_current    deferred_costs    deferred_financing_costs    deferred_income_tax    deferred_policy_acquisition_costs    deferred_tax    deferred_tax_assets    deferred_tax_liabilities_total    defined_pension_benefit    depletion    depletion    depositary_receipt_ratio    deposits_madeunder_assumed_reinsurance_contract    deposits_receivedunder_ceded_insurance_contract    depositsby_bank    depreciation    depreciation    depreciation_amortization_depletion    depreciation_amortization_depletion    depreciation_and_amortization    depreciation_and_amortization    derivative_assets    derivative_product_liabilities    development_expense    diluted_accounting_change    diluted_average_shares    diluted_cont_eps_growth    diluted_continuous_operations    diluted_discontinuous_operations    diluted_eps    diluted_eps_growth    diluted_eps_other_gains_losses    diluted_extraordinary    distribution_costs    dividend_income    dividend_paid_cfo    dividend_per_share    dividend_rate    dividend_received_cfo    dividend_yield    dividends_paid_direct    dividends_payable    dividends_received_cfi    dividends_received_direct    domestic_sales    dps_growth
earning_loss_of_equity_investments    earning_yield    earnings_from_equity_interest    earnings_losses_from_equity_investments    earningsfrom_equity_interest_net_of_tax    ebit    ebit_margin    ebitda    ebitda_margin    effect_of_exchange_rate_changes    electric_revenue    electric_utility_plant    employee_benefits    end_cash_position    enterprise_value    equipment    equity_attributable_to_owners_of_parent    equity_investments    equity_per_share_growth    equity_shares_investments    esop_debt_guarantee    ev_to_ebitda    exceptional_items    excess_tax_benefit_from_stock_based_compensation    exchange_id    excise_taxes    exploration_development_and_mineral_property_lease_expenses    extraordinary_items
facilities_and_other    fcf_per_share    fcf_ratio    fcf_yield    federal_funds_purchased    federal_funds_purchased_and_securities_sold_under_agreement_to_repurchase    federal_funds_sold    federal_funds_sold_and_securities_purchase_under_agreements_to_resell    federal_home_loan_bank_stock    fee_revenue_and_other_income    fees    fees_and_commissions    feesand_commission_expense    feesand_commission_income    file_date    file_date    finance_lease_receivables_current    finance_lease_receivables_non_current    financial_assets    financial_assets_designatedas_fair_value_through_profitor_loss_total    financial_health_grade    financial_instruments_sold_under_agreements_to_repurchase    financial_leverage    financial_liabilities_current    financial_liabilities_designatedas_fair_value_through_profitor_loss_total    financial_liabilities_measuredat_amortized_cost_total    financial_liabilities_non_current    financing_cash_flow    finished_goods    fiscal_year_end    fix_assets_turonver    fixed_maturities_available_for_sale    fixed_maturities_held_to_maturity    fixed_maturities_trading    fixed_maturity_investments    flight_fleet_vehicle_and_related_equipments    foreclosed_assets    foreign_component    foreign_currency_translation_adjustments    foreign_exchange_trading_gains    foreign_sales    form_type    form_type    forward_dividend_yield    forward_earning_yield    forward_pe_ratio    free_cash_flow    fuel    fuel_and_natural_gas    fuel_and_purchase_power    future_policy_benefits
gain_loss_on_investment_securities    gain_loss_on_sale_of_business    gain_loss_on_sale_of_ppe    gain_losson_derecognitionof_available_for_sale_financial_assets    gain_losson_derecognitionof_non_current_assets_not_heldfor_sale_total    gain_losson_financial_instruments_designatedas_cash_flow_hedges    gain_losson_saleof_assets    gain_on_sale_of_business    gain_on_sale_of_ppe    gain_on_sale_of_security    gainon_extinguishmentof_debt    gainon_investment_properties    gainon_redemptionand_extinguishmentof_debt    gainon_saleof_investment_property    gainon_saleof_loans    gains_loss_on_disposal_of_discontinued_operations    gains_losses_not_affecting_retained_earnings    gas_revenue    general_account_assets    general_and_administrative_expense    general_expense    general_partnership_capital    goodwill    goodwill_and_other_intangible_assets    gross_accounts_receivable    gross_dividend_payment    gross_loan    gross_margin    gross_notes_receivable    gross_ppe    gross_premiums_written    gross_profit    growth_grade    growth_score    guaranteed_investment_contract
headquarter_address_line1    headquarter_address_line2    headquarter_address_line3    headquarter_address_line4    headquarter_city    headquarter_country    headquarter_fax    headquarter_homepage    headquarter_phone    headquarter_postal_code    headquarter_province    hedging_assets_current    hedging_assets_non_current    hedging_liabilities_current    hedging_liabilities_non_current    held_to_maturity_securities
impairment_loss_reversal_recognizedin_profitor_loss    impairment_losses_reversals_financial_instruments_net    impairment_of_capital_assets    impairmentof_capital_assets_income    income_tax_paid_supplemental_data    income_tax_payable    income_taxes_refund_paid_cff    income_taxes_refund_paid_cfi    incomefrom_associatesand_other_participating_interests    incomefrom_sharesin_subsidiaries_group_undertakings    increase_decrease_in_deposit    increase_decrease_in_net_unearned_premium_reserves    increase_decreasein_lease_financing    increasein_interest_bearing_depositsin_bank    increasein_lease_financing    industry_template_code    insurance_and_claims    insurance_and_premiums    insurance_contract_assets    insurance_contract_liabilities    insurance_funds_non_current    insurance_payables    insurance_receivables    interest_bearing_borrowings_current    interest_bearing_borrowings_non_current    interest_bearing_borrowings_total    interest_bearing_deposits_assets    interest_bearing_deposits_liabilities    interest_coverage    interest_credited_on_policyholder_deposits    interest_expense    interest_expense_for_capitalized_lease_obligations    interest_expense_for_deposit    interest_expense_for_federal_funds_sold_and_securities_purchase_under_agreements_to_resell    interest_expense_for_long_term_debt    interest_expense_for_long_term_debt_and_capital_securities    interest_expense_for_short_term_debt    interest_expense_non_operating    interest_expense_operating    interest_income    interest_income_after_provision_for_loan_loss    interest_income_from_deposits    interest_income_from_federal_funds_sold_and_securities_purchase_under_agreements_to_resell    interest_income_from_interest_bearing_deposits    interest_income_from_investment_securities    interest_income_from_leases    interest_income_from_loans    interest_income_from_loans_and_lease    interest_income_from_other_money_market_investments    interest_income_from_securities    interest_income_from_trading_account_securities    interest_income_non_operating    interest_income_operating    interest_income_other_operating_income    interest_paid_cff    interest_paid_cfo    interest_paid_direct    interest_paid_supplemental_data    interest_payable    interest_received_cfi    interest_received_cfo    interest_received_direct    interestand_similar_income    inventories_adjustments_allowances    inventory    inventory_turnover    invested_capital    investing_cash_flow    investment_banking_profit    investment_contract_liabilities    investment_id    investment_properties    investment_tax_credits    investmentin_financial_assets    investments_and_advances    investments_in_affiliates_subsidiaries_associates_and_joint_ventures    investments_in_other_ventures_under_equity_method    investments_in_variable_interest_entity    investmentsin_associatesat_cost    investmentsin_joint_venturesat_cost    investmentsin_subsidiariesat_cost    ipo_date    is_depositary_receipt    is_direct_invest    is_dividend_reinvest    is_primary_share    issuance_of_capital_stock    issuance_of_debt    issuance_paymentof_other_equity_instruments_net    issue_expenses
land_and_improvements    leases    legal_name    legal_name_language_code    liabilities_heldfor_sale_current    liabilities_heldfor_sale_non_current    liabilities_heldfor_sale_total    liabilities_of_discontinued_operations    life_annuity_premiums    limited_partnership    limited_partnership_capital    line_of_credit    loan_capital    loans_held_for_resell    loans_held_for_sale    loans_receivable    loansand_advancesto_bank    loansand_advancesto_customer    long_term_capital_lease_obligation    long_term_contracts    long_term_debt    long_term_debt_and_capital_lease_obligation    long_term_debt_equity_ratio    long_term_debt_issuance    long_term_debt_payments    long_term_debt_total_capital_ratio    long_term_investments    long_term_provisions    loss_adjustment_expense    loss_and_loss_adjustment_expected_incurred    losson_extinguishmentof_debt
machinery_furniture_equipment    maintenance_and_repairs    market_cap    marketing_expense    materials_and_supplies    mineral_properties    minimum_pension_liabilities    minority_interest    minority_interest    minority_interests    misc_other_special_charges    miscellaneous_other_operating_income    money_market_investments    morningstar_economy_sphere_code    morningstar_industry_code    morningstar_industry_group_code    morningstar_sector_code    mortgage_and_consumerloans    mortgage_loan
nace    naics    natural_gas_fuel_and_other    natural_resource_assets    negative_goodwill_immediately_recognized    net_assets    net_business_purchase_and_sale    net_capital_expenditure_disposals    net_common_stock_issuance    net_debt    net_foreign_currency_exchange_gain_loss    net_foreign_exchange_gain_loss    net_income    net_income    net_income_common_stockholders    net_income_cont_ops_growth    net_income_continuous_operations    net_income_discontinuous_operations    net_income_extraordinary    net_income_from_continuing_and_discontinued_operation    net_income_from_continuing_operation_net_minority_interest    net_income_from_continuing_operations    net_income_from_other_gains_losses    net_income_from_tax_loss_carryforward    net_income_growth    net_income_including_noncontrolling_interests    net_intangibles_purchase_and_sale    net_interest_income    net_investment_income    net_investment_purchase_and_sale    net_issuance_payments_of_debt    net_loan    net_long_term_debt_issuance    net_margin    net_non_operating_interest_income_expense    net_occupancy_expense    net_operating_interest_income_expense    net_other_financing_charges    net_other_investing_changes    net_other_unrealized_gain_loss    net_outward_loans    net_policyholder_benefits_and_claims    net_ppe    net_ppe_purchase_and_sale    net_preferred_stock_issuance    net_premiums_written    net_proceeds_payment_for_loan    net_realized_gain_loss_on_investments    net_short_term_debt_issuance    net_tangible_assets    net_technology_purchase_and_sale    net_unrealized_gain_loss_foreign_currency    net_unrealized_gain_loss_investments    net_utility_plant    non_current_accounts_receivable    non_current_accrued_expenses    non_current_deferred_assets    non_current_deferred_liabilities    non_current_deferred_revenue    non_current_deferred_taxes_assets    non_current_deferred_taxes_liabilities    non_current_note_receivables    non_current_pension_and_other_postretirement_benefit_plans    non_current_prepaid_assets    non_interest_bearing_borrowings_current    non_interest_bearing_borrowings_non_current    non_interest_bearing_deposits    non_interest_expense    non_interest_income    non_operating_expenses    non_operating_income    non_recurring_operation_expense    normalized_basic_eps    normalized_diluted_eps    normalized_income    normalized_net_profit_margin    notes_receivable    occupancy_and_equipment    operating_cash_flow    operating_expense    operating_gains_losses    operating_income    operating_revenue    operating_taxesn_licenses
operation_and_maintenance    operation_income_growth    operation_margin    operation_revenue_growth3_month_avg    ordinary_shares_number    other_adjustmentsfor_which_cash_effects_are_investingor_financing_cash_flow    other_assets    other_capital_stock    other_cash_paymentsfrom_operating_activities    other_cash_receiptsfrom_operating_activities    other_current_assets    other_current_borrowings    other_current_liabilities    other_customer_services    other_deferred_costs    other_deposits    other_equity_adjustments    other_equity_interest    other_financing    other_gain_loss_from_disposition_of_discontinued_operations    other_impairment_of_capital_assets    other_income_expense    other_intangible_assets    other_interest_earning_assets    other_interest_expense    other_interest_income    other_inventories    other_invested_assets    other_liabilities    other_loan_assets    other_loans_current    other_loans_non_current    other_loans_total    other_non_cash_items    other_non_current_assets    other_non_current_liabilities    other_non_interest_expense    other_non_interest_income    other_non_operating_expenses    other_non_operating_income    other_non_operating_income_expenses    other_operating_expenses    other_operating_income_total    other_operating_inflows_outflowsof_cash    other_operating_revenue    other_payable    other_properties    other_real_estate_owned    other_receivables    other_reserves    other_short_term_investments    other_special_charges    other_staff_costs    other_taxes    other_write_down    other_write_off    otherunder_preferred_stock_dividend
participating_policyholder_equity    patents    payables    payables_and_accrued_expenses    payment_for_loans    payment_turnover    paymentof_bills    paymentsfor_premiumsand_claims_annuitiesand_other_policy_benefits    paymentsof_other_equity_instruments    paymentson_behalfof_employees    paymentsto_acquire_held_to_maturity_investments    paymentsto_suppliersfor_goodsand_services    payout_ratio    pb_ratio    pcf_ratio    pe_ratio    peg_payback    peg_ratio    pension_and_employee_benefit_expense    pension_and_other_postretirement_benefit_plans_total    pension_costs    pensionand_other_post_retirement_benefit_plans_current    period_ending_date    period_ending_date    placementwith_banksand_other_financial_institutions    policy_acquisition_expense    policy_fees    policy_loans    policy_reserves_benefits    policyholder_and_reinsurer_accounts    policyholder_benefits_ceded    policyholder_benefits_gross    policyholder_dividends    policyholder_funds    policyholder_interest    pre_tre_sha_num    preferred_securities_outside_stock_equity    preferred_shares_number    preferred_stock    preferred_stock_dividend_paid    preferred_stock_dividends    preferred_stock_equity    preferred_stock_issuance    preferred_stock_of_subsidiary    preferred_stock_payments    preferred_stocks_available_for_sale    premium_taxes_credit    premiums_receivable    prepaid_assets    prepaid_reinsurance_premiums    pretax_income    pretax_margin    primary_exchange_id    primary_share_class_id    primary_symbol    principle_investment_gain_loss    principle_transaction_revenue    proceeds_from_issuance_of_warrants    proceeds_from_loans    proceeds_from_stock_option_exercised    proceeds_payment_federal_funds_sold_and_securities_purchased_under_agreement_to_resell    proceeds_payment_in_interest_bearing_deposits_in_bank    proceedsfrom_disposalof_held_to_maturity_investments    proceedsfrom_government_grants_cff    proceedsfrom_government_grants_cfi    proceedsfrom_issuing_other_equity_instruments    professional_expense_and_contract_services_expense    profitability_grade    profiton_disposals    promotion_and_advertising    properties    property_casualty_premiums    property_liability_insurance_claims    provision_for_doubtful_accounts    provision_for_gain_loss_on_disposal    provision_for_loan_lease_and_other_losses    provisionand_write_offof_assets    provisions_total    ps_ratio    purchase_of_business    purchase_of_equity_securities    purchase_of_fixed_maturity_securities    purchase_of_intangibles    purchase_of_investment    purchase_of_long_term_investments    purchase_of_ppe    purchase_of_short_term_investments    purchase_of_technology    purchased_components    purchased_transportation_services    purchaseof_joint_venture_associate    purchaseof_subsidiaries
quick_ratio
raw_materials    real_estate    real_estate_and_real_estate_joint_ventures_held_for_investment    real_estate_held_for_sale    realized_capital_gain    realized_gain_loss_on_sale_of_loans_and_lease    receiptsfrom_customers    receivable_turnover    receivables    receivables_adjustments_allowances    reconciled_cost_of_revenue    reconciled_depreciation    redeemable_preferred_stock    regulatory_assets    regulatory_liabilities    reinsurance_assets    reinsurance_balances_payable    reinsurance_receivables    reinsurance_recoverable    reinsurance_recoverable_for_paid_losses    reinsurance_recoverable_for_unpaid_losses    reinsurance_recoveries_claimsand_benefits    reinsurance_shareof_insurance_contract    rent_and_landing_fees    reorganization_other_costs    repayment_of_debt    repaymentin_lease_financing    repurchase_of_capital_stock    research_and_development    research_expense    restricted_cash    restricted_cash_and_cash_equivalents    restricted_cash_and_investments    restricted_common_stock    restricted_investments    restructring_and_mn_a_income    restructuring_and_merger_and_acquisition_income    restructuring_and_mergern_acquisition    retained_earnings    revenue_growth    revenues_cargo    revenues_passenger    roa    roe    roic
salaries_and_wages    sale_of_business    sale_of_intangibles    sale_of_investment    sale_of_long_term_investments    sale_of_ppe    sale_of_short_term_investments    sale_of_technology    saleof_joint_venture_associate    saleof_subsidiaries    sales_of_equity_securities    sales_of_fixed_maturity_securities    sales_per_employee    sales_per_share    sales_yield    securities_activities    securities_amortization    securities_and_investments    securities_lending_collateral    securities_lending_payable    securities_loaned    security_agree_to_be_resell    security_borrowed    security_sold_not_yet_repurchased    security_type    selling_and_marketing_expense    selling_expense    selling_general_and_administration    separate_account_assets    separate_account_business    service_charge_on_depositor_accounts    share_based_payments    share_class_description    share_class_level_shares_outstanding    share_class_status    share_issued    shareof_associates    shareof_operating_profit_lossfrom_joint_ventures_and_associates    shares_outstanding    short_description    short_name    short_term_debt_issuance    short_term_debt_payments    short_term_investments_available_for_sale    short_term_investments_held_to_maturity    short_term_investments_trading    sic    size_score    social_security_costs    special_charge    special_income    special_income_charges    staff_costs    standard_name    stock_based_compensation    stock_type    stockholders_equity    student_loan    style_box    style_score    subordinated_liabilities    sustainable_growth_rate    symbol
tangible_book_value    tangible_book_value_per_share    tangible_bv_per_share3_yr_avg    tangible_bv_per_share5_yr_avg    tax_assets_total    tax_effect_of_unusual_items    tax_loss_carryforward_basic_eps    tax_loss_carryforward_diluted_eps    tax_provision    tax_rate    tax_rate_for_calcs    taxes_assets_current    taxes_receivable    taxes_refund_paid    taxes_refund_paid_direct    time_deposits_placed    time_deposits_placed    total_adjustmentsfor_non_cash_items    total_assets    total_capitalization    total_debt    total_debt_equity_ratio    total_deferred_credits_and_other_non_current_liabilities    total_deposits    total_employee_number    total_equity    total_equity_gross_minority_interest    total_expenses    total_investments    total_liabilities    total_liabilities_net_minority_interest    total_money_market_investments    total_non_current_assets    total_non_current_liabilities    total_non_current_liabilities_net_minority_interest    total_other_finance_cost    total_partnership_capital    total_premiums_earned    total_revenue    total_tax_payable    total_unusual_items    total_unusual_items_excluding_goodwill    total_yield    tradeand_other_payables_non_current    trading_and_other_receivable    trading_assets    trading_gain_loss    trading_liabilities    trading_securities    tradingand_financial_liabilities    transportation_revenue    treasury_shares_number    treasury_stock    trust_feesby_commissions    trust_preferred_securities
unbilled_receivables    unclassified_current_assets    underwriting_expenses    unearned_income    unearned_premiums    unpaid_loss_and_loss_reserve    unrealized_gain_loss    unrealized_gain_loss_on_investment_securities    unrealized_gains_losses_on_derivatives
value_score
wagesand_salaries    water_production    work_in_process    work_performedby_entityand_capitalized    working_capital    working_capital_per_share    working_capital_per_share3_yr_avg    working_capital_per_share5_yr_avg    write_down    write_off
'''

def pvr(context, data):
    ''' Custom chart and/or logging of profit_vs_risk returns and related information
        http://quantopian.com/posts/pvr
    '''
    import time
    from datetime import datetime as _dt
    from pytz import timezone      # Python will only do once, makes this portable.
                                   #   Move to top of algo for better efficiency.
    c = context  # Brevity is the soul of wit -- Shakespeare [for readability]
    if 'pvr' not in c:

        # For real money, you can modify this to total cash input minus any withdrawals
        manual_cash = c.portfolio.starting_cash
        time_zone   = 'US/Pacific'   # Optionally change to your own time zone for wall clock time

        c.pvr = {
            'options': {
                # # # # # # # # # #  Options  # # # # # # # # # #
                'logging'         : 0,    # Info to logging window with some new maximums
                'log_summary'     : 126,  # Summary every x days. 252/yr

                'record_pvr'      : 1,    # Profit vs Risk returns (percentage)
                'record_pvrp'     : 0,    # PvR (p)roportional neg cash vs portfolio value
                'record_cash'     : 0,    # Cash available
                'record_max_lvrg' : 1,    # Maximum leverage encountered
                'record_max_risk' : 1,    # Highest risk overall
                'record_shorting' : 0,    # Total value of any shorts
                'record_max_shrt' : 1,    # Max value of shorting total
                'record_cash_low' : 1,    # Any new lowest cash level
                'record_q_return' : 0,    # Quantopian returns (percentage)
                'record_pnl'      : 0,    # Profit-n-Loss
                'record_risk'     : 0,    # Risked, max cash spent or shorts beyond longs+cash
                'record_leverage' : 0,    # End of day leverage (context.account.leverage)
                # All records are end-of-day or the last data sent to chart during any day.
                # The way the chart operates, only the last value of the day will be seen.
                # # # # # # # # #  End options  # # # # # # # # #
            },
            'pvr'        : 0,      # Profit vs Risk returns based on maximum spent
            'cagr'       : 0,
            'max_lvrg'   : 0,
            'max_shrt'   : 0,
            'max_risk'   : 0,
            'days'       : 0.0,
            'date_prv'   : '',
            'date_end'   : get_environment('end').date(),
            'cash_low'   : manual_cash,
            'cash'       : manual_cash,
            'start'      : manual_cash,
            'tz'         : time_zone,
            'begin'      : time.time(),  # For run time
            'run_str'    : '{} to {}  ${}  {} {}'.format(get_environment('start').date(), get_environment('end').date(), int(manual_cash), _dt.now(timezone(time_zone)).strftime("%Y-%m-%d %H:%M"), time_zone)
        }
        if c.pvr['options']['record_pvrp']: c.pvr['options']['record_pvr'] = 0 # if pvrp is active, straight pvr is off
        if get_environment('arena') not in ['backtest', 'live']: c.pvr['options']['log_summary'] = 1 # Every day when real money
        log.info(c.pvr['run_str'])
    p = c.pvr ; o = c.pvr['options'] ; pf = c.portfolio ; pnl = pf.portfolio_value - p['start']
    def _pvr(c):
        p['cagr'] = ((pf.portfolio_value / p['start']) ** (1 / (p['days'] / 252.))) - 1
        ptype = 'PvR' if o['record_pvr'] else 'PvRp'
        log.info('{} {} %/day   cagr {}   Portfolio value {}   PnL {}'.format(ptype, '%.4f' % (p['pvr'] / p['days']), '%.3f' % p['cagr'], '%.0f' % pf.portfolio_value, '%.0f' % pnl))
        log.info('  Profited {} on {} activated/transacted for PvR of {}%'.format('%.0f' % pnl, '%.0f' % p['max_risk'], '%.1f' % p['pvr']))
        log.info('  QRet {} PvR {} CshLw {} MxLv {} MxRisk {} MxShrt {}'.format('%.2f' % (100 * pf.returns), '%.2f' % p['pvr'], '%.0f' % p['cash_low'], '%.2f' % p['max_lvrg'], '%.0f' % p['max_risk'], '%.0f' % p['max_shrt']))
    def _minut():
        dt = get_datetime().astimezone(timezone(p['tz']))
        return str((dt.hour * 60) + dt.minute - 570).rjust(3)  # (-570 = 9:31a)
    date = get_datetime().date()
    if p['date_prv'] != date:
        p['date_prv'] = date
        p['days'] += 1.0
    do_summary = 0
    if o['log_summary'] and p['days'] % o['log_summary'] == 0 and _minut() == '100':
        do_summary = 1              # Log summary every x days
    if do_summary or date == p['date_end']:
        p['cash'] = pf.cash
    elif p['cash'] == pf.cash and not o['logging']: return  # for speed

    shorts = sum([z.amount * z.last_sale_price for s, z in pf.positions.items() if z.amount < 0])
    new_key_hi = 0                  # To trigger logging if on.
    cash       = pf.cash
    cash_dip   = int(max(0, p['start'] - cash))
    risk       = int(max(cash_dip, -shorts))

    if o['record_pvrp'] and cash < 0:   # Let negative cash ding less when portfolio is up.
        cash_dip = int(max(0, cash_dip * p['start'] / pf.portfolio_value))
        # Imagine: Start with 10, grows to 1000, goes negative to -10, should not be 200% risk.

    if int(cash) < p['cash_low']:             # New cash low
        new_key_hi = 1
        p['cash_low'] = int(cash)             # Lowest cash level hit
        if o['record_cash_low']: record(CashLow = p['cash_low'])

    if c.account.leverage > p['max_lvrg']:
        new_key_hi = 1
        p['max_lvrg'] = c.account.leverage    # Maximum intraday leverage
        if o['record_max_lvrg']: record(MxLv    = p['max_lvrg'])

    if shorts < p['max_shrt']:
        new_key_hi = 1
        p['max_shrt'] = shorts                # Maximum shorts value
        if o['record_max_shrt']: record(MxShrt  = p['max_shrt'])

    if risk > p['max_risk']:
        new_key_hi = 1
        p['max_risk'] = risk                  # Highest risk overall
        if o['record_max_risk']:  record(MxRisk = p['max_risk'])

    # Profit_vs_Risk returns based on max amount actually invested, long or short
    if p['max_risk'] != 0: # Avoid zero-divide
        p['pvr'] = 100 * pnl / p['max_risk']
        ptype = 'PvRp' if o['record_pvrp'] else 'PvR'
        if o['record_pvr'] or o['record_pvrp']: record(**{ptype: p['pvr']})

    if o['record_shorting']: record(Shorts = shorts)             # Shorts value as a positve
    if o['record_leverage']: record(Lv     = c.account.leverage) # Leverage
    if o['record_cash']    : record(Cash   = cash)               # Cash
    if o['record_risk']    : record(Risk   = risk)  # Amount in play, maximum of shorts or cash used
    if o['record_q_return']: record(QRet   = 100 * pf.returns)
    if o['record_pnl']     : record(PnL    = pnl)                # Profit|Loss

    if o['logging'] and new_key_hi:
        log.info('{}{}{}{}{}{}{}{}{}{}{}{}'.format(_minut(),
            ' Lv '     + '%.1f' % c.account.leverage,
            ' MxLv '   + '%.2f' % p['max_lvrg'],
            ' QRet '   + '%.1f' % (100 * pf.returns),
            ' PvR '    + '%.1f' % p['pvr'],
            ' PnL '    + '%.0f' % pnl,
            ' Cash '   + '%.0f' % cash,
            ' CshLw '  + '%.0f' % p['cash_low'],
            ' Shrt '   + '%.0f' % shorts,
            ' MxShrt ' + '%.0f' % p['max_shrt'],
            ' Risk '   + '%.0f' % risk,
            ' MxRisk ' + '%.0f' % p['max_risk']
        ))
    if do_summary: _pvr(c)
    if get_datetime() == get_environment('end'):   # Summary at end of run
        _pvr(c) ; elapsed = (time.time() - p['begin']) / 60  # minutes
        log.info( '{}\nRuntime {} hr {} min'.format(p['run_str'], int(elapsed / 60), '%.1f' % (elapsed % 60)))