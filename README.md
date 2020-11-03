# Quantopian_Algos
This repository consists of several python based Algos for algorythmic trading formerly on the Quantopian platform.

As Quantopian is shutting down its Quantopian community platform, Quantopian are allowing to download code (algorithms and notebooks) to continue on a local machine more easily. This repository consists of several algos which Quantopian has allowed to download.

## Algo Results in the header
You will find that some of these algos have a funny stat of their name. Theses are the results from backtesting them including slippage and commisssion over a larger timeframe.

Result header:
* R = Return,    eg R195 = 195%
* S = Sharpe,    eg S1.19 = Sharpe of 1.19
* DD = Drawdown, eg DD13.15 = -13.5%

Slippage / Commission used:
* set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
* set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))

## Other contributions
https://quantopian-archive.netlify.app/
https://archive.org/details/quantopian-archive
https://archive.org/download/quantopian-archive

## Next Steps
* Please contact me if you would like to get in touch to share and exchange trading ideas!
