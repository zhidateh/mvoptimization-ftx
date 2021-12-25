from algo   import MVOptimization
from ftx    import FtxClient

import pandas as pd


ftx_api_key     = "SECRET"
ftx_api_secret  = "SECRET"
ftx_subacc      = None
market_name     = ['BTC-PERP','ETH-PERP','ADA-PERP']
start_time      = '2021-10-01T00:00:00+00:00'
end_time        = '2021-10-31T23:00:00+00:00'
intervals       =  60 * 60 # 1hour in seconds

mv      = MVOptimization()
client  = FtxClient(ftx_api_secret,ftx_api_secret,ftx_subacc)

for m in market_name:
    history = client.get_candle(m,intervals,start_time,end_time)
    df = pd.DataFrame(history)
    mv.addAsset(m, df.close.to_list())

output = mv.computeOptimalPortfolio()

mv.plotEfficientFrontier(show=False, save="test.png")
mv.saveOutput(output, 'test.json')