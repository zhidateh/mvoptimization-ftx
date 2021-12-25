# Mean-Variance Portfolio Optimization with FTX API

## Abstract
Generate a vector of weights for a portfolio using `mean-variance portfolio optimzation`. The
program leverages free-to-use FTX API (see https://docs.ftx.com/?python#rest-api ) to pull the portfolio data for computation.

----
### Prerequisite
* Tested in Ubuntu20 with Python 3.7.12
* Python packages listed in `requirements.txt`:
  * ciso8601 - Used by example FTX client script for parsing datetime
  * cvxopt
  * matplotlib
  * numpy
  * pandas
  * requests
  * urllib3

### Run & Test
````batch
# update ftx_api_key, ftx_api_secret, ftx_subacc in main.py

> python main.py
````

### Output

* `efficient frontier plot` - Optional, saved as "test.png" or equivalent.
* `portfolio weightages` - Optional, saved as "test.json" or equivalent.

