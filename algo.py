import json
import pandas               as pd
import numpy                as np
import matplotlib.pyplot    as plt
import cvxopt               as opt
from cvxopt                 import blas, solvers

# silent solver
solvers.options['show_progress'] = False


class MVOptimization:

    def __init__(self) -> None:
        self.assets = {}
        self.weights = []
        self.returns = []
        self.risks = []

    def getDataFrame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.assets)
        df = df.pct_change().dropna().reset_index(drop=True)
        return df

    def addAsset(self, name:str , history:list) -> None:
        self.assets[name] = history
        
    def getAssets(self) -> list:
        return list(self.assets.keys())

    def generatePortfolios(self, number: int=5) -> pd.DataFrame:
        np.random.seed(1)

        num_assets = len(self.assets)

        port_returns = []
        port_volatility = []
        port_weights = []
            
        for _ in range(number):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)

            data = self.getDataFrame()
            ret, cov = data.mean(axis=0), data.cov()

            returns = np.dot(weights,ret)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

            port_returns.append(returns)
            port_volatility.append(volatility)
            port_weights.append(weights)
        
        port_weights = np.array(port_weights)
        dct = {"Returns":port_returns,"Volatility":port_volatility}
        for asset,i in zip(self.assets,range(num_assets)):
            dct[asset + " Weight"] = port_weights[:,i]

        return pd.DataFrame(dct)

    def computeOptimalPortfolio(self) -> dict:
        data = self.getDataFrame()
        n = data.shape[1]
        returns = np.transpose(data.values)
        
        N = 100
        mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

        # Convert to cvxopt matrices
        S = opt.matrix(np.cov(returns))
        pbar = opt.matrix(np.mean(returns, axis=1))

        # Create constraint matrices
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)

        # Calculate efficient frontier weights using quadratic programming
        portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                    for mu in mus]
        ## CALCULATE RISKS AND RETURNS FOR FRONTIER
        returns = [blas.dot(pbar, x) for x in portfolios]
        risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
        ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
        m1 = np.polyfit(returns, risks, 2)
        x1 = np.sqrt(m1[2] / m1[0])
        # CALCULATE THE OPTIMAL PORTFOLIO
        wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
        self.weights = np.asarray(wt).round(decimals=4).flatten()
        self.returns = returns
        self.risks = risks

        print(f"The median expected return is {round(np.median(self.returns)*100,3)}%")
        print(f"The median variance would be {round(np.median(self.risks),3)}")
        print("The weight distribution of the portfolio with the best returns would be:")
        [print(f"The weight of {asset} is {self.weights[idx]}.") for idx, asset in enumerate(self.getAssets())]

        return { asset: self.weights[idx] for idx, asset in enumerate(self.getAssets()) }


    def plotEfficientFrontier(self, show:bool =True, save:str ="") -> None:
        portfolios = self.generatePortfolios(500)
        print(portfolios.head())
        fig = portfolios.plot.scatter(x='Volatility',y='Returns').get_figure()
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        if show: 
            plt.show()
        if(len(save)):
            fig.savefig(save)

    def saveOutput(self, dct: dict, fpath: str) -> None:
        with open(fpath, "w") as output_file:
            json.dump(dct, output_file)


if __name__ == '__main__':
    mv = MVOptimization()
    mv.addAsset("test", [1,2,3,4,5])
    mv.addAsset("test1", [3,7,12,2,4])
    mv.addAsset("test2", [4,8,4,20,2])

    output = mv.computeOptimalPortfolio()

    mv.plotEfficientFrontier(show=True, save="test.png")
    mv.saveOutput(output, 'test.json')