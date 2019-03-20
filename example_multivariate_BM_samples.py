import matplotlib.pyplot as plt 
import aux_math_fin as mf
import numpy as np

vol=0.15 # annual volatility
rfRate=0.04  # annual risk-free rate (non-instantaneous, as quoted in the markets)
corr=0.5
nBusDays=250

def main():
    dailyStep = 1.0/nBusDays
    instAnnDrift=np.log1p(rfRate)
    covar=mf. make_covar_mat(sdev_vec=np.array([vol, vol]), cross_corr=corr)
    mu=np.array([instAnnDrift, 2*instAnnDrift])
    nSteps=20*nBusDays
    
    print("Annualized covariance: \n" + np.array2string(covar))
    print("Annualized instantaneous drift: \n" + np.array2string(mu))
    
    
    prices=mf.sample_mvBM(
        start_val=np.array([1.0,2.0]), 
        mu=mu, 
        covar=covar,
        stepSize=dailyStep, 
        nSteps=nSteps)


    print("Start value: \n" + np.array2string(prices[:,0]))

    N=np.shape(prices)[1]
    increments = prices[:,1:N] - prices[:,0:N-1]
    
    print("Estimated annualized covariance: \n" + np.array2string(np.cov(increments)*nBusDays))
    print("Estimated annualized instantaneous drift: \n" + np.array2string(np.mean(increments, axis=1, keepdims=True)*nBusDays))
    
    plt.scatter(x=prices[0,:], y=prices[1,:])
    plt.show()
    
main()
