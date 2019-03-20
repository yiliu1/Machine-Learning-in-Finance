# Math finance module

import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky



# Covariance matrix for given volatilities and cross-correlation:

def make_covar_mat(sdev_vec, cross_corr):
    d = len(sdev_vec)
    out = np.multiply(cross_corr,np.ones([d, d]))
    for i in range(d):
        for j in range(d):
            out[i,j]=sdev_vec[i]*sdev_vec[j]*(1.0 if i==j else cross_corr)
    return out


# Sample Black-Scholes model on a daily grid, parametrized by annual mean and annual vol
def daily_sample_prices_array(s0, instAnnDrift, annVol, nDays, nBusDays=250):
    
    return sample_GBM(startVal=s0, 
                      mu=instAnnDrift,
                      sigma=annVol,
                      stepSize=1.0/nBusDays,
                      nSteps=nDays)

# Sample Geometric Brownian Motion on a discrete grid with arbitrary step size
def sample_GBM(startVal, mu, sigma, stepSize, nSteps):
    muPerStep=(mu-0.5*sigma*sigma)*stepSize 
    sigmaPerStep=sigma*np.sqrt(stepSize)
    # returns at times t_1,...,nSteps
    logReturns=np.random.normal(loc=muPerStep,scale=sigmaPerStep,size=nSteps) 
    # log-performance starts at 0.0 on day 0 (exponentiation turns the start value 0.0 into 1.0 afterwards)
    logPerformance=np.cumsum(np.concatenate((np.zeros(1), logReturns), axis=0))
    performance=np.exp(logPerformance) # Now it starts with 1.0
    gbmValues=startVal*performance
    
    return gbmValues


def sample_mvBM(start_val, mu, covar, stepSize, nSteps):
    a = _get_covar_root(covar)
    return _sample_mvBM_from_sigma_root(start_val=start_val, mu=mu, covar_root=a, stepSize=stepSize, nSteps=nSteps)

def _sample_mvBM_from_sigma_root(start_val, mu, covar_root, stepSize, nSteps):
    d = len(start_val)
    dim_mismatch = len(mu)!=d or np.shape(covar_root)!=(d,d) or d<1
    if (dim_mismatch):
        raise ValueError("Wrong input dimensions: start_val=" + np.array2string(start_val) + 
                         ", mu=" + np.array2string(mu) + 
                         ", covar_root=" + np.array2string(np.asarray(covar_root)))

    indepIncrements = np.random.normal(loc=0.0, scale=np.sqrt(stepSize), size=[d,nSteps])
    depIncrementsWithScale_and_Shift = np.add(np.transpose(np.asmatrix(mu))*stepSize, np.matmul(covar_root, indepIncrements))
    out = np.cumsum(np.concatenate((np.transpose(np.asmatrix(start_val)),depIncrementsWithScale_and_Shift),axis=1), axis=1)
    return out

def _get_covar_root(covar):    
    if (not(np.all(covar == np.transpose(covar)))):
        raise ValueError("Asymmetric covariance matrix: " + np.array2string(covar))
    a=np.transpose(cholesky(covar))
    return a
    
    
def sample_mvGBM(start_val, mu, covar, stepSize, nSteps):
    a =_get_covar_root(covar)
    adj_mu = _get_adj_mu(mu=mu, covar_root=a)
    logPerformance =_sample_mvBM_from_sigma_root(start_val=np.zeros(np.shape(start_val)), mu=adj_mu, covar_root=a, stepSize=stepSize, nSteps=nSteps) 
    performance = np.exp(logPerformance)
    out = np.multiply(np.transpose(np.asmatrix(start_val)), performance)
    return out
    
def _get_adj_mu(mu, covar_root):
    d=len(mu)
    if (np.shape(covar_root)!=(d,d)):
        raise ValueError("Wrong input dimensions: mu=" + np.array2string(mu) + ", covar_root=" + array2string(covar_root))
    return mu - 0.5*np.sum(covar_root**2, axis=1)

# Black-Scholes price of a European call option for a given numer of days to maturity
def bs_eur_call_for_daysToMaturity(spot, strike, instAnnRfRate, annVol, daysToMaturity, nBusDays):
    
    return bs_eur_call(S=spot, 
                       K=strike,
                       r=instAnnRfRate,
                       sigma=annVol,
                       T=daysToMaturity/nBusDays)


# Black-Scholes price of a European call option with time to maturity in years:
def bs_eur_call(S, K, r, sigma, T):
    
    d1=(np.log(S/K) + (r +0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
   
    delta = norm.cdf(d1); # this is also nr. of shares in the stock
    bankAccount = - np.exp(-r*T) * K * norm.cdf(d2) 
    
    return OptionValueWithDeltaHedge(delta, S, bankAccount)

class OptionValueWithDeltaHedge:
    'Delta hedging position in a market consisting of a single stock and a zero-coupon bond'
    
    def __init__(self, delta, stockVal, bankAccount):
        self.delta = delta
        self.stockVal = stockVal
        self.bankAccount = bankAccount
        self.optionVal = stockVal*delta + bankAccount

