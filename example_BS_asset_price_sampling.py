
import matplotlib.pyplot as plt 
import aux_math_fin as mf
import numpy as np

vol=0.15 # annual volatility
rfRate=0.04  # annual risk-free rate (non-instantaneous, as quoted in the markets)

def main():
    prices=mf.daily_sample_prices_array(s0=100.0, instAnnDrift=np.log1p(rfRate), annVol=vol, nDays=500)
    plt.plot(prices);
    plt.xlabel("Business days")
    plt.show()
    
main()
