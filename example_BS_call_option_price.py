# To be written...

import aux_math_fin as mf
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


vol=0.15 # annual volatility
rfRate=0.04  # annual risk-free rate (non-instantaneous, as quoted in the markets)

def main():
    S = np.linspace(start=0.8, stop=1.2, num=101, endpoint=True, retstep=False, dtype=None)
    T = np.arange(start=100, stop=1, step=-1, dtype=None)
    S, T = np.meshgrid(S, T)

    callValuesWithHedges = mf.bs_eur_call_for_daysToMaturity(spot=S, 
                                                             strike=1.0, 
                                                             instAnnRfRate=np.log1p(rfRate), 
                                                             annVol=vol, 
                                                             daysToMaturity=T,
                                                             nBusDays=250)

    getOptValues = np.vectorize(lambda x : x.optionVal)
    getOptDeltas = np.vectorize(lambda x : x.delta)
    getBankAccounts = np.vectorize(lambda x : x.bankAccount)
   
    #Z = getOptValues(callValuesWithHedges)
    Z = getOptDeltas(callValuesWithHedges)
    #Z = getBankAccounts(callValuesWithHedges)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
 
    # Plot the surface.
    surf = ax.plot_surface(X=T, Y=S, Z=Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, cstride=5, rstride=5)

    # Customize the axes.
    cmax = max(0.0, np.max(Z))
    cmin = min(0.0, np.min(Z))
    ax.set_zlim(cmin - 0.01, cmax + 0.01)

    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
 
    ax.set_xlabel("T")
    ax.set_ylabel("S")
    ax.set_zlabel("Delta(S,T)")    
    #ax.set_zlabel("C(S,T)")    

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
 
    plt.show()

    
main()
    