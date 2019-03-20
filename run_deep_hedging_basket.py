import numpy as np
import tensorflow as tf
import deep_hedging_basket as dh_basket
import aux_math_fin as mf


my_seed=1
my_dim=10
normalized_w_vec=np.ones([my_dim])/my_dim
S0=100.0
K=100.0
annual_rf_Rate=0.04
instAnnRfRate=np.log1p(annual_rf_Rate)
annual_vols=np.multiply(0.15,np.ones([my_dim]))
cross_corr=0.5
nBusDays=250
T_in_days=30
n_time_steps=10
learning_rate=1e-2
n_maxstep=4000


def _1d_BS_benchmarks(spot, strike, instAnnRfRate, annVol, daysToMaturity, nBusDays):
    
    exact_BS_results = mf.bs_eur_call_for_daysToMaturity(spot=spot, 
                                                         strike=strike, 
                                                         instAnnRfRate=instAnnRfRate, 
                                                         annVol=annVol, 
                                                         daysToMaturity=daysToMaturity, 
                                                         nBusDays=nBusDays)
    price=exact_BS_results.optionVal*np.ones([1]) # the price from deep hedging has this data structure
    hedge=exact_BS_results.delta*normalized_w_vec # the deltas from deep hedging have this data structure
    
        
    print("Benchmark price: ")
    print(price)
    
    print("Benchmark hedge: ")
    print(hedge)

    return price, hedge

def _mc_basket_benchmarks(spot, strike, normalized_w_vec, instAnnRfRate, annual_vols, cross_corr, daysToMaturity, nBusDays, nRuns):

    print("Computing the benchmark option price via Monte Carlo...")
    
    T_in_years = T_in_days/nBusDays
    payoffs = np.zeros(nRuns)
    
    for i in range(nRuns):
        final_prices = mf.sample_mvGBM(start_val=np.ones([my_dim])*spot, mu=instAnnRfRate*np.ones(my_dim), 
                                       covar=mf.make_covar_mat(sdev_vec=annual_vols, cross_corr=cross_corr), 
                                       stepSize=T_in_years, 
                                       nSteps=1)[:,1]
        final_basket_value = np.matmul(np.asmatrix(normalized_w_vec), final_prices)
        payoffs[i] = max(final_basket_value - strike, 0)        

    price = np.exp(-instAnnRfRate*T_in_years)*np.mean(payoffs)*np.ones([1]) # the price from deep hedging has this data structure
    
        
    print("Benchmark price: ")
    print(price)
    
    print("Benchmark hedge: ")
    print("Not yet implemented (None)")

    
    hedge = None
    return price, hedge
    

def main():
    
    print("Cross-correlation: ")
    print(cross_corr)
    print("Basket weights: \n" + np.array2string(normalized_w_vec))
    print("Spot: ")
    print(S0)
    print("Strike: ") 
    print(K)
    
#!!! annual_vols[0] is a dirty hack to ensure that the stuff works for d=1
#     price, hedge = _1d_BS_benchmarks(spot=S0, strike=K, instAnnRfRate=instAnnRfRate, 
#                                                          annVol=annual_vols[0], 
#                                                          daysToMaturity=T_in_days, 
#                                                          nBusDays=nBusDays)
    
    price, hedge = _mc_basket_benchmarks(spot=S0, strike=K, 
                                         normalized_w_vec=normalized_w_vec,
                                         instAnnRfRate=instAnnRfRate, 
                                         annual_vols=annual_vols, 
                                         cross_corr=cross_corr, 
                                         daysToMaturity=T_in_days, 
                                         nBusDays=nBusDays, 
                                         nRuns=100000)

    
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(my_seed)
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print("Begin to price a European call option on a basket of assets.")
        model = dh_basket.BasketSolver(
            annual_rf_Rate=annual_rf_Rate, 
            d=my_dim,
            annual_vols=annual_vols, 
            cross_corr=cross_corr,
            w_vec=normalized_w_vec,
            S0_basket=S0, 
            K_basket=K, 
            T_in_days=T_in_days, 
            n_time_steps=n_time_steps, 
            nBusDays=nBusDays,
            learning_rate=learning_rate,
            n_maxstep=n_maxstep,
            benchmark_option_price_at_time_0=price, 
            benchmark_hedge_at_time_0=hedge)

        model.build()

#        writer = tf.summary.FileWriter('./graphs', sess.graph)

        model.train(sess)
#        output = np.zeros((len(model.init_history),3))
#        output[:,0] = (np.arange(len(model.init_history))) * model.n_displaystep
#        output[:,1] = model.loss_history
#        output[:,2] = model.init_history
#        np.savetxt(fname="./outputs/basket.txt", 
#                   X=output, 
#                   fmt=['%d', '%.5e', '%.5e'], 
#                   delimiter=",",
#                   header="step, loss function, target value, runtime", 
#                   comments='')
        
        writer.close()


if (__name__ == '__main__'):
    np.random.seed(my_seed)

main()

