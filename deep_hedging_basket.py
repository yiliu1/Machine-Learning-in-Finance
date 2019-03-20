import time 
import numpy as np
import tensorflow as tf
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops
import aux_math_fin as mf


class BasketSolver(object):

    # The nonlinearity term in our BSDE:     
    def f_tf(self,X, Y, Z):
        # X: price(s) of the volatile replicating instrument(s) at time t. Dimension: [batch_size, self.d] (money market excluded)
        # Y: value of the replicating position (at time t)
        # Z: hedging position(s) (nr. of shares in X (or X_i) we need to hold at time t)
        # It depends on t only through X, Y, Z in our case (Black-Scholes pricing of European options)
        # axis=0 is the batch index (implementation detail only, not related to the hedging problem!)
        # Economic meaning: 
        # return_value = - risk-free rate * (value(money market account) required by our strategy)
        # The money market (risk-free lending/borrowing) is the hedging instrument missing in X
        #return self.r*(tf.reduce_sum(Z*X,axis=1,keep_dims=True) - Y)
        return self.r*(tf.reduce_sum(tf.multiply(Z,X),axis=1,keep_dims=True)-Y)# shape[batch_size,1]

    # The terminal value in our BSDE: (same as the terminal/boundary condition in the corresponding PDE):
    def g_tf(self, X, w):
        # X: price(s) of the option's underlying asset(s) at time t (dimension: [batch_size, d]). 
        # w:basket weights (summing up to 1) (dimension: [1,d])
        # In the Black-Scholes hedging problem we have d=1, and X is also the hedging instrument. But it may be different in another hedging problem. 
        # It depends on t only through X in our case (Black-Scholes pricing of European options)
        # axis=0 is the batch index (implementation detail only, not related to the hedging problem!)
        basket_minus_strike=tf.matmul(a=X,b=w,transpose_b=True)-self.K # shape[batch_size, 1]
        a=tf.shape(basket_minus_strike)[0]
        return tf.maximum(basket_minus_strike,tf.zeros([a,1], tf.float64))
    
    def _normalize_weights(self, w):
        if (np.sometrue(w<0.0)):
            raise ValueError ("Negative basket weights")
        s = np.sum(w)
        if (s==0.0):
            raise ValueError ("Basket weights sum up to zero")
        return np.divide(w, s)

    def loss_func(self, X):
    #    X: hedging error. Dimension: [batch_size, 1]
        Xclipped = tf.clip_by_value(X, -self.err_thresh, self.err_thresh)
        return tf.reduce_mean(Xclipped**2)
    #    return tf.reduce_mean(X**2) # in reports, mean square error is more intuitive than the sum of squares. It tells more about the hedging error that actually occurs.

    
    def sample_path_multivar(self, n_paths):
        S_sample = np.zeros([n_paths, self.d, self.n_time_steps+1])
        dS_sample = np.zeros([n_paths, self.d, self.n_time_steps])

        for i in range (n_paths):
            prices=mf.sample_mvGBM(start_val=self.S0*np.ones([self.d]), # let all asset prices start with S0. Then their weighted sum is S0
                                   mu=self.r*np.ones(self.d), # all asset prices are martingales after discounting with r 
                                   covar=self.covar_mat, 
                                   stepSize=self.h, 
                                   nSteps=self.n_time_steps)
            S_sample[i,:,:] = prices
            dS_sample[i,:,:] = prices[: , 1:np.shape(prices)[1]]-prices[: , 0:np.shape(prices)[1]-1]

        return dS_sample, S_sample

    
    def __init__(self, annual_rf_Rate, d, annual_vols, cross_corr, w_vec, S0_basket, K_basket, 
                 T_in_days, n_time_steps, 
                 nBusDays,
                 learning_rate, n_maxstep, 
                 n_displaystep=None, 
                 benchmark_option_price_at_time_0=None,
                 benchmark_hedge_at_time_0=None):
        if (len(annual_vols)!=d):
            raise ValueError ("Wrong dimension of the volatilities array: " + np.array2string(annual_vols))
        if (len(w_vec)!=d):
            raise ValueError ("Wrong dimension of the basket weights array: " + np.array2string(w_vec))
        
        self.d = d 
        self.nBusDays = nBusDays # our minimalistic business days convention
        self.n_time_steps = n_time_steps # discretization grid
        self.T = T_in_days/self.nBusDays # time to maturity in years 
        self.annual_vols = annual_vols # annual volatilities
        self.cross_corr = cross_corr
        self.covar_mat = mf.make_covar_mat(self.annual_vols, self.cross_corr) # covariance matrix
        self.r = np.log1p(annual_rf_Rate) # instantaneous annual risk-free rate (the one in the discounting term exp(-rT) for discounting from t=T to t=0) 
        self.K = K_basket # the option strike. Must have dimension 1.
        self.S0 = S0_basket # asset value at time 0. Must have dimension 1. We don'd use d>1 yet.  
        self.w_normalized=self._normalize_weights(w_vec)  
        self.h = self.T/(self.n_time_steps) # time step size in years
        self.t_stamp = np.arange(0,self.n_time_steps+1)*self.h # The last time step is T!
        self.layer_conf = [self.d,self.d+10, self.d+10,self.d]
        self.n_layers = len(self.layer_conf) 
        self.Yini = np.add(max(self.S0-self.K,0.0), [0.0, 0.1]) # the option price is above discounted_payoff_function(S0,K)...
        self.err_thresh = 0.5*self.K # hedging errors above this threshold get clipped
        self._extra_train_ops = []
        self.batch_size = 64
        self.valid_size = 256
        self.learning_rate = learning_rate 
        self.n_maxstep = n_maxstep 
        self.n_displaystep = n_displaystep if n_displaystep!=None else 100
        self.benchmarkY0=benchmark_option_price_at_time_0 #shape=[1]
        self.benchmarkZ0=benchmark_hedge_at_time_0#[shape=[1,d]]
    
    def build(self):
        start_time = time.time()
        
        # Initial value (also the option price at time 0
        self.Y0 = tf.Variable(initial_value=tf.random_uniform(shape=[1], 
                                                              minval = self.Yini[0],
                                                              maxval = self.Yini[1], 
                                                              dtype=tf.float64), 
                              trainable=True, collections=None, name="Y0", dtype=tf.float64)

        # Delta at time 0 (it is trained in a different fashion than the following deltas)
        self.Z0 = tf.Variable(initial_value=tf.random_uniform(shape=[1,self.d], 
                                                              minval=-0.2, 
                                                              maxval=0.2, 
                                                              dtype=tf.float64), 
                              trainable=True, collections=None, name="Z0", dtype=tf.float64)
        
        # The asset price (can be directly sampled in the BS-model)
        self.S = tf.placeholder(tf.float64, [None, self.d, self.n_time_steps+1], name="S")
        # The change of the asset price (in the BS-model we can sample S and then compute dS)
        # In other models we may need to sample dS and compute S from dS. 
        self.dS = tf.placeholder(tf.float64, [None, self.d, self.n_time_steps], name="dS")# dW goes into dS
        self.is_training = tf.placeholder(tf.bool)
        
        ##############
        self.allones=tf.ones(shape=[tf.shape(self.dS)[0],1], dtype=tf.float64)
        Y = self.allones*self.Y0 # Y0 shape [batch_size , 1]
        Z = tf.matmul(self.allones,self.Z0) # Z0 shape [batch_size , self.d]
        # normalized basket_weights:
        self.w_tensor = tf.convert_to_tensor(np.asmatrix(self.w_normalized), dtype=tf.float64)
        
        

        with tf.variable_scope('forward'):
            # For times t_0,..., t_{self.n_time_steps-2}:
            for i in range(self.n_time_steps-1):
                # build a deep neural network for the hedge from step t_{i+1} to t_{i+2}:
                Y=self._update_trading_result(Y, Z, self.S, self.dS, i)
                Z = self._one_time_net(self.S[:,:,i+1], str(i+1))#Z_n+1
            # For t_{self.n_time_steps-1} (we do not need to train a Z for the terminal time, no Z needed:
            Y=self._update_trading_result(Y, Z, self.S, self.dS, self.n_time_steps-1)
            
        # terminal time:
        # hedging error:
        term_hedge_error=Y-self.g_tf(self.S[:,:,self.n_time_steps], self.w_tensor)# shape=[batch_size,1] 
        # training loss function:
        self.loss = self.loss_func(term_hedge_error) 
        # building time:
        self.t_bd = time.time()-start_time

    
    def _update_trading_result(self, Y, Z, S, dS, i):
        return Y - self.f_tf(S[:,:,i], Y, Z)*self.h + tf.reduce_sum(Z*dS[:,:,i],axis=1,keep_dims=True)
    
    def _one_time_net(self, x, name):
        with tf.variable_scope(name):
             x_norm  = self._batch_norm(x, name='layer0_normal')
             layer1 = self._one_layer(x_norm, self.layer_conf[1], name='layer1')
             layer2 = self._one_layer(layer1, self.layer_conf[2], name='layer2')
             z = self._one_layer(layer2, self.layer_conf[3], 
                                 activation_fn=None,
                                 name='final')
        return z 
    
    def _one_layer(self, input_,
                        out_sz,
                        activation_fn=tf.nn.relu,
                        std=5.0, 
                        name='linear'):
        with tf.variable_scope(name):
            input_shape = input_.get_shape().as_list()
            w = tf.get_variable(name='Matrix', 
                                shape=[input_shape[1], out_sz],
                                dtype=tf.float64,
                                initializer=norm_init(
                                    mean=0.0,
                                    stddev=std/np.sqrt(input_shape[1] + out_sz),
                                    dtype=tf.float64))
            hidden = tf.matmul(input_, w)
            hidden_bn = self._batch_norm(hidden, name='normal')
            if (activation_fn != None):
                return activation_fn(hidden_bn)
            else:
                return hidden_bn
            
    def _batch_norm(self, x, name):
        """Batch normalization""" 
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            
            beta = tf.get_variable(name='beta', 
                                   shape=params_shape, 
                                   dtype=tf.float64,
                                   initializer=norm_init(mean=0.0, 
                                                         stddev=0.1, 
                                                         dtype=tf.float64),
                                   trainable=True) 
            
            gamma = tf.get_variable(name='gamma',
                                    shape=params_shape,
                                    dtype=tf.float64, 
                                    initializer=unif_init(minval=0.1, 
                                                          maxval=0.5, 
                                                          dtype=tf.float64), 
                                    trainable=True)
            
            mv_mean = tf.get_variable(name='moving_mean',
                                      shape=params_shape,
                                      dtype=tf.float64,
                                      initializer=const_init(value=0.0,
                                                             dtype=tf.float64),
                                      trainable=False)
            
            mv_var = tf.get_variable(name='moving_variance',
                                     shape=params_shape,
                                     dtype=tf.float64,
                                     initializer=const_init(value=1.0, 
                                                             dtype=tf.float64),
                                     trainable=False)
            
            # These ops will only be performed when training:
            mean, variance = tf.nn.moments(x=x, axes=[0], name='moments')
            self._extra_train_ops.append(assign_moving_average(variable=mv_mean,
                                                               value=mean, 
                                                               decay=0.99))
            self._extra_train_ops.append(assign_moving_average(variable=mv_var,
                                                               value=variance, 
                                                               decay=0.99)) 
            mean, variance = control_flow_ops.cond(pred=self.is_training, 
                                                   true_fn = lambda: (mean, variance),
                                                   false_fn = lambda: (mv_mean, mv_var)) 
            y = tf.nn.batch_normalization(x=x, 
                                          mean=mean, 
                                          variance=variance, 
                                          offset=beta, 
                                          scale=gamma, 
                                          variance_epsilon=1e-6)
            y.set_shape(x.get_shape())
            return y
    
    def train(self, sess):
        start_time = time.time()
        
        # train operations
        self.global_step = tf.get_variable(name='global_step',
                                           shape=[],
                                           initializer=const_init(value=1,dtype=tf.int32),
                                           trainable=False,
                                           dtype=tf.int32)
        
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_vars), 
                                             global_step = self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self.loss_history = []
        self.init_history = []
        
        #for validation
        dS_valid, S_valid = self.sample_path_multivar(self.valid_size)
        
        # initialization
        step = 1
        sess.run(tf.global_variables_initializer())
        
        self._validate_and_print(sess, dS_valid, S_valid, step, start_time)
        
        # begin SGD iteration
        for _ in range(self.n_maxstep+1):
            step = sess.run(self.global_step)
            dS_train, S_train = self.sample_path_multivar(self.batch_size)
            feed_dict_train = {self.dS : dS_train,
                               self.S : S_train,
                               self.is_training : True}
            sess.run(self.train_op, feed_dict=feed_dict_train)
            
            if (step % self.n_displaystep == 0): 
                print(step)
                self._validate_and_print(sess, dS_valid, S_valid, step, start_time)
            
            step += 1 # why do we need this?
        
        end_time = time.time()
        print("running time: %.3f s" % (end_time - start_time + self.t_bd))
      
        
    def _validate_and_print(self, sess, dS_valid, S_valid, step, start_time):      
        feed_dict_valid = {self.dS : dS_valid,
                           self.S : S_valid,
                           self.is_training : False}
        temp_loss = sess.run(self.loss, feed_dict=feed_dict_valid)
        temp_Y0 = self.Y0.eval()
        temp_Z0 = self.Z0.eval()
        self.loss_history.append(temp_loss)
        self.init_history.append(temp_Y0)
        runtime = time.time()-start_time+self.t_bd
        
        print("step: %5u, loss: %.4e, valid_size: %3u, runtime: %4u    " % (step, temp_loss, self.valid_size, runtime)) 
        self._print_current_vs_benchmark("Y0", temp_Y0, self.benchmarkY0)
        self._print_current_vs_benchmark("Z0", temp_Z0, self.benchmarkZ0) 
        
    
    def _print_current_vs_benchmark(self, name, current, benchmark):
        my_formatter = {'float_kind':lambda x: "%.4e" % x}
        current_str = np.array2string(current, formatter=my_formatter)
        benchmark_str = "None" if (benchmark is None) else np.array2string(benchmark, formatter=my_formatter)
        print(name + ": " + current_str + ",   benchmark " + name +  ": " + benchmark_str)
            
            