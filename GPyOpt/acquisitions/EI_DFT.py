# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
# MIT Photovoltaics Laboratory

import pandas as pd  # Added
import numpy as np  # Added
import GPy  # Added

from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionEI_DFT(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, files, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionEI_DFT, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.constraint_model = GP_model(files)  # Added
        self.files = files
    

    @staticmethod
    def fromConfig(model, space, files, optimizer, cost_withGradients, config):
        return AcquisitionEI_DFT(model, space, files, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        
        mean, prob, conf_interval = mean_and_propability(x, self.constraint_model) # Added
        f_acqu = f_acqu * prob # Added
        #print('Acq!') # Added
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        #print('Acq-grad!')  # Added
        mean, prob, conf_interval = mean_and_propability(x, self.constraint_model) # Added
        print('x='+str(x)+', acqu='+str(f_acqu)+', grad='+str(df_acqu))
        f_acqu = f_acqu * prob # Added
        df_acqu = df_acqu * prob # Added
        print('P='+str(prob)+'-->acqu='+str(f_acqu)+', grad='+str(df_acqu))
        return f_acqu, df_acqu

# Added the rest of the file.
def GP_model(files):
    file_CsFA_2 = files[0]
    file_FAMA_2 = files[1]
    file_CsMA_2 = files[2]

    data_CsFA_2 = pd.read_csv(file_CsFA_2)
    data_FAMA_2 = pd.read_csv(file_FAMA_2)
    data_CsMA_2 = pd.read_csv(file_CsMA_2)

    data_all = pd.concat([data_CsFA_2, data_FAMA_2, data_CsMA_2])#, data_CsMAFA_2])#, data_Janak]) # This includes Janak's observations. It's either this or the previous row.
    # TO DO: Newest Bayesian Opt version works for any order of elements. Need
    # to update also DFT to do that one at some point.
    variables = ['Cs', 'MA', 'FA']
    # sample inputs and outputs
    X = data_all[variables] # This is 3D input
    Y = data_all[['dGmix (ev/f.u.)']] # Negative value: stable phase. Uncertainty = 0.025 
    X = X.iloc[:,:].values # Optimization did not succeed without type conversion.
    Y = Y.iloc[:,:].values
    # RBF kernel
    kernel = GPy.kern.RBF(input_dim=3, lengthscale=0.03, variance=0.025)
    # Logistic kernel --> No!
    #kernel = GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[0], variance = 0.05) * GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[1], variance = 0.05) * GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[2], variance = 0.05)
    model = GPy.models.GPRegression(X,Y,kernel)
    
    # optimize and plot
    model.optimize(messages=True,max_f_eval = 100)
    
    #print(model)
    
    #GP.predict(X) (return mean), and __pass it to a sigmoid (0,1)__ (return), GP.raw_predict
    
    return model
    
    # This code should return the whole GP model for Gibbs.    
    # Then, write another function here that will take composition X and model GP
    # as an input, calculate the predicted mean value of Gibbs using the model, pass
    # it to a sigmoid (0,1) to transform it to a "propability" and give that one as
    # an output.
    
    
    # X should be a numpy array containing the suggested composition(s) in the
    # same order than listed in the variables.
def mean_and_propability(x, model):#, variables):
    #if variables != ['Cs', 'FA', 'MA']:
    #    raise ValueError('The compositions in x do not seem to be in the same order than the model expects.')
    #print(x)
    mean = model.predict_noiseless(x) # Manual: "This is most likely what you want to use for your predictions."
    mean = mean[0] # TO DO: issue here with dimensions?
    conf_interval = model.predict_quantiles(np.array(x)) # 95% confidence interval by default. TO DO: Do we want to use this for something?

    propability = 1/(1+np.exp(mean/0.025)) # Inverted because the negative Gibbs energies are the ones that are stable.
    #print(propability)
    return mean, propability, conf_interval

# For testing of GP_model() and mean_and_propability():
'''
model = GP_model()
model.plot(visible_dims=[0,2])
model.plot(visible_dims=[0,1])
model.plot(visible_dims=[1,2])
x1 = np.linspace(0,1,20)
x2 = np.ones(x1.shape) - x1
x3 = np.zeros(x1.shape)
x_CsMA = np.column_stack([x1,x2,x3]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_CsMA, P_CsMA, conf_interval = mean_and_propability(x_CsMA, model)
x_CsFA = np.column_stack([x2,x3,x1]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_CsFA, P_CsFA, conf_interval = mean_and_propability(x_CsFA, model)
x_MAFA = np.column_stack([x3,x1,x2]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_MAFA, P_MAFA, conf_interval = mean_and_propability(x_MAFA, model)

plt.show()
mpl.rcParams.update({'font.size': 22})
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax.set(xlabel='% of compound', ylabel='dGmix (ev/f.u.)',
       title='Modelled Gibbs energy')
ax2.set(xlabel='% compound', ylabel='P(is stable)',
       title='Modelled probability distribution')
ax.grid()
ax2.grid()
ax.plot(x1, mean_CsMA, label='Mean, Cs in CsMA')
ax.plot(x1, mean_CsFA, label = 'Mean, FA in CsFA')
ax.plot(x1, mean_MAFA, label ='Mean, MA in MAFA')
ax.legend()

ax2.plot(x1, P_CsMA, '--', label='P, Cs in CsMA')
ax2.plot(x1, P_CsFA, '--', label='P, FA in CsFA')
ax2.plot(x1, P_MAFA, '--', label='P, MA in MAFA')
ax2.legend()

x = np.linspace(-2,2,200)
y1 = 1/(1+np.exp(x/0.2))
y2 = 1/(1+np.exp(x/0.025))

fig3, ax3 = plt.subplots()
ax3.set(xlabel='x i.e. Gibbs energy', ylabel='inverted sigmoid i.e. P')
ax3.grid()
ax3.plot(x, y1, label = 'scale 0.2')
ax3.plot(x, y2, label = 'scale 0.025')
ax3.legend()

fig4, ax4 = plt.subplots()
ax4.set(xlabel='x', ylabel='inverted sigmoid')
ax4.grid()
ax4.plot(x, y1, label = 'scale 0.2')
ax4.plot(x, y2, label = 'scale 0.025')
ax4.set_xlim(-0.2, 0.2)


plt.show()
'''
