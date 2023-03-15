# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

#imports functions and modules from the "sequence_jacobian" package. It imports the "simple" and "create_model" functions from the "sequence_jacobian" module, and it imports the "hetblocks" and "grids" modules.

from sequence_jacobian import simple, create_model  # functions
from sequence_jacobian import hetblocks, grids      # modules

#defines a variable "hh" that references an object in the "hetblocks" module. It then prints information about the object's inputs, macro outputs, and micro outputs.
hh = hetblocks.hh_labor.hh

print(hh)
print(f'Inputs: {hh.inputs}')
print(f'Macro outputs: {hh.outputs}')
print(f'Micro outputs: {hh.internals}')

# defines a function called "make_grid" that takes in several parameters and returns a tuple of arrays. 
#The function generates a grid of values for economic variables, including the expected value of earnings, 
#the probability distribution of earnings, the transition matrix between states, and a grid of values for assets.
def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid

#defines a function called "transfers" that takes in several parameters and returns an array. 
#The function calculates transfers to households based on the incidence rules proportional to skill, 
#given the distribution of earnings and a set of tax and dividend policies.
def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

#defines a function called "wages" that takes in two parameters and returns an array. 
#The function calculates the wage rate for each skill level based on a given overall wage rate and the distribution of skills.
def wages(w, e_grid):
    we = w * e_grid
    return we

#modified "hh" object called "hh1" that includes the previously defined "make_grid", "transfers", and "wages" functions as heterogeneous inputs. 
#It then prints information about this modified object's inputs.
hh1 = hh.add_hetinputs([make_grid, transfers, wages])

print(hh1)
print(f'Inputs: {hh1.inputs}')

#modified "hh" object called "hh_ext" that includes a new heterogeneous output function called "labor_supply" 
#that calculates the total labor supply for each skill level based on the distribution of skills and a given overall labor supply.hh_ext = hh1.add_hetoutputs([labor_supply])

def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

hh_ext = hh1.add_hetoutputs([labor_supply])
print(hh_ext)
print(f'Outputs: {hh_ext.outputs}')


#defines several more functions using Python decorators "@simple". 
#These functions include a model for a firm's labor and dividend decisions based on its output, wage, and productivity;
# a model for determining the interest rate based on inflation, a natural interest rate,
# and a parameter representing the responsiveness of inflation to interest rates; 
#a model for determining tax revenue based on the interest rate and government debt; 
#a model for ensuring market clearing in the labor, asset, and goods markets; 
#and a model for the non-linear Phillips curve that determines wages based on productivity and the inflation rate.

@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


@simple
def fiscal(r, B):
    Tax = r * B
    return Tax


@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return asset_mkt, labor_mkt, goods_mkt

#defines a non-linear Phillips curve function "nkpc_ss" that takes in the productivity and the parameter "mu" 
#as inputs and returns the wage rate for each skill level.

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w


#defines a list called "blocks_ss" that includes the "hh_ext", "firm", "monetary", "fiscal", "mkt_clearing", and "nkpc_ss" functions. These functions are then passed as arguments to the "create_model" function, which generates a new model called "hank_ss". The "name" parameter specifies the name of the new model.
blocks_ss = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc_ss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

print(hank_ss)
print(f"Inputs: {hank_ss.inputs}")

#Calibration
calibration = {'eis': 0.5, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 500, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6}

#parameters related to agent heterogeneity, such as the elasticity of intertemporal substitution (eis), the Frisch labor supply elasticity (frisch), the persistence of the Markov process for skill levels (rho_e), and the standard deviation of the process (sd_e). Other parameters relate to the macroeconomic environment, such as the productivity level (Z), the steady-state inflation rate (pi), and the discount factor (beta).

#Solve for the steady state
#The code also defines two dictionaries called "unknowns_ss" and "targets_ss". These dictionaries are used to specify the unknown parameters and targets for the steady-state solution of the model. In this case, the unknowns are the discount factor (beta) and the inverse of the Frisch elasticity (vphi), and the targets are the market clearing conditions for assets and labor.
unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0}

#the code calls the "solve_steady_state" method of the "hank_ss" model to compute the steady-state solution of the model given the parameter values, unknowns, and targets specified in the dictionaries. The "solver" parameter specifies the numerical method used to solve the system of equations that determines the steady state. In this case, the "hybr" solver is used, which is a hybrid method that combines a numerical root-finding algorithm with a gradient-based optimization algorithm.
ss0 = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

#Print market clearings
print(f"Asset market clearing: {ss0['asset_mkt']: 0.2e}")
print(f"Labor market clearing: {ss0['labor_mkt']: 0.2e}")
print(f"Goods market clearing (untargeted): {ss0['goods_mkt']: 0.2e}")

#plotting a graph of labor supply as a function of assets for the steady state solution ss0 of the 
#Heterogeneous-Agent New Keynesian (HANK) model.

#The plt.plot() function is used to create a line plot with ss0.internals['hh']['a_grid'] as the x-axis and 
#ss0.internals['hh']['n'].T as the y-axis. ss0.internals contains all the internal variables of the model in steady state, 
#and 'hh' is the name of the hetblock that represents heterogeneous households.

#The plt.xlabel() and plt.ylabel() functions are used to set the x-axis and y-axis labels, respectively. 
#Finally, plt.show() is used to display the plot.
plt.plot(ss0.internals['hh']['a_grid'], ss0.internals['hh']['n'].T)
plt.xlabel('Assets'), plt.ylabel('Labor supply')
plt.show()


#This code defines a function nkpc that takes in several parameters and returns an expression that represents the New Keynesian Phillips Curve.

#The function is then added to a list called blocks along with other blocks that represent different components of a one-asset Heterogeneous Agent New Keynesian (HANK) model. The list of blocks is then used to create a hank model using the create_model function.

#The code then prints out each block in the hank model.

@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res


blocks = [hh_ext, firm, monetary, fiscal, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")

print(*hank.blocks, sep='\n')

#This code is performing a test to ensure that the steady state ss calculated by hank.steady_state(ss0) is consistent with the initial guess ss0.

#Specifically, for each key k in ss0, the code checks whether all the elements of ss[k] are approximately equal to the corresponding elements in ss0[k] using np.isclose() and np.all(). If this condition is not satisfied for any k, the assert statement will raise an AssertionError, indicating that the steady state calculation is incorrect. If the condition is satisfied for all k, the test passes and the code continues to run.
ss = hank.steady_state(ss0)

for k in ss0.keys():
    assert np.all(np.isclose(ss[k], ss0[k]))
    
    
   # The code above calculates the general equilibrium Jacobian matrix G of the One-Asset HANK model defined earlier.

#T is the number of periods for which the Jacobian is calculated.
#exogenous is a list of exogenous variables, which are assumed to be known in the model.
#unknowns is a list of endogenous variables that are being solved for.
#targets is a list of the model equations that are being solved.
#The hank.solve_jacobian method computes the derivatives of the endogenous variables with respect to the other endogenous and exogenous variables in the model, and stores the results in the Jacobian matrix G.

#The resulting Jacobian matrix can be used to analyze the dynamic properties of the model, such as stability and impulse response functions.

# setup
T = 300
exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

# general equilibrium jacobians
G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

print(G)

#Now let's consider 25 basis point monetary policy shocks with different persistences and plot the response of inflation.
rhos = np.array([0.2, 0.4, 0.6, 0.8, 0.9])

drstar = -0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dpi = G['pi']['rstar'] @ drstar

plt.plot(10000 * dpi[:21])
plt.title(r'Inflation responses monetary policy shocks')
plt.xlabel('quarters')
plt.ylabel('bp deviation from ss')
plt.show()