from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np


def CSVR(y, x, epsilon, u):
    # Function to compute the Convex Support Vector Regression (CSVR) model 
    #         given output y, inputs X, maximum error, and tuning parameter u.
    # Sheng Dai, Aalto University School of Business, Finland
    # Oct 14, 2021
    
    # transfrom x and y
    y = to_1d_list(trans_list(y))
    x = to_2d_list(trans_list(x))
    
    # Initialize the CVSR model
    model = ConcreteModel(name = "CSVR")

    # initialize the sets
    model.I = Set(initialize=range(len(y)))
    model.J = Set(initialize=range(len(x[0])))

    # variables associated with virtual DMUs
    model.alpha = Var(model.I, doc='alpha')
    model.beta = Var(model.I, model.J, bounds=(0.0, None), doc='beta')
    model.ksia = Var(model.I, bounds=(0.0, None), doc='Ksi a')  
    model.ksib = Var(model.I, bounds=(0.0, None), doc='Ksi b')

    # objective function 
    def objective_rule(model):
        return u * (sum(model.ksia[i] for i in model.I) + sum(model.ksib[i] for i in model.I)) + \
                    sum(model.beta[i, j]**2 for i in model.I for j in model.J)
    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Objective function')

    # regression equations
    def regression1_rule(model, i):
        return  y[i] - sum(model.beta[i, j] * x[i][j] for j in model.J) - model.alpha[i] <= epsilon + model.ksia[i]
    model.regression1 = Constraint(model.I, rule=regression1_rule, doc='First regression equantion')

    def regression2_rule(model, i):
        return  sum(model.beta[i, j] * x[i][j] for j in model.J) + model.alpha[i] - y[i] <= epsilon + model.ksib[i]
    model.regression2 = Constraint(model.I, rule=regression2_rule, doc='Second regression equantion')

    # Afriat's inequalities 
    def afriat_rule(model, i, h):
        if i == h:
            return Constraint.Skip
        return  model.alpha[i] + sum(model.beta[i, j] * x[i][j] for j in model.J) <= \
                        model.alpha[h] + sum(model.beta[h, j] * x[i][j] for j in model.J)
    model.afriat = Constraint(model.I, model.I, rule=afriat_rule, doc='afriat inequalities')

    # solve model
    solver = SolverFactory("mosek")
    solver.solve(model)

    # Store estimates
    alpha = np.asarray(list(model.alpha[:].value))
    beta = np.asarray([i + tuple([j]) for i, j in zip(list(model.beta), list(model.beta[:, :].value))])
    beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
    beta = beta.pivot(index='Name', columns='Key', values='Value')
    beta = beta.to_numpy()
    ksia = np.asarray(list(model.ksia[:].value))
    ksib = np.asarray(list(model.ksib[:].value))

    return alpha, beta, ksia, ksib

def trans_list(li):
    if type(li) == list:
        return li
    return li.tolist()

def to_1d_list(li):
    if type(li) == int or type(li) == float:
        return [li]
    if type(li[0]) == list:
        rl = []
        for i in range(len(li)):
            rl.append(li[i][0])
        return rl
    return li

def to_2d_list(li):
    if type(li[0]) != list:
        rl = []
        for value in li:
            rl.append([value])
        return rl
    return li
