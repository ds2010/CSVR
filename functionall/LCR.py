from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
from tools import trans_list, to_1d_list, to_2d_list


def LCR(y, x, L):
    # Function to compute the Lipschitz Convex Regression (LCR) model
    #         given output y, inputs X, maximum error, and tuning parameter L.
    # Sheng Dai, Aalto University School of Business, Finland
    # Dec 9, 2021
    
    # transfrom x and y
    y = to_1d_list(trans_list(y))
    x = to_2d_list(trans_list(x))
    
    # Initialize the CVSR model
    model = ConcreteModel(name = "LCR")

    # initialize the sets
    model.I = Set(initialize=range(len(y)))
    model.J = Set(initialize=range(len(x[0])))

    # variables associated with virtual DMUs
    model.alpha = Var(model.I, doc='alpha')
    model.beta = Var(model.I, model.J, doc='beta')
    model.epsilon = Var(model.I, doc='residual')  

    # objective function 
    def objective_rule(model):
        return  sum(model.epsilon[i]**2 for i in model.I)
                  
    model.objective = Objective(rule=objective_rule, sense=minimize, doc='Objective function')

    # regression equations
    def regression_rule(model, i):
        return  y[i] == model.alpha[i] + sum(model.beta[i, j] * x[i][j] for j in model.J) + model.epsilon[i]

    model.regression = Constraint(model.I, rule=regression_rule, doc='First regression equantion')

    # Afriat's inequalities 
    def afriat_rule(model, i, h):
        if i == h:
            return Constraint.Skip
        return  model.alpha[i] + sum(model.beta[i, j] * x[i][j] for j in model.J) <= \
                        model.alpha[h] + sum(model.beta[h, j] * x[i][j] for j in model.J)

    model.afriat = Constraint(model.I, model.I, rule=afriat_rule, doc='afriat inequalities')

    # Lipschitz norm bounded by L
    def lipschitz_norm_rule(model, i):
        return sum(model.beta[i, j]**2 for j in model.J) <= L**2

    model.lipschitz_norm = Constraint(model.I, rule=lipschitz_norm_rule, doc='Lipschitz norm')

    # solve model
    solver = SolverFactory("mosek")
    solver.solve(model)

    # Store estimates
    alpha = np.asarray(list(model.alpha[:].value))
    beta = np.asarray([i + tuple([j]) for i, j in zip(list(model.beta), list(model.beta[:, :].value))])
    beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
    beta = beta.pivot(index='Name', columns='Key', values='Value')
    beta = beta.to_numpy()
    epsilon = np.asarray(list(model.epsilon[:].value))

    return alpha, beta, epsilon