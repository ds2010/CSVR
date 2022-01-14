# import dependencies
from re import compile
from os import environ
import numpy as np
from pyomo.opt import SolverFactory, SolverManagerFactory
from constant import CET_ADDI, CET_MULT, CET_Model_Categories, OPT_LOCAL, OPT_DEFAULT, RTS_CRS
__email_re = compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')


def set_neos_email(address):
    """pass email address to NEOS server 

    Args:
        address (String): your own vaild email address.
    """
    if address == OPT_LOCAL:
        # print("Optimizing locally.")
        return False
    if not __email_re.match(address):
        raise ValueError("Invalid email address.")
    environ['NEOS_EMAIL'] = address
    return True


def optimize_model(model, email, cet, solver=OPT_DEFAULT):
    if not set_neos_email(email):
        if solver is not OPT_DEFAULT:
            assert_solver_available_locally(solver)
        elif cet == CET_ADDI:
            solver = "mosek"
        elif cet == CET_MULT:
            raise ValueError(
                "Please specify the solver for optimizing multiplicative model locally.")
        solver_instance = SolverFactory(solver)
        # print("Estimating the {} locally with {} solver.".format(
        #     CET_Model_Categories[cet], solver), flush=True)
        return solver_instance.solve(model), 1
    else:
        if solver is OPT_DEFAULT and cet is CET_ADDI:
            solver = "mosek"
        elif solver is OPT_DEFAULT and cet == CET_MULT:
            solver = "knitro"
        solver_instance = SolverManagerFactory('neos')
        print("Estimating the {} remotely with {} solver.".format(
            CET_Model_Categories[cet], solver))
        return solver_instance.solve(model, tee=True, opt=solver), 1


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


def assert_valid_basic_data(y, x, z=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_1d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if len(y_shape) == 2 and y_shape[1] != 1:
        raise ValueError(
            "The multidimensional output data is supported by direciontal based models.")

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if type(z) != type(None):
        z = trans_list(z)
        z = to_2d_list(z)
        z_shape = np.asarray(z).shape
        if y_shape[0] != z_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and z.")

    return y, x, z


def assert_valid_mupltiple_y_data(y, x):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")
    return y, x


def assert_valid_reference_data(y, x, yref, xref):
    yref = trans_list(yref)
    xref = trans_list(xref)

    yref = to_2d_list(yref)
    xref = to_2d_list(xref)

    yref_shape = np.asarray(yref).shape
    xref_shape = np.asarray(xref).shape

    if yref_shape[0] != xref_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in xref and yref.")
    if yref_shape[1] != np.asarray(y).shape[1]:
        raise ValueError(
            "Number of outputs must be the same in y and yref.")
    if xref_shape[1] != np.asarray(x).shape[1]:
        raise ValueError(
            "Number of inputs must be the same in x and xref.")
    return yref, xref


def assert_valid_reference_data_with_bad_outputs(y, x, b, yref, xref, bref):
    yref, xref = assert_valid_reference_data(y, x, yref, xref)

    if type(b) == type(None):
        return yref, xref, None

    bref = to_2d_list(bref)
    bref_shape = np.asarray(bref).shape

    if bref_shape[0] != np.asarray(yref).shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in yref and bref.")
    if bref_shape[1] != np.asarray(b).shape[1]:
        raise ValueError(
            "Number of undesirable outputs  must be the same in b and bref.")

    return yref, xref, bref


def assert_valid_direciontal_data(y, x, b=None, gy=[1], gx=[1], gb=None):
    y = trans_list(y)
    x = trans_list(x)

    y = to_2d_list(y)
    x = to_2d_list(x)

    gy = to_1d_list(gy)
    gx = to_1d_list(gx)

    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if y_shape[1] != len(gy):
        raise ValueError("Number of outputs must be the same in y and gy.")

    if x_shape[1] != len(gx):
        raise ValueError("Number of inputs must be the same in x and gx.")

    if type(b) != type(None):
        b = trans_list(b)
        b = to_2d_list(b)
        gb = to_1d_list(gb)
        b_shape = np.asarray(b).shape
        if b_shape[0] != b_shape[0]:
            raise ValueError(
                "Number of DMUs must be the same in y and b.")
        if b_shape[1] != len(gb):
            raise ValueError(
                "Number of undesirable outputs must be the same in b and gb.")

    return y, x, b, gy, gx, gb


def assert_optimized(optimization_status):
    if optimization_status == 0:
        raise Exception(
            "Model isn't optimized. Use optimize() method to estimate the model.")


def assert_contextual_variable(z):
    if type(z) == type(None):
        raise Exception(
            "Estimated coefficient (lambda) cannot be retrieved due to no contextual variable (z variable) included in the model.")


def assert_undesirable_output(b):
    if type(b) == type(None):
        raise Exception(
            "Estimated coefficient (delta) cannot be retrieved due to no undesirable output (b variable) included in the model.")


def assert_various_return_to_scale(rts):
    if rts == RTS_CRS:
        raise Exception(
            "Estimated intercept (alpha) cannot be retrieved due to the constant returns-to-scale assumption.")


def assert_various_return_to_scale_omega(rts):
    if rts == RTS_CRS:
        raise Exception(
            "Omega cannot be retrieved due to the constant returns-to-scale assumption.")


def assert_solver_available_locally(solver):
    if not SolverFactory(solver).available():
        raise ValueError("Solver {} is not available locally.".format(solver))


def assert_valid_univariate_data(y, x):
    y_shape = np.asarray(y).shape
    x_shape = np.asarray(x).shape

    if y_shape[0] != x_shape[0]:
        raise ValueError(
            "Number of DMUs must be the same in x and y.")

    if x_shape[1] != 1:
        raise ValueError(
            "univariate CNLS only allows single x as the input.")

    y = np.asarray(y).reshape(y_shape[0], 1)
    x = np.asarray(x).reshape(x_shape[0], 1)

    # sorting the observed data in ascending order according to x
    data = np.concatenate((y, x), axis=1)
    data = data[np.argsort(data[:, 1])].T

    y = trans_list(data[0])
    x = trans_list(data[1])

    y = to_1d_list(y)
    x = to_1d_list(x)

    return y, x
