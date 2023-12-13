import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.special import expit

"""
    the program for the accelerated proximal gradient method for non-smooth multi-objective problem:CB3 & DEM
"""

# define objective functions and their gradient functions,proximal operator
def f1(x, mu):
    res = 0
    a = (x[0]**4 + x[1]**2)/mu
    b = ((2-x[0])**2 + (2-x[1])**2)/mu
    c = (2 * expit(x[1] - x[0]))/mu
    if np.any(mu > 0):
        res = mu * np.log(expit(a) + expit(b) + expit(c))
    return res

def grad_f1(x, mu):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f1_wrapper(x):
        return f1(x, mu)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f1_wrapper, epsilon)
    return grad

def f2(x, mu):
    res = 0
    a = (5 * x[0] + x[1])/mu
    b = (-5 * x[0] + x[1])/mu
    c = (x[0]**2 + x[1]**2 + 4*x[0])/mu
    if mu>0:
        res = mu * np.log(expit(a) + expit(b) +expit(c))
    return res

def grad_f2(x, mu):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f2_wrapper(x):
        return f2(x, mu)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f2_wrapper, epsilon)
    return grad



def g1(x):
    return 0


def g2(x):
    return 0

def f(x, mu):
    return np.array([f1(x, mu), f2(x, mu)])


def g(x):
    res = np.array([g1(x), g2(x)])
    return res


def jac_f(x, mu):
    term11 = grad_f1(x, mu)[0]
    term12 = grad_f1(x, mu)[1]
    term21 = grad_f2(x, mu)[0]
    term22 = grad_f2(x, mu)[1]
    res = np.array([[term11, term12], [term21, term22]])
    return res



def prox_wsum_g(x,weight):
    res = x
    return res

def optimize_process(x0):
    k = 0
    iter_max = 1e6
    xk = x0
    xk_old = np.zeros_like(xk)
    mu_k = 1
    w = 0.5
    lr = 1e-6
    eta = 0.5
    beta = 3
    epsilon = 0.00001
    sigma = 0.7
    lam = np.array([w, 1 - w])
    w = float(w)
    max_iter = 1000
    tol = 1e-12
    while True:
        mu_k = mu_k / ((k + beta - 1) * (np.log(k + beta - 1) ** sigma))
        gamma_k = (k - 1) / (k + beta - 1)
        yk = xk + gamma_k * (xk - xk_old)
        grad_f1_x = grad_f1(yk, mu_k)  # 1*2
        grad_f2_x = grad_f2(yk, mu_k)
        wsum_nabla_f_yk = w * grad_f1_x + (1 - w) * grad_f2_x  # 1*2
        y_minus_wsum_nabla_f_yk = yk - lr * wsum_nabla_f_yk  # 1*2
        F_xk_old_mu_k = np.array([f1(xk, mu_k), f2(xk, mu_k)]) + g(xk)  # 1*2



        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, 0.8*lr*mu_k*w)

        def fun(w):
            primal_value_minus_y = primal_value - y_minus_wsum_nabla_f_yk
            return -np.inner(lam, g(primal_value)) - np.linalg.norm(primal_value_minus_y) / (2 * lr) + (
                    lr / 2) * np.linalg.norm(
                wsum_nabla_f_yk) ** 2 + w * (f1(yk, mu_k) - f1(xk, mu_k) - g(xk)[0]) + (1 - w) * (
                               f2(yk, mu_k) - f2(xk, mu_k) - g(xk)[1])

        res = minimize(
                fun,
                x0= 0.5,
                bounds = [(0, 1)],
                options={"maxiter": max_iter, "ftol": tol},
            )

        lam_star = np.array([res.x, 1 - res.x])
        wsum_nabla_f_yk_star = lam_star[0] * grad_f1_x+ lam_star[1] * grad_f2_x
        #wsum_nabla_f_yk_star = lam_star.flatten().dot(nabla_f_yk)
        y_minus_wsum_nabla_f_yk_star = yk - lr * wsum_nabla_f_yk_star
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,0.8*lr*mu_k*res.x)
        primal_value_star_re = np.array([float(primal_value_star[0]), float(primal_value_star[1])])
        xk_old = xk
        xk = primal_value_star_re
        if np.linalg.norm(xk - yk,ord=np.infty) < epsilon:
            k +=1
            print(f'符合循环条件')
            print(f'当前迭代次数为{k}')
            break
        if k >= iter_max:
            warnings.warn("iter_max exceeded")
        k += 1
    return xk


def f11(x):
    a = (x[0] ** 4 + x[1] ** 2)
    b = ((2 - x[0]) ** 2 + (2 - x[1]) ** 2)
    c = (2 * expit(x[1] - x[0]))
    res = np.max([a,b,c])
    return res

def f22(x):
    a = (5 * x[0] + x[1])
    b = (-5 * x[0] + x[1])
    c = (x[0] ** 2 + x[1] ** 2 + 4 * x[0])
    res = np.max([a,b,c])
    return res


x = np.array([2,2])
res = optimize_process(x)
f11 = f11(res)
f22 = f22(res)
value = np.array([f11,f22])
print(f'the final result is {res},the funtion value is{value}')