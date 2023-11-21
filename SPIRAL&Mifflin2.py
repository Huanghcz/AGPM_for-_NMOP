import numpy as np
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

"""
    this test use apgm to sovle problem :SPIRAL & Mifflin 2,
    which start point is (-1,-1),
    we get the optimal result is (-0.766046383074987,0.032276777101751225),
    which is better than (−0.4898, 0.0193) (IMPB),(−0.3620, 0.2611) (MPB)in paper[1].
    [1] Hoseini Monjezi N, Nobakhtian S. An inexact multiple proximal bundle algorithm for nonsmooth nonconvex multiobjective optimization problems[J]. 
        Annals of Operations Research, 2022, 311(2): 1123-1154.
"""

def f2(x, mu):
    a = (x[0] - np.sqrt(x[0]**2 + x[1]**2) * np.cos(np.sqrt(x[0]**2 + x[1]**2)))**2 + 0.005 * (x[0]**2 + x[1]**2)
    b = (x[0] - np.sqrt(x[0]**2 + x[1]**2) * np.sin(np.sqrt(x[0]**2 + x[1]**2)))**2 + 0.005 * (x[0]**2 + x[1]**2)
    res = 0
    if mu > 0:
        res = mu * np.log(np.exp(a) + np.exp(b))
    return res

def grad_f2(x, mu):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f1_wrapper(x):
        return f1(x, mu)

    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f1_wrapper, epsilon)
    return grad

def f1(x, mu):
    sum = np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    if np.any(sum > mu):
        res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        res = (x[0] ** 2 + x[1] ** 2 - 1) ** 2 / (2 * mu) + mu / 2
    return res


def grad_f1(x, mu):
    condition = np.any(abs(x[0] ** 2 + x[1] ** 2 - 1) > mu)

    if condition:
        df_dx1 = -3.5 * x[0] * abs(x[0] ** 2 + x[1] ** 2 - 1)
        df_dx2 = -3.5 * x[1] * abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        df_dx1 = (x[0] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu
        df_dx2 = (x[1] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu

    return np.array([df_dx1, df_dx2])


def g2(x):
    return 0


def g1(x):
    res = - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res


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



def optimize_process(x0):
    k = 0
    iter_max = 1e6
    f11_prev = f11(x0)
    f22_prev = f22(x0)
    xk_old = x0
    yk = xk_old
    mu_k = 4/5
    w = 0.5
    lr = 1e-9
    eta = 0.5
    beta = 4
    epsilon = 0.0001
    sigma = 3/4
    lam = np.array([w, 1 - w])
    grad_f1_x = grad_f1(yk, mu_k)  # 1*2
    grad_f2_x = grad_f2(yk, mu_k)  # 1*2
    w = float(w)
    max_iter = 1000
    tol = 1e-12
    while k <= iter_max:
        wsum_nabla_f_yk = w * grad_f1_x + (1 - w) * grad_f2_x  # 1*2
        y_minus_wsum_nabla_f_yk = yk - lr * wsum_nabla_f_yk  # 1*2
        F_xk_old_mu_k = np.array([f1(xk_old, mu_k), f2(xk_old, mu_k)]) + g(xk_old)  # 1*2

        def prox_wsum_g(x, w):
            return np.array([(x[0] + 1 - w) / (4 * (1 - w) + 1), x[1] / (4 * (1 - w) + 1)])

        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, lr*mu_k*w)

        def fun(w):
            primal_value_minus_y = primal_value - y_minus_wsum_nabla_f_yk
            return -np.inner(lam, g(primal_value)) - np.linalg.norm(primal_value_minus_y) / (2 * lr) + (
                    lr / 2) * np.linalg.norm(
                wsum_nabla_f_yk) ** 2 + w * (f1(yk, mu_k) - f1(xk_old, mu_k) - g(xk_old)[0]) + (1 - w) * (
                               f2(yk, mu_k) - f2(xk_old, mu_k) - g(xk_old)[1])

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
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,lr*mu_k*res.x)
        primal_value_star_re = np.array([float(primal_value_star[0]),float(primal_value_star[1])])
        if np.linalg.norm(primal_value_star - yk, ord=np.infty) >= epsilon:
            print(f'未符合准则，继续循环')
            xk = primal_value_star_re
            lr = lr**eta
            mu_k = mu_k / ((k + beta - 1) * (np.log(k + beta - 1) ** sigma))
            gamma_k = (k - 1) / (k + beta - 1)
            yk_plus_1 = xk + gamma_k * (xk - xk_old)
            # Update f, g, f_yk, nabla_f_yk here as needed
            yk = yk_plus_1
            k += 1
        else:
            print(f'符合判断准则，输出xk')
            xk = primal_value_star_re
            return xk

        f11_current = f11(xk)
        f22_current = f22(xk)
        if f11_prev -f11_current < 1e-1 and f22_prev-f22_current < 1e-1:
            print(f'迭代终止于第 {k} 次')
            break
        f11_prev = f11_current
        f22_prev = f22_current

    return xk

def f22(x):
    a = (x[0] - np.sqrt(x[0] ** 2 + x[1] ** 2) * np.cos(np.sqrt(x[0] ** 2 + x[1] ** 2))) ** 2 + 0.005 * (
                x[0] ** 2 + x[1] ** 2)
    b = (x[0] - np.sqrt(x[0] ** 2 + x[1] ** 2) * np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))) ** 2 + 0.005 * (
                x[0] ** 2 + x[1] ** 2)
    res = np.maximum(a,b)
    return res


def f11(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res

point = np.array([-1,-1])
optimized_result = optimize_process(point)
f110 = f11(point)
f220 = f22(point)
f111 = f11(optimized_result)
f221 = f22(optimized_result)
print(f'点为{point},优化后为{optimized_result},f11函数初始值为{f110},优化后为{f111};f22函数初始值为{f220},优化后为{f221}')