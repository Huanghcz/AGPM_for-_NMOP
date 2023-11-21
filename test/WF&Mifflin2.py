import numpy as np
from scipy.optimize import minimize

"""
    this test use apgm to sovle problem :Crescent & Mifflin 2,
    which start point is (-1,-1),
    we get the optimal result is (0.1632,-0.3042),
    which is better than (0.17156, −0.72429) in paper[1],
    and better than (0.3939, −0.8529) in paper[2]
    [1] Mäkelä M M, Karmitsa N, Wilppu O. Multiobjective proximal bundle method for nonsmooth optimization[R]. TUCS, Technical Report, 2014.
    [2] Hoseini Monjezi N, Nobakhtian S. An inexact multiple proximal bundle algorithm for nonsmooth nonconvex multiobjective optimization problems[J]. 
        Annals of Operations Research, 2022, 311(2): 1123-1154.
"""

def f1(x, mu):
    a = x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    b = -x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    c = x[0] - 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    res = 0
    if mu > 0:
        res = mu * np.log(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu))
    return res


def grad_f1(x, mu):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    res = 0
    if mu > 0:
        res1 = (0.5 *(1+ 1/((x[0]+0.1)**2))/mu) * np.exp(0.5*a/mu)+0.5 *(-1+ 1/((x[0]+0.1)**2))/mu * np.exp(0.5*a/mu) +0.5 *(1 - 1/((x[0]+0.1)**2))/mu * np.exp(0.5*a/mu)
        diff1 = mu * ((res1)/(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu)))
        res2 = (2*x[1]/mu) * np.exp(0.5*a/mu) + (2*x[1]/mu) * np.exp(0.5*b/mu) + (2*x[1]/mu) * np.exp(0.5*c/mu)
        diff2 = mu * ((res2)/(np.exp(0.5*a/mu) + np.exp(0.5*b/mu) + np.exp(0.5*c/mu)))
        res = np.array([diff1,diff2])
    return res

def f2(x, mu):
    sum = np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    if np.any(sum > mu):
        res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        res = (x[0] ** 2 + x[1] ** 2 - 1) ** 2 / (2 * mu) + mu / 2
    return res


def grad_f2(x, mu):
    condition = np.any(abs(x[0] ** 2 + x[1] ** 2 - 1) > mu)

    if condition:
        df_dx1 = -3.5 * x[0] * abs(x[0] ** 2 + x[1] ** 2 - 1)
        df_dx2 = -3.5 * x[1] * abs(x[0] ** 2 + x[1] ** 2 - 1)
    else:
        df_dx1 = (x[0] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu
        df_dx2 = (x[1] * (x[0] ** 2 + x[1] ** 2 - 1)) / mu

    return np.array([df_dx1, df_dx2])


def g1(x):
    return 0


def g2(x):
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


def line_search(xk, grad, lr, mu_k):
    alpha = 1.0  # 初始步长
    beta = 0.5  # 步长缩减因子
    c = 1e-4  # 用于检查目标函数减少的常数
    while True:
        new_xk = xk - alpha * grad  # 更新迭代点
        if f(new_xk, mu_k).sum() <= f(xk, mu_k).sum() - c * alpha * np.linalg.norm(grad) ** 2:
            return new_xk  # 如果满足减少条件，则返回新的迭代点
        alpha *= beta  # 减少步长
        if alpha < 1e-8:  # 防止步长过小
            break
    return xk  # 如果没有找到满足条件的步长，则返回原始迭代点

def optimize_process(x0):
    k = 0
    iter_max = 1e6
    f11_prev = f11(x0)
    f22_prev = f22(x0)
    xk = x0
    xk_old = np.zeros_like(xk)
    mu_k = 0.3
    w = 0.5
    lr = 1e-2
    eta = 0.5
    beta = 6
    epsilon = 0.01
    sigma = 3/4
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
        primal_value_star_re = np.array([float(primal_value_star[0]), float(primal_value_star[1])])
        xk_old = xk
        xk = primal_value_star_re
        if np.linalg.norm(primal_value_star - yk,ord=np.infty) < epsilon:
            print(f'符合循环条件')
            break
        if k >= iter_max:
            warnings.warn("iter_max exceeded")
        k += 1



        f11_current = f11(xk)
        f22_current = f22(xk)
        if f11_prev -f11_current < 1e-22 and f22_prev-f22_current < 1e-22:
            print(f'函数下降不明显，迭代终止于第 {k} 次')
            break
        f11_prev = f11_current
        f22_prev = f22_current
    return xk

def f11(x):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    res = np.max([0.5 * a, 0.5 * b, 0.5 * c])
    return res


def f22(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res

point = np.array([3,1])
optimized_result = optimize_process(point)
f110 = f11(point)
f220 = f22(point)
f111 = f11(optimized_result)
f221 = f22(optimized_result)
print(f'点为{point},优化后为{optimized_result},f11函数初始值为{f110},优化后为{f111};f22函数初始值为{f220},优化后为{f221}')