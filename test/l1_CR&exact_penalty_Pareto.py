import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import warnings
from deap import base, creator, tools, algorithms
import random
from scipy.optimize import approx_fprime

r"""
    we use two test problem in [1] to construct this test problem,
    which is l1-penalty censored regression problem and negative form of exact penalty problem.
    [l1-penalty censored regression problem]
    \min_{0 \leq x \leq 1} \left\| \max{Ax,0} - b \right\|_1 + 0.01 \left\| x\right\|_1 
    [negative form of exact penalty problem]
    \min_{0 \leq x \leq 1} \left\| \max{\left\| Ax-b \right\|_1 - xi,0} + 0.01 \left\| x\right\|_1 
    [1] Wu F, Bian W. Smoothing Accelerated Proximal Gradient Method with Fast Convergence Rate 
        for Nonsmooth Convex Optimization Beyond Differentiability[J]. 
        Journal of Optimization Theory and Applications, 2023, 197(2): 539-572.
"""
def phi(x,mu):
    if np.all(abs(x) > mu):
        res = np.maximum(x, 0)
    else:
        res = ((x + mu)**2)/(4 * mu)
    return res

def theta(x, mu):
    if np.all(abs(x) > mu):
        res = abs(x)
    else:
        res = x ** 2 / (2 * mu) + mu/2
    return res

def f1(x, mu, A):
    bb = A.dot(x)
    b = np.maximum(bb, np.zeros(bb.shape))
    res = 0
    for i in range(100):
        res += theta(phi(A.dot(x)[i], mu)- b[i], mu)
    return res

def f2(x, mu, A, xi):
    bb = A.dot(x)
    b = np.maximum(bb,np.zeros(bb.shape))
    res1 = 0
    for i in range(100):
        res1 = theta(A.dot(x)[i] - b[i], mu)
    res = -phi(res1 - xi, mu)
    return res

# def grad_f1(x, mu, A):
#     res = A[0]*nabla_phi(nabla_phi(A[0].dot(x), mu), mu) + A[1] * nabla_phi(nabla_phi(A[1].dot(x), mu), mu)
#     return res
#
# def grad_f2(x, mu, A,b, xi):
#     bb = A.dot(x)
#     b = np.maximum(bb, np.zeros(bb.shape))
#     res1 = theta(A[0].dot(x) - b[0], mu) + theta(A[1].dot(x) - b[1], mu)
#     part1 = nabla_phi(res1 - xi, mu)
#     part2 = A[0]*(nabla_theta(A[0].dot(x) - b[0], mu)) + A[1]*(nabla_theta(A[1].dot(x) - b[1], mu))
#     res = part1*(part2)
#     return res

def grad_f1(x, mu,A):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f1_wrapper(x):
        return f1(x, mu,A)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f1_wrapper, epsilon)
    return grad

def grad_f2(x, mu,A,xi):
    # Define a wrapper for f1 that only takes x as input for approx_fprime
    def f2_wrapper(x):
        return f2(x, mu,A,xi)
    # Calculate the gradient of f1 at x
    epsilon = np.sqrt(np.finfo(float).eps)  # Optimal step size
    grad = approx_fprime(x, f2_wrapper, epsilon)
    return grad

def g1(x):
    return 0.01* np.linalg.norm(x, ord=1)

def g2(x):
    return -0.03 * np.linalg.norm(x, ord=1)

def prox_wsum_g(x, weight):
    res = np.maximum(abs(x) - weight, 0) * np.sign(abs(x) - weight)
    return res

def nabla_phi(x, mu):
    if np.all(abs(x) > mu):
        if np.all(x > 0):
            res = 1
        else:
            res = 0
    else:
        res = (x + mu)/(2 * mu)
    return res


def nabla_theta(x, mu):
    if np.all(abs(x) > mu):
        if np.all(x > 0):
            res = 1
        elif np.all(x == 0):
            res = 0
        else:
            res = -1
    else:
        res = x / mu
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


def optimize_process(x0,A,xi):
    k = 0
    xk = x0
    xk_old = np.zeros_like(xk)
    mu_k = 1/2
    w = 0.5
    lr = 1e-3
    beta = 4
    epsilon = 1e-3
    sigma = 3/4
    lam = np.array([w, 1 - w])
    w = float(w)
    max_iter = 1000
    tol = 1e-12
    iter_max = 1000
    f11_prev = f11(x0,A)
    f22_prev = f22(x0,A,xi)
    while True:
        mu_k = mu_k / ((k + beta - 1) * (np.log(k + beta - 1) ** sigma))
        gamma_k = (k - 1) / (k + beta - 1)
        yk = xk + gamma_k * (xk - xk_old)
        yk_flat = yk.flatten()  # 将 yk 展平为一维数组
        grad_f1_x = grad_f1(yk_flat, mu_k, A)  # 使用展平后的 yk
        grad_f2_x = grad_f2(yk_flat, mu_k, A, xi)  # 使用展平后的 yk
        wsum_nabla_f_yk = lr * w * grad_f1_x + lr * (1 - w) * grad_f2_x  # 1*2
        wsum_nabla_f_yk_re = wsum_nabla_f_yk.reshape((200, 1))
        # print(np.shape(wsum_nabla_f_yk))
        y_minus_wsum_nabla_f_yk = yk - wsum_nabla_f_yk_re  # 1*2
        # print(f'shape={np.shape(y_minus_wsum_nabla_f_yk)}')
        F_xk_old_mu_k = np.array([f1(xk_old, mu_k,A), f2(xk_old, mu_k,A,xi)]) + g(xk_old)  # 1*2
        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, w*lr*0.04)

        def fun(w):
            primal_value_minus_y = primal_value - y_minus_wsum_nabla_f_yk
            res1 = np.inner(lam, g(primal_value))
            res2 = np.linalg.norm(primal_value_minus_y) / (2 * lr)
            res3 = (lr / 2) * np.linalg.norm(wsum_nabla_f_yk,ord=2) ** 2
            res4 = w * (f1(yk, mu_k,A) - f1(xk, mu_k,A) - g(xk)[0])
            # print(f'shape={res4}')
            res5 = (1 - w) * (f2(yk, mu_k,A,xi) - f2(xk, mu_k,A,xi) - g(xk)[1])
            res = -res1 - res2 + res3 + res4 + res5
            # print(f'shape2={np.shape(res)}')
            return res

        res = minimize(
            fun,
            x0=0.5,
            bounds=[(0, 1)],
            options={"maxiter": max_iter, "ftol": tol},
        )

        lam_star = np.array([res.x, 1 - res.x])
        wsum_nabla_f_yk_star = lam_star[0] * grad_f1_x+ lam_star[1] * grad_f2_x
        #wsum_nabla_f_yk_star = lam_star.flatten().dot(nabla_f_yk)
        wsum_nabla_f_yk_star_re = wsum_nabla_f_yk_star.reshape((200,1))
        y_minus_wsum_nabla_f_yk_star = yk - lr * wsum_nabla_f_yk_star_re
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,res.x*lr*0.04)
        # print(f'primal={np.shape(primal_value_star)}')

        xk_old = xk
        xk = primal_value_star
        if np.linalg.norm(xk - yk, ord=np.infty) < epsilon:
            print(f'符合循环条件')
            break
        if k >= iter_max:
            warnings.warn("iter_max exceeded")
        k += 1
        # f11_current = f11(xk,A)
        # f22_current = f22(xk,A,xi)
        # if abs(f11_current - f11_prev) < 0.00001 and abs(f22_current - f22_prev) < 0.00001:
        #     print(f'迭代终止于第 {k} 次')
        #     break
        # f11_prev = f11_current
        # f22_prev = f22_current
    return xk

def f11(x, A):
    bb = A.dot(x)
    b = np.maximum(bb, np.zeros(bb.shape))
    res = np.linalg.norm(np.maximum(A @ x, 0)-b,ord=1) + 0.01 * np.linalg.norm(x,ord=1)
    return res


def f22(x, A, xi):
    b = A.dot(x)
    res = -np.maximum(np.linalg.norm(A @ x - b,ord=1) - xi,0) - 0.03 * np.linalg.norm(x,ord=1)
    return res


# 生成50个随机的2维向量作为初始点
A = np.random.randn(100,200)
s = 0.1 * 200
# Number of vectors
xi = 1e-3
# general initial points
initial_points = []

for i in range(50):
    x = np.random.uniform(0,1,(200,1))
    x[:200 - int(s)] = 0
    np.random.shuffle(x)
    x[x > 1] = 1
    initial_points.append(x)
# use NGSA-II algorithm to find approximate initial points
# # 设置 DEAP 框架
# creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
# creator.create("Individual", list, fitness=creator.FitnessMulti)
#
# toolbox = base.Toolbox()
#
# # 自定义初始点生成函数
# def custom_init():
#     x = np.random.uniform(0, 1, (200, 1))
#     x[:200 - int(s)] = 0
#     np.random.shuffle(x)
#     x[x > 1] = 1
#     return x
#
# # 注册自定义初始点生成方法
# toolbox.register("attr_float", custom_init)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
#
# # 评估函数
# def evaluate(individual):
#     return f11(individual, A), f22(individual, A, xi)
#
#
# toolbox.register("evaluate", evaluate)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
# toolbox.register("select", tools.selNSGA2)
#
# # 算法参数
# population = toolbox.population(n=50)
# NGEN = 100
# CXPB = 0.9
# MUTPB = 0.1
#
# # 运行算法
# for gen in range(NGEN):
#     offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
#     fits = toolbox.map(toolbox.evaluate, offspring)
#     for fit, ind in zip(fits, offspring):
#         ind.fitness.values = fit
#     population = toolbox.select(offspring, k=len(population))
#
# # 对种群进行非支配排序
# fronts = tools.sortNondominated(population, len(population), first_front_only=False)
#
# # 选择前 50 个解点
# initial_points = []
# for front in fronts:
#     for ind in front:
#         initial_points.append(ind)
#         if len(initial_points) == 100:
#             break
#     if len(initial_points) == 100:
#         break

# save the optimal result
f11_values = []
f22_values = []

# save the initial result
f11_initial_values = []
f22_initial_values = []

# compute the result for all the initial points
for point in initial_points:
    b = A.dot(point)
    optimized_result = optimize_process(point,A,xi)
    f110 = f11(point,A)
    f220 = f22(point,A,xi)
    f111 = f11(optimized_result,A)
    f221 = f22(optimized_result,A,xi)
    print(
        f'f11函数初始值为{f110},优化后为{f111};f22函数初始值为{f220},优化后为{f221}')

    # check
    if not (f111 > f110 and f221 > f220):
        f11_initial_values.append(f110)
        f22_initial_values.append(f220)
        f11_values.append(f111)
        f22_values.append(f221)


# plot the result
plt.figure(figsize=(10, 6))
# plt.scatter(f11_initial_values, f22_initial_values, c='blue', marker='x', label='Initial Points')
plt.scatter(f11_values, f22_values, c='red', marker='o', label='Optimized Points')
plt.xlabel('f11')
plt.ylabel('f22')
plt.title('Pareto Front of Optimization Results')
plt.legend()



plt.savefig("l1_CR&excat_penalty_pareto_front.png")
plt.show()