import warnings
from deap import base, creator, tools, algorithms
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
import random
from scipy.special import expit

"""
    the program for the accelerated proximal gradient method for non-smooth multi-objective problem
"""

# define objective functions and their gradient functions,proximal operator
def f1(x, mu):
    res = 0
    if np.any(mu > 0):
        res = mu * np.log(expit((x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1) / mu) + expit(
            (-x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1) / mu))
    return res

def grad_f1(x, mu):
    res = 0
    if np.any(mu > 0):
        a = x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1
        b = -x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1
        term1 = ((expit(a / mu) - expit(b / mu)) * (2 * x[0])) / (expit(a / mu) + expit(b / mu))
        term2 = ((2 * (x[0] - 1) + 1) * (expit(a / mu)) + (-2 * (x[0] - 1) + 1) * (expit(b / mu))) / (
                    expit(a / mu) + expit(b / mu))
        res = np.array([term1, term2])
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



def optimize_process(x0):
    k = 0
    iter_max = 1e6
    f11_prev = f11(x0)
    f22_prev = f22(x0)
    xk = x0
    xk_old = np.zeros_like(xk)
    mu_k = 0.5
    w = 0.5
    lr = 1e-9
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

        def prox_wsum_g(x, w):
            return np.array([(x[0] + 1 - w) / (4 * (1 - w) + 1), x[1] / (4 * (1 - w) + 1)])

        primal_value = prox_wsum_g(y_minus_wsum_nabla_f_yk, lr*mu_k*w)

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
        primal_value_star = prox_wsum_g(y_minus_wsum_nabla_f_yk_star,lr*mu_k*res.x)
        primal_value_star_re = np.array([float(primal_value_star[0]), float(primal_value_star[1])])
        xk_old = xk
        xk = primal_value_star_re
        if np.linalg.norm(xk - yk,ord=np.infty) < epsilon:
            print(f'符合循环条件')
            break
        if k >= iter_max:
            warnings.warn("iter_max exceeded")
        k += 1
        # f11_current = f11(xk)
        # f22_current = f22(xk)
        # if f11_prev - f11_current < 1e-22 and f22_prev - f22_current < 1e-22:
        #     print(f'函数下降不明显，迭代终止于第 {k} 次')
        #     break
        # f11_prev = f11_current
        # f22_prev = f22_current
    return xk
# define the primal objective funtion
def f11(x):
    res = np.maximum(x[0] ** 2 + (x[1] - 1) ** 2 + x[1] - 1, -x[0] ** 2 - (x[1] - 1) ** 2 + x[1] + 1)
    return res

def f22(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res


# 设置 DEAP 框架
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数
def evaluate(individual):
    return f11(individual), f22(individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# 算法参数
population = toolbox.population(n=50)
NGEN = 10
CXPB = 0.9
MUTPB = 0.1

# 运行算法
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# 对种群进行非支配排序
fronts = tools.sortNondominated(population, len(population), first_front_only=False)

# 选择前 50 个解点
solution_points = []
for front in fronts:
    for ind in front:
        solution_points.append(ind)
        if len(solution_points) == 10:
            break
    if len(solution_points) == 10:
        break

# np.random.seed(0)
# initial_points = np.random.uniform(-1, 1, (100, 2))

# save the optimal result
f11_values = []
f22_values = []

# save the initial result
f11_initial_values = []
f22_initial_values = []

# compute the result for all the initial points
for point in solution_points:
    optimized_result = optimize_process(point)
    f110 = f11(point)
    f220 = f22(point)
    f111 = f11(optimized_result)
    f221 = f22(optimized_result)
    print(
        f'点为{point},优化后为{optimized_result},f11函数初始值为{f110},优化后为{f111};f22函数初始值为{f220},优化后为{f221}')

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



plt.savefig("Crescent&Mifflin2_pareto_front.png")
plt.show()