import warnings
from deap import base, creator, tools, algorithms
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
import random
from scipy.special import expit
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

def f2(x, mu):
    a = x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    b = -x[0] + 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    c = x[0] - 10*x[0]/(x[0]+0.1)+2*(x[1]**2)
    if mu > 0:
        res = mu * np.log(expit(0.5*a/mu) + expit(0.5*b/mu) + expit(0.5*c/mu))
    return res


def grad_f2(x, mu):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    if mu > 0:
        res1 = (0.5 *(1+ 1/((x[0]+0.1)**2))/mu) * expit(0.5*a/mu)+0.5 *(-1+ 1/((x[0]+0.1)**2))/mu * expit(0.5*a/mu) +0.5 *(1 - 1/((x[0]+0.1)**2))/mu * expit(0.5*a/mu)
        diff1 = mu * ((res1)/(expit(0.5*a/mu) + expit(0.5*b/mu) + expit(0.5*c/mu)))
        res2 = (2*x[1]/mu) * expit(0.5*a/mu) + (2*x[1]/mu) * expit(0.5*b/mu) + (2*x[1]/mu) * expit(0.5*c/mu)
        diff2 = mu * ((res2)/(expit(0.5*a/mu) + expit(0.5*b/mu) + expit(0.5*c/mu)))
        res = np.array([diff1,diff2])
    return res

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
    xk = x0
    xk_old = np.zeros_like(xk)
    mu_k = 0.5
    w = 0.5
    lr = 1e-9
    eta = 0.5
    beta = 6
    epsilon = 0.00001
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
    return xk

def f22(x):
    a = x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    b = -x[0] + 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    c = x[0] - 10 * x[0] / (x[0] + 0.1) + 2 * (x[1] ** 2)
    res = np.max([0.5 * a, 0.5 * b, 0.5 * c])
    return res


def f11(x):
    res = 1.75 * np.abs(x[0] ** 2 + x[1] ** 2 - 1) - x[0] + 2 * (x[0] ** 2 + x[1] ** 2 - 1)
    return res

# 设置 DEAP 框架
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # 评估函数
# def evaluate(individual):
#     return f11(individual), f22(individual)
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
# solution_points = []
# for front in fronts:
#     for ind in front:
#         solution_points.append(ind)
#         if len(solution_points) == 100:
#             break
#     if len(solution_points) == 100:
#         break
np.random.seed(0)
initial_points = np.random.uniform(-1, 1, (100, 2))
# save the optimal result
f11_values = []
f22_values = []

# save the initial result
f11_initial_values = []
f22_initial_values = []

# compute the result for all the initial points
for point in initial_points:
    optimized_result = optimize_process(point)
    f110 = f11(point)
    f220 = f22(point)
    f111 = f11(optimized_result)
    f221 = f22(optimized_result)

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



plt.savefig("WF&Mifflin_pareto_front.png")
plt.show()