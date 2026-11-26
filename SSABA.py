import time
import numpy as np


# Search Step Adjustment Based Algorithm (SSABA)
def SSABA(X, fobj, lb, ub, Max_iter):
    N, dim = X.shape
    fitness = np.array([fobj(individual) for individual in X])
    Xbest = X[np.argmin(fitness)]

    Convergence_curve = np.zeros(Max_iter)
    start_time = time.time()

    # Main loop
    for t in range(1, Max_iter + 1):
        St = 0.5 + (1.0 - 0.5) * (1 - (t / Max_iter))
        for i in range(N):
            A = np.random.rand()
            r = np.random.uniform(-1, 1, dim)
            Xnewi = X[i] + St * (Xbest - X[i]) * A * r
            Xnewi = np.clip(Xnewi, lb, ub)  # Ensure new solution is within bounds

            f = fitness[i]
            f_new = fobj(Xnewi)
            if f_new[i] < f:
                X[i] = Xnewi[i]
                fitness[i] = f_new[i]

        Xbest = X[np.argmin(fitness)]
        Convergence_curve[t-1] = np.min(fitness)

    end_time = time.time() - start_time
    best_fitness = np.min(fitness)
    return best_fitness, Convergence_curve, Xbest, end_time
