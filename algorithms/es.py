import numpy as np

def fitness(schedule, max_daily_hours=2.5):
    variance = np.var(schedule)
    overload_penalty = np.sum(np.maximum(0, schedule - max_daily_hours))
    return variance + overload_penalty

def optimise_es(original_schedule, generations=120, mu=10, lam=40):
    total_hours = np.sum(original_schedule)
    days = len(original_schedule)

    # Initial population
    population = np.random.dirichlet(np.ones(days), mu) * total_hours
    sigma = 0.6  # mutation step size

    for _ in range(generations):
        offspring = []

        for parent in population:
            for _ in range(lam // mu):
                child = parent + np.random.normal(0, sigma, days)
                child = np.clip(child, 0, None)

                # Preserve total weekly hours
                child *= total_hours / np.sum(child)
                offspring.append(child)

        offspring = np.array(offspring)
        fitness_scores = np.array([fitness(o) for o in offspring])

        # Select best μ solutions
        population = offspring[np.argsort(fitness_scores)[:mu]]

        # Self-adaptation
        sigma *= 0.98

    return population[0]
