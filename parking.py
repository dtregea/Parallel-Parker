"""
Daniel Tregea
Main program to use a genetic algorithm to create parallel parkers
"""
from individual import Individual
from population import Population
from population import generate_random


def main():
    # GA Algorithm Configuration
    Population.size = 299
    Individual.binary_len = 7
    Individual.number_of_genes = 8
    Individual.gamma_lb = -.524
    Individual.gamma_ub = .524
    Individual.beta_lb = -5
    Individual.beta_ub = 5
    Individual.penalty = 200
    Individual.mutation_rate = 0.005

    # GA Algorithm start
    prev_generation = generate_random()
    prev_generation.evaluate()
    print("Generation 0: " + str(prev_generation.best().cost()))
    i = 1
    while not prev_generation.reached_termination():
        new_generation = Population(
            prev_generation.crossover() +
            [Individual(prev_generation.best().gammas, prev_generation.best().betas),
             Individual(prev_generation.second_best().gammas, prev_generation.second_best().betas)
             ]
        )
        new_generation.evaluate()
        print("Generation", i, ": J =", new_generation.best().cost())
        prev_generation = new_generation
        i += 1

    # Print results
    best_individual = prev_generation.best()
    best_individual.plot_trajectory()
    best_individual.plot_controls()
    best_individual.plot_states()
    x_f = best_individual.states[-1][0]
    y_f = best_individual.states[-1][1]
    alpha_f = best_individual.states[-1][2]
    v_f = best_individual.states[-1][3]

    print()
    print("x_f =", x_f)
    print("y_f =", y_f)
    print("alpha_f =", alpha_f)
    print("v_f =", v_f)

    gammas = best_individual.binary_to_gamma()
    betas = best_individual.binary_to_beta()
    lines = []
    for i in range(Individual.number_of_genes):
        lines.append(str(gammas[i]) + "\n")
        lines.append(str(betas[i]) + "\n")
    file = open("controls.dat", "w")
    file.writelines(lines)


if __name__ == '__main__':
    main()
