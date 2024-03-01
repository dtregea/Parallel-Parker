"""
Daniel Tregea
A class to represent an individual in a genetic algorithm
"""

import math
import random
import numpy as np
from random import random as random
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt

"""
Randomly mutate an array of parameters
"""
def mutate_params(parameters):
    for i in range(len(parameters)):
        for j in range(len(parameters[i])):
            if random() < Individual.mutation_rate:
                parameters[i] = parameters[i][0:j] + flip(parameters[i][j]) + parameters[i][j + 1:]

"""
Generate a random individuals
"""
def generate_random():
    gammas = [binary_to_gray(bin(round(random() * 2 ** Individual.binary_len))[2:].zfill(Individual.binary_len)) for x
              in
              range(Individual.number_of_genes)]
    betas = [binary_to_gray(bin(round(random() * 2 ** Individual.binary_len))[2:].zfill(Individual.binary_len)) for x
             in
             range(Individual.number_of_genes)]
    return Individual(gammas, betas)


"""
The following 4 functions were found from the following source
to convert to and from Gray Code
https://www.geeksforgeeks.org/gray-to-binary-and-binary-to-gray-conversion/
"""

def xor(a, b):
    return '0' if (a == b) else '1'


def flip(c):
    return '1' if (c == '0') else '0'


def binary_to_gray(binary):
    gray = ""
    gray += binary[0]

    for i in range(1, len(binary)):
        gray += xor(binary[i - 1],
                    binary[i])
    return gray


def gray_to_binary(gray):
    binary = ""
    binary += gray[0]
    for i in range(1, len(gray)):
        if gray[i] == '0':
            binary += binary[i - 1]
        else:
            binary += flip(binary[i - 1])
    return binary


class Individual:
    binary_len = 0
    number_of_genes = 0
    gamma_lb = 0
    gamma_ub = 0
    beta_lb = 0
    beta_ub = 0
    penalty = 0
    mutation_rate = 0

    def __init__(self, gammas, betas):
        self.gammas = gammas
        self.betas = betas
        self.states = []
        self.controls = []
        self.J = None

    """
    Simulate the control states of an individual to determine its cost
    """
    def cost(self):
        if self.J is not None:
            return self.J

        gammas = self.binary_to_gamma()
        betas = self.binary_to_beta()
        linespace = np.linspace(0, 10, Individual.number_of_genes)
        gamma_poly = scipy.interpolate.CubicSpline(linespace, gammas)
        beta_poly = scipy.interpolate.CubicSpline(linespace, betas)
        self.states = [[0, 8, 0, 0, 0]]
        prev_state = self.states[0]

        for i in np.arange(0, 10, 0.1):
            gamma = gamma_poly(i)
            beta = beta_poly(i)

            next_x = prev_state[0] + (prev_state[3] * (math.cos(prev_state[2]) * 0.1))
            next_y = prev_state[1] + (prev_state[3] * (math.sin(prev_state[2]) * 0.1))
            next_a = prev_state[2] + (gamma * 0.1)
            next_v = prev_state[3] + (beta * 0.1)
            next_c = prev_state[4]
            # check error
            if next_x <= -4 and next_y <= 3:
                next_c += (3 - next_y) ** 2
            elif -4 < next_x < 4 and next_y <= -1:
                next_c += (-1 - next_y) ** 2
            elif next_x >= 4 and next_y <= 3:
                next_c += (3 - next_y) ** 2
            new_state = [next_x, next_y, next_a, next_v, next_c]
            prev_state = new_state
            self.states.append(new_state)
            self.controls.append([gamma, beta])

        if prev_state[4] > 0:
            cost = Individual.penalty + prev_state[4]
        else:
            cost = math.sqrt(abs(prev_state[1]) ** 2 + abs(prev_state[0]) ** 2 + abs(prev_state[2]) ** 2 + abs(
                prev_state[3]) ** 2)
        self.J = cost
        return cost

    """
    Determine the fitness value of an individual
    """
    def fitness(self):
        return 1 / (self.cost() + 1)

    """
    Convert an individuals gray code gamma values to floats
    """
    def binary_to_gamma(self):
        return [((int(gray_to_binary(value), 2) / (2 ** Individual.binary_len - 1)) * (
                Individual.gamma_ub - Individual.gamma_lb)) + Individual.gamma_lb for value in self.gammas]

    """
    Convert an individuals gray code beta values to floats
    """
    def binary_to_beta(self):
        return [((int(gray_to_binary(value), 2) / (2 ** Individual.binary_len - 1)) * (
                Individual.beta_ub - Individual.beta_lb)) + Individual.beta_lb for value in self.betas]

    """
    Mutate all of this individuals parameters
    """
    def mutate(self):
        mutate_params(self.gammas)
        mutate_params(self.betas)

    """
    Plot the state history of this individual
    """
    def plot_trajectory(self):
        x = []
        y = []
        for i in range(len(self.states)):
            x.append(self.states[i][0])
            y.append(self.states[i][1])
        plt.style.use('seaborn-poster')
        plt.figure(figsize=(10, 8))

        plt.plot(x, y, 'g')
        plt.plot([-15, -4], [3, 3], 'k')
        plt.plot([-4, -4], [3, -1], 'k')
        plt.plot([-4, 4], [-1, -1], 'k')
        plt.plot([4, 4], [-1, 3], 'k')
        plt.plot([4, 15], [3, 3], 'k')
        plt.xlabel('x (ft)')
        plt.ylabel('y (ft)')
        plt.xlim([-15, 15])
        plt.ylim([-10, 15])
        plt.grid(color='black')
        plt.title("Parallel Parking Path by Most Fit Individual")
        plt.show()

    """
    Plot the control histories of this individual
    """
    def plot_controls(self):
        plt.style.use('seaborn-poster')
        plt.figure(figsize=(10, 8))
        plt.grid(color='black')
        x = np.linspace(0, 10, 100)

        y_gammas = []
        y_betas = []
        for i in range(100):
            y_gammas.append(self.controls[i][0])
            y_betas.append(self.controls[i][1])

        plt.xlabel('Time (s)')
        plt.ylabel('Gamma (ft^2)')
        plt.plot(x, y_gammas, 'b')
        plt.show()

        plt.style.use('seaborn-poster')
        plt.figure(figsize=(10, 8))
        plt.grid(color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Beta (ft^2)')
        plt.plot(x, y_betas, 'b')
        plt.show()

    """
    Plot the state parameter histories of this individual
    """
    def plot_states(self):
        plt.style.use('seaborn-poster')
        plt.figure(figsize=(10, 8))
        plt.grid(color='black')
        x = np.linspace(0, 10, 100)

        x_states = []
        y_states = []
        for i in range(100):
            x_states.append(self.states[i][0])
            y_states.append(self.states[i][1])

        plt.xlabel('Time (s)')
        plt.ylabel('x (ft)')
        plt.plot(x, x_states, 'b')
        plt.show()

        plt.style.use('seaborn-poster')
        plt.figure(figsize=(10, 8))
        plt.grid(color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('y (ft)')
        plt.plot(x, y_states, 'b')
        plt.show()
