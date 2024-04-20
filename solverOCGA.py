from ast import literal_eval
from copy import deepcopy
import random
import sys
import time
from typing import List, Tuple
import numpy as np
import pandas as pd


"""
Results from some quick hyperparameter tuning:

(pop size, seconds, mean cost)
 150,  2, 82.8k
 150,  5, 79.1k
1000,  5, 82.6k
 100,  5, 77.6k
  50,  5, 74.8k
  75,  5, 76.3k
  30,  5, 71.7k
  30, 20, 62.5k
  50, 20, 65.4k
  20, 20, 61.1k <-- best result
"""

MAX_ROUTE_LENGTH = 12 * 60


def cost(routes: List[List[int]], distance_matrix: np.ndarray) -> float:
    """
    Computes the cost function for driving the set of given routes. Takes into account number of drivers and total drive time.
    """
    number_of_drivers = len(routes)
    total_number_of_driven_minutes = sum(
        [get_route_length(r, distance_matrix) for r in routes]
    )

    return number_of_drivers * 500 + total_number_of_driven_minutes


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_route_length(route: List[int], distance_matrix: np.ndarray) -> float:
    """
    Computes the length of a cyclic route starting at the origin and making each delivery in route in order
    """
    total_length = 0
    for i, j in zip([0] + route, route + [0]):
        total_length += distance_matrix[i, j]

    return total_length


class OCGAGene:
    """
    A gene for the Optimized Crossover Genetic Algorithm detailed here:
    https://doi.org/10.1016/j.apm.2011.08.010

    Algorithm created by Nazif and Lee in 2012.

    Each gene is a permutation of the set of deliveries to be made,
    and the beginning and end of routes is determined greedily.
    """

    @staticmethod
    def make_random(gene_length, distance_matrix):
        """
        Create a gene corresponding to a random route.
        """
        perm = random.sample(range(1, gene_length + 1), gene_length)
        return OCGAGene(perm, distance_matrix)

    def __init__(self, perm: List[int], distance_matrix):
        self.perm = perm
        self.routes = self._convert_to_routes(distance_matrix)
        self.cost = cost(self.routes, distance_matrix)
        self.distance_matrix = distance_matrix

    def _convert_to_routes(self, distance_matrix) -> List[List[int]]:
        """
        Parses the permutation in the gene into blocks that each correspond to one route.
        Partitioning is done greedily, following the design of the original OCGA.
        """
        # Start with one empty route
        routes = [[]]

        # now add in every point in order
        for i in self.perm:
            # if the added point makes the route too long, start a new one
            if get_route_length(routes[-1] + [i], distance_matrix) > MAX_ROUTE_LENGTH:
                routes.append([i])
            else:
                # otherwise append to the current route
                routes[-1].append(i)

        return routes

    def randomize(self):
        """
        Used for when two genes are found to be duplicates of one another. Randomizes the permutation of this gene in-place.
        """
        gene_length = len(self.perm)
        self.perm = random.sample(range(1, gene_length + 1), gene_length)
        self.routes = self._convert_to_routes(self.distance_matrix)
        self.cost = cost(self.routes, self.distance_matrix)

    def swap_node_mutation(self):
        """
        Performs a mutation that swaps two nodes in the gene, and returns the result.
        Does not modify in-place.
        """
        # select a random pair of elements
        a, b = random.sample(range(len(self.perm)), 2)

        # swap them and return the result
        new_perm = deepcopy(self.perm)
        new_perm[a] = self.perm[b]
        new_perm[b] = self.perm[a]

        return OCGAGene(new_perm, self.distance_matrix)

    def inversion_mutation(self):
        """
        Performs a mutation that inverts a subsequence of the gene, and returns the result.
        Does not modify in-place.
        """
        # swap a random substring
        a, b = random.sample(range(len(self.perm)), 2)

        new_perm = deepcopy(self.perm)

        new_perm[a:b] = new_perm[a:b][::-1]

        return OCGAGene(new_perm, self.distance_matrix)

    def swap_string_mutation(self):
        """
        Performs a mutation that swaps two substrings of the gene, and returns the result.
        Does not modify in-place.
        """
        # select a random pair of substrings
        a1, a2, b1, b2 = sorted(random.sample(range(len(self.perm)), 4))

        new_perm = deepcopy(self.perm)

        new_perm[b1:b2] = self.perm[a1:a2]
        new_perm[a1:a2] = self.perm[b1:b2]

        return OCGAGene(new_perm, self.distance_matrix)


def get_permutation_inverse(permutation: List[int]) -> List[int]:
    """
    Computes the inverse of a permutation, i.e. p[p_inv[i]] = p_inv[p[i]] = i.
    """
    result = [0] * len(permutation)
    for i in range(len(permutation)):
        result[permutation[i] - 1] = i + 1

    return result


def convert_int_to_bit_string(num: int, string_length: int) -> List[int]:
    """
    Converts an int to a little-endian bit string.
    """
    bit_string = []
    for _ in range(string_length):
        bit_string.append(num % 2)
        num //= 2

    return bit_string


def generate_random_bit_strings(num_strings, string_length):
    """
    Generates a selection of random bitstrings of length string_length.
    If num_strings < 2**10 they are gauranteed to be unique, otherwise they are generated independently.
    """
    if string_length < 10:
        return [
            convert_int_to_bit_string(num, string_length)
            for num in random.sample(range(2**string_length), num_strings)
        ]
    else:
        strings = []
        for _ in range(num_strings):
            string = []
            for _ in range(string_length):
                string.append(random.randint(0, 1))
            strings.append(string)
        return strings


def create_crossover_child(
    p1: OCGAGene, p2: OCGAGene, cycles: List[List[int]], decision_variables: List[bool]
) -> OCGAGene:
    """
    Creates a child gene from optimized crossover, as detailed by Nazif and Lee in https://doi.org/10.1016/j.apm.2011.08.010, section 2.2.

    The child is created from cycles that are either taken from p1 or p2. Non-cycles are the same for both parents.
    """
    new_perm = deepcopy(p1.perm)
    for decision, cycle in zip(decision_variables, cycles):
        if decision:
            for i in cycle:
                new_perm[i - 1] = p2.perm[i - 1]

    return OCGAGene(new_perm, p1.distance_matrix)


def perform_optimized_crossover(
    p1: OCGAGene, p2: OCGAGene, max_children_considered=32
) -> Tuple[OCGAGene, OCGAGene]:
    """
    Performes optimized crossover, as detailed by Nazif and Lee in https://doi.org/10.1016/j.apm.2011.08.010, section 2.2.

    Returns two children; an optimal o-child, and an exploratory e-child.
    """

    # we'll need the permutation inverse to find cycles
    p2_perm_inv = get_permutation_inverse(p2.perm)

    # list out decisions, e.g. cycles in the bipartite graph created with
    # the successor function given by S(i) = p2_perm_inv[p1.perm[i]]

    cycles = []
    fixed_points = []
    remaining = deepcopy(p1.perm)
    while len(remaining) > 0:
        cycle = []
        next_element = remaining[0]
        while True:
            cycle.append(next_element)
            remaining.remove(next_element)
            # routes and cycles are 1-indexed, but python lists are 0-indexed, hence the index shifts.
            next_element = p2_perm_inv[p1.perm[next_element - 1] - 1]
            if next_element == cycle[0]:
                break

        if len(cycle) > 1:
            cycles.append(cycle)
        else:
            fixed_points.append(cycle[0])

    # sample up to max_children_considered decisions (bit strings)
    num_decisions = len(cycles)
    bit_strings = generate_random_bit_strings(
        num_strings=min(max_children_considered, 2**num_decisions),
        string_length=num_decisions,
    )

    # both parents will be considered for the optimal child
    if p1.cost < p2.cost:
        o_child = OCGAGene(deepcopy(p1.perm), p1.distance_matrix)
        o_child_bits = [False] * num_decisions
    else:
        o_child = OCGAGene(deepcopy(p2.perm), p2.distance_matrix)
        o_child_bits = [False] * num_decisions

    # construct potential children from decision strings and choose the lowest cost one to be the optimal-child
    for bit_string in bit_strings:
        potential_o_child = create_crossover_child(p1, p2, cycles, bit_string)

        if potential_o_child.cost < o_child.cost:
            o_child = potential_o_child
            o_child_bits = bit_string

    # create the exploratory-child by inverting decisions that lead to o-child
    e_child_bits = [not b for b in o_child_bits]
    e_child = create_crossover_child(p1, p2, cycles, e_child_bits)

    return o_child, e_child


def select_parents(population: List[OCGAGene], p_s=0.75) -> Tuple[OCGAGene, OCGAGene]:
    """
    Chooses two parents from the population, from which we'll generate offspring.

    Keeping in line with Nazif and Lee in https://doi.org/10.1016/j.apm.2011.08.010,
    the selection method is a probabilistic binary tournament.

    Each parent is found by:
    1) choosing two members of the population at random
    2) selecting the more fit one with likelihood p_s

    Returns two parent genes.
    """
    # select first parent
    pair = random.sample(population, k=2)

    # sort so that pair[0] is lower cost
    pair.sort(key=lambda x: x.cost)

    # choose the lower cost (higher fitness) parent with likelihood p_s
    if random.random() < p_s:
        p1 = pair[0]
    else:
        p1 = pair[1]

    # now repeate the process to select the second parent
    pair = random.sample(population, k=2)
    pair.sort(key=lambda x: x.cost)
    if random.random() < p_s:
        p2 = pair[0]
    else:
        p2 = pair[1]

    return p1, p2


def produce_offspring(
    p1: OCGAGene, p2: OCGAGene, p_c=0.75
) -> Tuple[OCGAGene, OCGAGene]:
    """
    Creates offspring from two parents. This follows the method outlined
    by Nazif and Lee in https://doi.org/10.1016/j.apm.2011.08.010.
    """
    # crossover is applied at random
    use_crossover = random.random() < p_c

    if use_crossover:
        o_child, e_child = perform_optimized_crossover(p1, p2)
        return o_child, e_child
    else:
        # if we don't use crossover, we apply swap node mutations to both parents
        return p1.swap_node_mutation(), p2.swap_node_mutation()


def cull_and_filter(
    population: List[OCGAGene], pop_size, apply_filter
) -> List[OCGAGene]:
    """
    Removes the more costly half of the population, and optionally
    filters out all duplicate genes, replacing them with random ones.
    """
    # sort population from lowest cost to highest
    population.sort(key=lambda x: x.cost)

    # Elitism replacement (remove highest cost genes)
    population = population[:pop_size]

    # Filtration (remove duplicates, and replace with random genes)
    if apply_filter:
        for i in range(1, pop_size):
            if population[i].perm == population[i - 1].perm:
                population[i - 1].randomize()

    return population


def OCGA(distance_matrix, pop_size, time_limit_seconds) -> List[List[int]]:
    """
    An implementation of the Optimized Crossover Genetic Algorithm,
    created by Nazif and Lee in 2012 - see https://doi.org/10.1016/j.apm.2011.08.010

    From paper: Optimised crossover genetic algorithm for capacitated vehicle routing problem

    The algorithm has been modified to run until time_limit_seconds is up, rather than for a set number of generations.

    Returns the best route discovered in the time limit.
    """
    start_time = time.time()
    end_time = start_time + time_limit_seconds

    # There are the same number of ints in a gene as points to visit
    gene_length = len(distance_matrix) - 1

    # initialize population (includes computing fitness)
    pop = []
    for _ in range(pop_size):
        pop.append(OCGAGene.make_random(gene_length, distance_matrix))

    generation = 0
    while time.time() < end_time:
        generation += 1

        # produce offspring
        children = []
        for _ in range(pop_size // 2):
            p1, p2 = select_parents(pop)
            children += produce_offspring(p1, p2)

        # mutate population
        for i in range(pop_size):
            if random.random() < 0.5:
                pop[i] = pop[i].inversion_mutation()
            else:
                pop[i] = pop[i].swap_string_mutation()

        # mix children into the population
        pop += children

        # now remove the least fit genes, and possibly duplicates
        filter_duplicates = generation % 50 == 0
        pop = cull_and_filter(pop, pop_size, filter_duplicates)

    pop.sort(key=lambda x: x.cost)
    return pop[0].routes


if __name__ == "__main__":
    # get the path to the problem specification file
    path_to_file = sys.argv[1]

    # load and parse the data
    data = pd.read_csv(path_to_file, delimiter=" ")
    data = data.set_index("loadNumber")
    data.pickup = data.pickup.apply(literal_eval)
    data.dropoff = data.dropoff.apply(literal_eval)

    # precompute a distance matrix (this speeds up computation roughly 4-fold)
    # d[i,j] = distance along path from dropoff[i] to pickup[j] to dropoff[j]
    # point 0 is the origin
    distance_matrix = np.zeros((len(data) + 1, len(data) + 1), dtype=float)
    for i in range(len(data) + 1):
        for j in range(len(data) + 1):
            coming_from = (0, 0)
            pickup_point = (0, 0)
            dropoff_point = (0, 0)
            if i != 0:
                coming_from = data.dropoff.loc[i]

            if j != 0:
                pickup_point = data.pickup.loc[j]
                dropoff_point = data.dropoff.loc[j]

            dist_to_pickup = dist(coming_from, pickup_point)
            dist_to_dropoff = dist(pickup_point, dropoff_point)
            distance_matrix[i, j] = dist_to_pickup + dist_to_dropoff

    # these hyperparamters were selected via tuning on the training set
    # note that we've ended up with an unusually small population size;
    # 50 to 150 is more common to see in the literature
    solution = OCGA(distance_matrix, pop_size=20, time_limit_seconds=20)

    # print the result
    for route in solution:
        print(route)
