# Capacitated Vehicle Routing Problem

This repo contains a Python implementation of a modified version of the genetic algorithm presented in Nazif and Lee's 2012 paper [Optimised crossover genetic algorithm for capacitated vehicle routing problem](https://www.sciencedirect.com/science/article/pii/S0307904X11005105). The following changes were made:
- Instead of using "number of generations > cutoff" as a stopping condition, this algorithm runs for a set amount of time.
- When sampling children from the set of crossover possibilities, both parents are gauranteed to fall into the candidate set.

The population size has been tuned to optimize performance across the test set of training problems contained in `trainingProblems`.

## Running the Code

The code relies on `numpy` and `pandas`, so before running install them with `python3 -m pip install numpy pandas`.

The code's performance on the benchmark can then be run with `python3 evaluateShared.py --cmd "python3 solverOCGA.py" --problemDir trainingProblems`. The average cost acheived is 61189.

A greedy solver has been included for comparison, it can be evaluated with `python3 evaluateShared.py --cmd "python3 solverGreedy_.py" --problemDir trainingProblems`. It produces an average cost of 88512.