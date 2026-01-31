# File: mainCT4.py
# Written by: Angel Hernandez
# Description: Module 4 - Critical Thinking
#
# Requirement(s):  Define a simple real-world search problem requiring a heuristic solution. You can base the problem
# on the 8-puzzle (or n-puzzle) problem, Towers of Hanoi, or even Traveling Salesman. The problem and solution can be
# utilitarian or entirely inventive. Write an interactive Python script (using either simpleAI's library or your
# resources) that utilizes either Best-First search, Greedy Best First search, Beam search, or A* search methods
# to calculate an appropriate output based on the proposed function. The search function does not have to be optimal
# nor efficient but must define an initial state, a goal state, reliably produce results by finding the sequence
# of actions leading to the goal state
#
# Validated the results here: https://www.mathsisfun.com/games/towerofhanoi.html

import os
import sys
import subprocess

class DependencyChecker:
    @staticmethod
    def ensure_package(package_name):
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing missing package: {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' was installed successfully.")

class RuntimeConfig:
    number_disks = 0
    def __init__(self, number_disks = 3):
        self.number_disks = number_disks

class TowerOfHanoiProblem:
    @property
    def problem(self):
        return self._problem

    def __init__(self, config = RuntimeConfig()):
        self.__config = config
        from simpleai.search import SearchProblem
        from simpleai.search import astar # Let's use A*
        self.__astar = astar

        class Problem(SearchProblem):
            def __init__(self, _config):
                self.__config = _config
                initial_state = (tuple(range(1, self.__config.number_disks + 1)),(),())
                super().__init__(initial_state) #Passing initial state (configuration) to SearchProblem base class

            def is_goal(self, state):
                return len(state[2]) == self.__config.number_disks

            def actions(self, state):
                actions = []
                for x in range(3): # We have 3 pegs
                    if not state[x]:
                        continue
                    disk = state[x][0]
                    for z in range(3):
                        if x != z and (not state[z] or disk < state[z][0]):
                            actions.append((x, z))
                return actions

            def result(self, state, action):
                x, z = action
                new_state = [list(peg) for peg in state]
                disk = new_state[x].pop(0)
                new_state[z].insert(0, disk)
                return tuple(tuple(peg) for peg in new_state)

            def cost(self, state, action, state2):
                return 1

            def heuristic(self, state):
                return self.__config.number_disks - len(state[2])

        self._problem = Problem(self.__config)

    def resolve_and_print(self):
        result = self.__astar(self._problem)
        print("\n[== Tower of Hanoi - Solution found ==]")
        print(f"Number of moves: {len(result.path()) - 1}")
        print("\n[== Sequence of moves (from peg to peg) ==]")
        for a, s in result.path():
            if a is not None: print(f"Move disk from peg {a[0]} to peg {a[1]}")

class TestCaseRunner:
    @staticmethod
    def run_test():
        tower_problem = TowerOfHanoiProblem(TestCaseRunner.__input_runtime_config())
        tower_problem.resolve_and_print()

    @staticmethod
    def __input_runtime_config():
        number_disks = input("Specify number of disks (default is 3):")
        try:
            # Search algorithms become computationally expensive for Towers of Hanoi due to exponential
            # growth in the state space. For demonstration purposes, this implementation limits the number of disks
            # to 4 to ensure reasonable runtime.
            number_disks = int(number_disks)
            number_disks = 3 if number_disks > 4 else number_disks
        except ValueError:
            number_disks = 3
        return RuntimeConfig(number_disks)

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['simpleai']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 4 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()