# File: mainCT3.py
# Written by: Angel Hernandez
# Description: Module 3 - Critical Thinking
# Requirement(s): Using your research and resources, write a basic 2-layer Artificial Neural Network utilizing static
# backpropagation using Numpy in Python. Your neural network can perform a basic function, such as guessing the next
# number in a series. Using the activation function of your choice to calculate the predicted output ŷ, known as the
# feedforward function, and updating the weights and biases through gradient descent (backpropagation) based on your
# choice of a basic loss function.

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

class ANNConfig:
    def __init__(self, times = 1000, render_network = False):
        self.train_times = times
        self.render_ann = render_network

class ShallowANN:
    def __init__(self, config = ANNConfig()):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        self.__np = np
        self.__nx = nx
        self.__plt = plt
        self.__config = config

class TestCaseRunner:
    @staticmethod
    def run_test():
        ann_instance = ShallowANN(TestCaseRunner.__input_runtime_config())

    @staticmethod
    def __input_runtime_config():
         num_times = int(input("Specify number of times to train ANN (default is 1000):"))
         show_network = bool(input("Show visual representation of network? (y/n): (default is y"))
         num_times = 1000 if num_times < 1 else num_times
         show_network = True if show_network else False
         return ANNConfig(num_times, show_network)

def clear_screen():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def main():
    try:
        dependencies = ['numpy', 'matplotlib', 'networkx']
        for d in dependencies: DependencyChecker.ensure_package(d)
        clear_screen()
        print('*** Module 3 - Critical Thinking ***\n')
        TestCaseRunner.run_test()
    except Exception as e:
        print(e)

if __name__ == '__main__': main()