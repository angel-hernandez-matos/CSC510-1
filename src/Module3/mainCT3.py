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
    def __init__(self, times = 1500, render_network = False, input_size = 1,
                 hidden_size = 1, output_size = 1, number_to_predict = 5):
        self.train_times = times
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.render_ann = render_network
        self.number_to_predict = number_to_predict

class ShallowANN:
    def __init__(self, config = ANNConfig()):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        self.__np = np
        self.__nx = nx
        self.__plt = plt
        self.__config = config
        self.__learning_rate = 0.1

    # Activation function
    def __sigmoid(self, x):
        return 1 / (1 + self.__np.exp(-x))

    # derivative assumes x = sigmoid(x)
    # Required to adjust weights in the direction that reduces the loss, and that direction is
    # determined by gradients. Impossible to calculate without derivative
    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    def train_ann(self):
        # Training data
        _input = self.__np.array([[1], [2], [3], [4]], dtype=float)
        output = self.__np.array([[2], [3], [4], [5]], dtype=float)

        # Normalization
        _input = _input / self.__np.max(_input)
        output = output / self.__np.max(output)

        # Initialize weights and biases
        self.__np.random.seed(50)
        weight1 = self.__np.random.randn(self.__config.input_size, self.__config.hidden_size)
        bias1 = self.__np.random.randn(1, self.__config.hidden_size)
        weight2 = self.__np.random.randn(self.__config.hidden_size, self.__config.output_size)
        bias2 = self.__np.random.randn(1, self.__config.output_size)

        # Training loop
        for i in range(self.__config.train_times):
            # Feedforward
            _1 = self.__np.dot(_input, weight1) + bias1
            sigmoid1 = self.__sigmoid(_1)
            _2 = self.__np.dot(sigmoid1, weight2) + bias2
            y_hat = self.__sigmoid(_2)

            # Loss (Mean squared error)
            loss = self.__np.mean((output - y_hat) ** 2)

            # Backpropagation
            error_output = (y_hat - output) * self.__sigmoid_derivative(y_hat)

            # Hidden layer error
            error_hidden = self.__np.dot(error_output, weight2.T) * self.__sigmoid_derivative(sigmoid1)

            # Gradient descent (Weight updates)
            weight2 -= self.__learning_rate * self.__np.dot(sigmoid1.T, error_output)
            bias2 -= self.__learning_rate * self.__np.sum(error_output, axis=0, keepdims=True)
            weight1 -= self.__learning_rate * self.__np.dot(_input.T, error_hidden)
            bias1 -= self.__learning_rate * self.__np.sum(error_hidden, axis=0, keepdims=True)

            # Print loss every 150 iterations
            if i % 150 == 0: print(f"Iteration {i}, Loss: {loss:.6f}")

        return _input, output, weight1, bias1, weight2, bias2

    def test_network(self, _input, output, weight1, bias1, weight2, bias2):
        test_result = self.__np.array([[self.__config.number_to_predict]]) / self.__np.max(_input)
        hidden_value = self.__sigmoid(self.__np.dot(test_result, weight1) + bias1)
        prediction = self.__sigmoid(self.__np.dot(hidden_value, weight2) + bias2)
        print(f"\nPrediction for input {self.__config.number_to_predict}:", prediction * self.__np.max(output))

    def print_network_if_needed(self, weight1, weight2):
        if self.__config.render_ann:
            g = self.__nx.DiGraph()

            # Node names
            input_nodes = [f"Input {i + 1}" for i in range(self.__config.input_size)]
            hidden_nodes = [f"Hidden {h + 1}" for h in range(self.__config.hidden_size)]
            output_nodes = [f"Output {o + 1}" for o in range(self.__config.output_size)]

            # Add nodes with positions
            for i, n in enumerate(input_nodes): g.add_node(n, pos=(0, i))
            for h, n in enumerate(hidden_nodes): g.add_node(n, pos=(1, h))
            for o, n in enumerate(output_nodes): g.add_node(n, pos=(2, o))

            # Add edges input to hidden
            for i, in_node in enumerate(input_nodes):
                for h, hid_node in enumerate(hidden_nodes):
                    w = weight1[i, h]
                    g.add_edge(in_node, hid_node, weight=w)

            # Add edges hidden to output
            for h, hid_node in enumerate(hidden_nodes):
                for o, out_node in enumerate(output_nodes):
                    w = weight2[h, o]
                    g.add_edge(hid_node, out_node, weight=w)

            # Draw
            pos = self.__nx.get_node_attributes(g, "pos")
            weights = self.__nx.get_edge_attributes(g, "weight")

            self.__plt.figure(figsize=(8, 4))
            self.__nx.draw(g, pos, with_labels=True, node_size=1200, node_color="lightblue", font_size=10,
                           arrows=True, arrowstyle="-|>", arrowsize=15)

            # Color edges by sign, thickness by magnitude
            edges = g.edges()
            colors = ["green" if weights[e] > 0 else "red" for e in edges]
            widths = [1 + 2 * abs(weights[e]) for e in edges]

            self.__nx.draw_networkx_edges(g, pos,  edge_color=colors, width=widths, arrows=True,
                                          arrowstyle="-|>", arrowsize=15)

            self.__plt.gcf().canvas.manager.set_window_title("Artificial Neural Network Architecture")
            self.__plt.axis("off")
            self.__plt.show()

class TestCaseRunner:
    @staticmethod
    def run_test():
        ann_instance = ShallowANN(TestCaseRunner.__input_runtime_config())
        _input, output, weight1, bias1, weight2, bias2 = ann_instance.train_ann()
        ann_instance.test_network(_input, output, weight1, bias1, weight2, bias2)
        ann_instance.print_network_if_needed(weight1, weight2)

    @staticmethod
    def __input_runtime_config():
         # single‑input, single‑output (SISO) - Shallow Neural Network
         input_size = 1 # Just one input to train the model
         output_size = 1 # Just one output
         prediction = 5
         num_times = input("Specify number of times to train ANN (default is 1500):")
         hidden_size = input("Specify the hidden size (default is 1):")
         show_network = input("Show visual representation of network? (y/n - (default is n): ")
         print("Predicting number 5")
         num_times = int(num_times) if num_times.isdigit() and int(num_times) > 0 else 1500
         hidden_size = int(hidden_size) if hidden_size.isdigit() and int(hidden_size) > 0 else 1
         show_network = False if not show_network.strip() or show_network == 'n' else True
         return ANNConfig(num_times, show_network, input_size, hidden_size, output_size, prediction)

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