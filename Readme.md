# MicroGrad: A Scalar Autograd Engine

This project is a minimal, educational implementation of an automatic differentiation (autograd) engine in Python. It defines a `Value` object that tracks operations to build a computation graph and efficiently computes gradients using backpropagation.

This implementation is heavily inspired by and follows the structure of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It's intended as a learning tool to understand how neural networks and modern deep learning libraries like PyTorch work "under the hood."

The project also includes a small neural network library (`nn.py`) built on top of this autograd engine, capable of building and training Multi-Layer Perceptrons (MLPs).

---

## 🚀 Features

* **`Value` Object:** A scalar wrapper that tracks its "children" (the values it was computed from) and the operation used.
* **Automatic Differentiation:** Computes gradients for any mathematical expression by traversing the computation graph.
* **Neural Network Library:** Basic modules for building neural networks:

  * `Neuron`
  * `Layer`
  * `MLP` (Multi-Layer Perceptron)
* **Activations:** Includes `tanh` and `ReLU` activation functions.
* **Visualization:** Utility functions to draw the computation graph using `graphviz`.

---

## 📂 Project Structure

```
micrograd_project/
├── micro_grad.py       # Core autograd engine (Value class)
├── nn.py               # Neural network library (Neuron, Layer, MLP)
├── viz.py              # Visualization utilities (draw_dot)
├── train_moons.py      # Script to train an MLP on the moons dataset
├── plot_neural_net.py  # Visualizes a small neural network computation graph
├── examples.ipynb      # Jupyter notebook with interactive examples
├── requirements.txt    # Python dependencies
└── README.md           # You are here :)
```

---

## 🛠️ How to Use

### 1. Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/its-nott-me/micro_grad
cd micro_grad
```

Create a virtual environment and activate it:

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Train a Model

Run the main example to train an MLP on the “moons” dataset:

```bash
python train_moons.py
```

This will train a small network and display the decision boundary.

---

### 3. Visualize a Neural Network

To visualize how the computation graph is built and gradients flow through the model, run:

```bash
python plot_neural_net.py
```

This will:

* Build a small 2→2→1 neural network.
* Run a forward pass on a sample input.
* Generate a computation graph (`simple_net.svg`) showing how values and operations connect.

Example output (simplified view):

![Computation Graph Example](assets/simple_net_example.svg)

> 💡 The graph shows how each `Value` node connects through operations like `mul`, `add`, and activation functions — demonstrating how backpropagation works under the hood.

---

### 4. Explore the Notebook

For a step-by-step breakdown of how the `Value` object and backpropagation work, open:

```bash
jupyter notebook examples.ipynb
```

---

## 📋 Requirements

* `numpy`
* `matplotlib`
* `graphviz` (You may also need to install the Graphviz binary. See the [Graphviz download page](https://graphviz.org/download/))
* `scikit-learn`
* `jupyter` (for running notebooks)

---

## 🙏 Acknowledgements

This codebase is a Python reimplementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
All credit for the original concept and educational inspiration goes to him.
