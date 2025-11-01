# MicroGrad: A Scalar Autograd Engine

This project is a minimal, educational implementation of an automatic differentiation (autograd) engine in Python. It defines a `Value` object that tracks operations to build a computation graph and efficiently computes gradients using backpropagation.

This implementation is heavily inspired by and follows the structure of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). It's intended as a learning tool to understand how neural networks and modern deep learning libraries like PyTorch work "under the hood."

The project also includes a small neural network library (`nn.py`) built on top of this autograd engine, capable of building and training Multi-Layer Perceptrons (MLPs).

## 🚀 Features

  * **`Value` Object:** A scalar value wrapper that tracks its "children" (the values it was computed from) and the operation used.
  * **Automatic Differentiation:** Computes gradients for any mathematical expression by traversing the computation graph.
  * **Neural Network Library:** Basic modules for building neural networks:
      * `Neuron`
      * `Layer`
      * `MLP` (Multi-Layer Perceptron)
  * **Activations:** Includes `tanh` and `ReLU` activation functions.
  * **Visualization:** Utility functions to draw the computation graph using `graphviz`.

## 📂 Project Structure

The project is split into logical components for clarity and maintainability:

```
micrograd_project/
├── micro_grad.py      # The core autograd engine (Value class)
├── nn.py              # Neural network library (Neuron, Layer, MLP)
├── viz.py             # Visualization utilities (draw_dot)
├── train_moons.py     # Main runnable script to train an MLP on the moons dataset
├── examples.ipynb     # Jupyter notebook with step-by-step examples
├── requirements.txt   # Python dependencies
└── README.md          # You are here!
```

## 🛠️ How to Use

### 1\. Installation

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/YOUR_USERNAME/micrograd_project.git
cd micrograd_project
```

Next, it's highly recommended to create a virtual environment:

```bash
python -m venv venv
source venv\Scripts\activate
```

Finally, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2\. Run the Main Example

To see the neural network in action, run the `train_moons.py` script. This will train an MLP to classify the "moons" dataset and generate a plot of the decision boundary.

```bash
python train_moons.py
```

### 3\. Explore the Examples

For a detailed, step-by-step breakdown of how the `Value` object works, from simple derivatives to backpropagation in a full neuron, open the Jupyter Notebook:

```bash
jupyter notebook examples.ipynb
```

## 📋 Requirements

  * `numpy`
  * `matplotlib`
  * `graphviz` (You may also need to install the Graphviz binary. See the [Graphviz download page](https://graphviz.org/download/) for instructions.)
  * `scikit-learn`
  * `jupyter` (For running the notebook)

## 🙏 Acknowledgements

This code is a Python implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). All credit for the original idea, structure, and educational content goes to him.