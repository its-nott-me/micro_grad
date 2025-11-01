import random
from micro_grad import Value

class Model:

  def zero_grad(self):
    for n in self.parameters():
      n.grad = 0.0

  def parameters(self):
    return []

class Neuron(Model):

  def __init__(self, nin, activation='linear'):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.activation = activation

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)    # self.b is used as the start value

    if self.activation == 'relu':
      out = act.relu()
    elif self.activation == 'tanh':
      out = act.tanh()
    else:
      out = act

    return out

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{self.activation} Neuron({len(self.w)})"

class Layer(Model):                # a list of neurons

  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):    # forward pass
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

  def __repr__(self):
    return f"[Layer of {[str(neuron) for neuron in self.neurons]}]"

class MLP(Model):                  # list of layers

  def __init__(self, nin, nouts, activation=None):
    sz = [nin] + nouts      # concat
    if activation is None:
      activation = ['relu']*(len(nouts) - 1) + ['linear']
    self.layers = [Layer(sz[i], sz[i+1], activation=activation[i]) for i in range(len(nouts))]

  def __call__(self, x):    # forward pass
    for layers in self.layers:
      x = layers(x)         # pass x to first layer get output, pass output to second layer get second_output ... pass n-1_output to final layer and get final_output
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def __repr__(self):
    return f"MLP of [{',\n '.join(str(layer) for layer in self.layers)}]"
