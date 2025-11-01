import math

class Value:

  def __init__(self, data, _children=(), _op='', label='', grad=0.0):
    self.data = data
    self.label = label
    self.grad = grad
    self._backward = lambda : None
    self._prev = set(_children)   # children nodes
    self._op = _op                # the operation that produced this node

  # to print in a readable format
  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  # ---------------- mathematical operations ---------------
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), _op='+')
    def _backward():
      self.grad += 1 * out.grad      # local derivative * gradient
      other.grad += 1 * out.grad     # '+=' as we have to accumulate gradients from all parents..
    out._backward = _backward
    return out

  def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data - other.data, (self, other), _op='-')
    def _backward():
      self.grad += 1 * out.grad
      other.grad -= 1 * out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), _op='*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __truediv__(self, other):
    # a/b = a*(1/b) = a*(b**-1)
    return self * other**-1

  def __radd__(self, other):    # other, self
    return self + other

  def __rsub__(self, other):    # other, self
    return Value(other) - self

  def __rmul__(self, other):    # other, self
    return self * other         # this will call self.__mul__(other)

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')
    def _backward():
      self.grad += other * (self.data ** (other-1)) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  # --------------- activations ------------------

  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t, (self, ), 'tanh')
    def _backward():
      self.grad += (1-t**2) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  # --------------- backward fxn ----------------
  def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
