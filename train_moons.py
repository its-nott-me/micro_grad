import matplotlib.pyplot as plt
import numpy as np
from nn import Neuron, Layer, MLP
from micro_grad import Value
from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1

plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')

model = MLP(2, [16, 16, 1], activation=['relu', 'relu', 'linear'])
print(model)
print(f"Number of parameters: {len(model.parameters())}")

def loss(batch_size=None):

  if batch_size is None:
    Xb, yb = X, y
  else:
    rand_idx = np.random.permutaion(X.shape[0])[:batch_size]
    Xb, yb = X[rand_idx], y[rand_idx]
  inputs = [list(map(Value, xrow)) for xrow in Xb]

  scores = list(map(model, inputs))
  # max-margin loss
  losses = [(1 + -y_i*score_i).relu() for y_i, score_i in zip(yb, scores)]
  data_loss = sum(losses) * (1.0/len(losses))
  # L2 regularization
  alpha = 1e-4
  reg_loss = alpha * sum((p*p for p in model.parameters()))
  total_loss = data_loss + reg_loss

  accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
  return total_loss, sum(accuracy)/len(accuracy)

total_loss, acc = loss()
print(total_loss.data, acc)

epochs = 100
for k in range(epochs):

  total_loss, acc = loss()

  model.zero_grad()
  total_loss.backward()

  lr = 1.0 - 0.9*k/epochs
  for p in model.parameters():
    p.data -= lr*p.grad

  if k % 10 == 0:
    print(f"epoch: {k} | loss: {total_loss.data} | accuracy: {acc*100}%")

h = 0.25

xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 0].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                     np.arange(ymin, ymax, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list((map(model, inputs)))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.jet)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())