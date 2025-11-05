from micro_grad import Value
from nn import MLP
from vis import draw_dot

# Create a small MLP: 2 inputs → 2 hidden neurons → 1 output
model = MLP(2, [2, 1], activation=['relu', 'linear'])

# Example input
x_example = [Value(1.0), Value(-0.5)]

# Forward pass
out = model(x_example)

# Print output value (for debugging)
print(f"Output Value: {out.data:.4f}")

# Draw computation graph
dot = draw_dot(out)

# Save graph to file (SVG format)
dot.render(filename="simple_net", format="svg", cleanup=True)

print("Computation graph saved to 'simple_net.svg'")
