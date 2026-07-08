import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
Takes the reachable sets of a 2d network and plots them
"""

# relu activation function
def relu_interval(z_low, z_high):
    return np.maximum(0, z_low), np.maximum(0,z_high)


# linear function
def linear_layer(x_low, x_high, W, b):
    y_low = []
    y_high = []

    for row, bias in zip(W, b):
        lower = bias
        upper = bias

        for w, low, high in zip(row, x_low, x_high):
            if w >= 0:
                lower += w * low
                upper += w * high
            else:
                lower += w * high
                upper += w * low
        
        y_low.append(lower)
        y_high.append(upper)

    return np.array(y_low), np.array(y_high)
    

# forward pass
def nn_forward(x_low, x_high, weights, biases):
    low = x_low
    high = x_high

    for i in range(len(weights)):
        low, high = linear_layer(low, high, weights[i], biases[i])

        if i < len(weights) - 1:
            low, high = relu_interval(low, high)

    return low, high


# applies dynamics
def dynamics_step(x_low, x_high, u_low, u_high, A, B):
    xu_low = np.concatenate([x_low, u_low])
    xu_high = np.concatenate([x_high, u_high])

    M = np.hstack([A, B])
    b = np.zeros(A.shape[0])

    return linear_layer(xu_low, xu_high, M, b)


# one step in the closed loop
def loop_step(x_low, x_high, weights, biases, A, B):
    
    u_low, u_high = nn_forward(x_low, x_high, weights, biases)

    next_low, next_high = dynamics_step(
        x_low,
        x_high,
        u_low,
        u_high,
        A,
        B
    )

    return next_low, next_high, u_low, u_high
# Weights and dynamics for the simulation:

weights = [
    np.array([
        [1.0, -0.5],
        [0.3,  0.8],
        [-0.7, 0.2]
    ]),

    np.array([
        [0.6, -1.2, 0.9]
    ])
]

biases = [
    np.array([0.1, -0.2, 0.0]),
    np.array([0.05])
]

A = np.array([
    [1.0, 0.3],
    [0.0, 1.0]
])

B = np.array([
    [0.4],
    [0.3]
])


# initial input intervals
x_low = np.array([-1.0, -0.5])
x_high = np.array([1.0, 0.5])

# steps
K = 5

# storage for intermediate boxes:
boxes = []
boxes.append((x_low.copy(), x_high.copy()))  # stores first set before running loop

# loop to generate the forward passes
for k in range(K):
    x_low, x_high, u_low, u_high = loop_step(x_low, x_high, weights, biases, A, B)
    boxes.append((x_low.copy(), x_high.copy()))

    print(f"Step {k + 1}")
    print(f"u in [{u_low}, {u_high}]")
    print(f"x1 in [{x_low[0]:.4f}, {x_high[0]:.4f}]")
    print(f"x2 in [{x_low[1]:.4f}, {x_high[1]:.4f}]")
    print()


# plot results
fig, ax = plt.subplots(figsize=(8, 8))

for k, (low, high) in enumerate(boxes):

    x_min = low[0]
    x_max = high[0]

    y_min = low[1]
    y_max = high[1]

    rect = Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        linewidth=2
    )

    ax.add_patch(rect)

    ax.text(
        x_min,
        y_max,
        f"X{k}"
    )

ax.grid(True)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Reachable Sets")

ax.axis("equal")

plt.savefig("reachable_sets.png")
print("saved")