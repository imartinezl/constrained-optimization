# CONSTRAINED OPTIMIZATION WITH GRADIENT DESCENT METHODS

# min f(x)
# subject to
# g(x) == 0
# h(x) <= 0

# %%

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax, random

# https://github.com/crowsonkb/mdmm-jax
# https://papers.nips.cc/paper_files/paper/1987/file/a87ff679a2f3e71d9181a67b7542122c-Paper.pdf
# https://stackoverflow.com/questions/12284638/gradient-descent-with-constraints-lagrange-multipliers
# https://users.wpi.edu/~pwdavis/Courses/MA1024B10/1024_Lagrange_multipliers.pdf

damping = 1
weight = 0.1
n = 2

def objective(x):
    return (x[0]-1.2)**2 + 2*(x[1] - 1.2)**2

def inequality_constraint(x):
    return jnp.array([ lax.clamp(0.0, x[0], 3.0)-x[0], lax.clamp(0.0, x[1], 2.0)-x[1], x[0]+x[1]-1.0])

def lagrangian(x):
    return objective(x[:n]) + jnp.dot(x[n:], inequality_constraint(x[:n])) 

@jax.jit
def loss(x):
    return lagrangian(x) + weight * jnp.sum(damping*inequality_constraint(x)**2)/2

grad_loss = jax.jit(jax.grad(loss))

def constrained_optimization(x_init, max_iter=500):
    x = x_init
    
    loss_values, x_values = [], []
    for i in range(max_iter):
        gradient = grad_loss(x)

        p, alpha = x[:n], x[n:]
        lr = 0.1
        p -= lr*gradient[:n]
        alpha += lr*gradient[n:]
        x = jnp.concatenate([p, alpha])
        
        loss_values.append(loss(x))
        x_values.append(x[:n])

    return x, loss_values, x_values

key = random.PRNGKey(0)

p = random.normal(key, (n,))
# p = jnp.zeros((n,))
p = jnp.ones((n,))*(1)
alpha = jnp.zeros_like(inequality_constraint(p))
x = jnp.concatenate([p,alpha])

x_opt, loss_values, x_values = constrained_optimization(x)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_values)
plt.figure()
plt.plot(x_values)
plt.show()
print("Optimal solution:", x_opt, objective(x_opt))

# %%
