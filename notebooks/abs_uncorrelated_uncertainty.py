'''
We consider the same scenario as in abs_correlated_uncertainty, but this time we
disallow correlations between network weights/biases. That is, they all need to
be sampled independently.

To simplify things, we start by considering a case where the network has learned
(to high confidence) two independent networks, but hasn't learned to favour one
over the other.
'''

import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-5,5,100)

# ReLU
r = np.vectorize(lambda x: x if x > 0. else 0.)

# The only uncertain variables are those determining the weightings given to the
# subnets.
def plot_it(s, q):
    plt.plot(xs, s * (r(xs-1)+r(-xs-1)+1) + q * (r(xs)+r(-xs)))

# Using a standard deviation of 0.1 provides a good range of values in [-1,1],
# but fits the data poorly.
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0.5,0.1)
    q = np.random.normal(0.5,0.1)
    plot_it(s, q)
plt.show()

# A smaller standard deviation fits better, but gives a smaller confidence
# interval for the unknown region.
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0.5,0.01)
    q = np.random.normal(0.5,0.01)
    plot_it(s, q)
plt.show()

# We can add more than two subnets.
def plot_them(vs):
    ss = np.linspace(0,1,len(vs))
    plt.plot(xs, sum([v * (r(xs-s)+r(-xs-s)+s) for s,v in zip(ss,vs)]))

# But this actually has the same problem, it's just smoother. Our only choice is
# still to average over the subnets, and because we can't ensure we get the same
# total weight each time, we need to keep the standard deviation low.
num_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(1./num_subnets,0.01,num_subnets)
    plot_them(s)
plt.show()

# However, a solution is to do the subnets in a different way.

# We learn one subnet that fits the data.
fitter = lambda x: r(x-1) + r(-x-1) + 1
plt.plot(xs,abs(xs))
plt.plot(xs,fitter(xs))
plt.show()

# And then we learn a bunch of subnets that are zero on the data.
junk = lambda s,x: s * (r(x + 1) + r(x - 1) - 2 * r(x))

def plot_smart(vs):
    ss = np.linspace(0,1,len(vs))
    plt.plot(xs, fitter(xs) + sum([v * junk(s,xs) for s,v in zip(ss,vs)]))

# This provides a big range of guesses. The actual range will be determined by
# the ratio between the randomness and regularisation weightings.
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0,0.5,num_subnets)
    plot_smart(s)
plt.show()
