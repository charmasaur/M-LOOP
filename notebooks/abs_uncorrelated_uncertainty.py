'''
We consider the same scenario as in abs_correlated_uncertainty, but this time we
disallow correlations between network weights/biases. That is, they all need to
be sampled independently.

To simplify things, we start by considering a case where the network has learned
(to high confidence) two independent networks, but hasn't learned to favour one
over the other.

     O
    /|\
   /|||\
  O OOO O
   ....

Each subnetwork is responsible for one of the nodes in the second last layer.
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
plt.title("High STD: poor fit")
plt.show()

# A smaller standard deviation fits better, but gives a smaller confidence
# interval for the unknown region too.
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0.5,0.01)
    q = np.random.normal(0.5,0.01)
    plot_it(s, q)
plt.title("Low STD: poor variance")
plt.show()

# We can add more than two subnets.
def plot_them(vs):
    ss = np.linspace(0,1,len(vs))
    plt.plot(xs, sum([v * (r(xs-s)+r(-xs-s)+s) for s,v in zip(ss,vs)]))

# But this actually has the same problem, it's just smoother. Our only choice is
# still to average over the subnets, and because we can't ensure we get the same
# total weight each time we need to keep the standard deviation low (we can't
# just add more and more subnets, because the variance increases with number of
# subnets).
num_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(1./num_subnets,0.01,num_subnets)
    plot_them(s)
plt.title("Many subnets: same problem")
plt.show()

# With dropout, the variance actually decreases with more subnets, so we can get
# a very good fit of the data by using lots of subnets. Problem with this is
# that it still doesn't actually give us extra variance in the unknown region.
num_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    p = 0.99
    s = np.random.binomial(1, p, num_subnets) / (num_subnets * p)
    plot_them(s)
plt.title("Many subnets with dropout: same problem")
plt.show()

# However, a solution is to do the subnets in a different way.

# We learn one subnet that fits the data.
fitter = lambda x: r(x-1) + r(-x-1) + 1
plt.plot(xs,abs(xs))
plt.plot(xs,fitter(xs))
plt.title("Subnet that fits perfectly")
plt.show()

# And then we learn a bunch of subnets that are zero on the data.
junk = lambda s,x: s * (r(x + 1) + r(x - 1) - 2 * r(x))

def plot_smart(vs):
    ss = np.linspace(0,1,len(vs))
    plt.plot(xs, fitter(xs) + sum([v * junk(s,xs) for s,v in zip(ss,vs)]))

# This provides a big range of guesses. The actual range will be determined by
# the ratio between the randomness and regularisation weightings.
num_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0,0.5,num_subnets)
    plot_smart(s)
plt.title("Combination of perfect subnet and junk")
plt.show()

# And the dropout version. Here we use many good subnets to simulate high
# confidence.
num_fitter_subnets = 1000
num_junk_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    ss = np.linspace(-1,1,num_junk_subnets)
    subnets = ([lambda x: fitter(x) / num_fitter_subnets] * num_fitter_subnets
            + [lambda x,s=s: junk(s,x) / num_junk_subnets for s in ss])
    num_subnets = num_fitter_subnets + num_junk_subnets

    p = 0.5
    vs = np.random.binomial(1, p, num_subnets) / p
    plt.plot(xs, sum([v * subnet(xs) for v, subnet in zip(vs, subnets)]))
plt.title("Dropout version")
plt.show()

# But we can also have the spaz subnets non-zero on the data -- instead they can
# contribute just like the regular ones. The important bit is that we have a small
# number of major contributors to the high-variance region, and a large number of
# small contributors to the low-variance region.
num_fitter_subnets = 1000
num_junk_subnets = 20
plt.plot(xs,abs(xs))
for _ in range(100):
    ss = np.linspace(-1,1,num_junk_subnets)
    num_subnets = num_fitter_subnets + num_junk_subnets
    subnets = ([lambda x: fitter(x) / num_subnets] * num_fitter_subnets
            + [lambda x,s=s: junk(s,x) / num_junk_subnets + fitter(x) / num_subnets
                    for s in ss])

    p = 0.5
    vs = np.random.binomial(1, p, num_subnets) / p
    plt.plot(xs, sum([v * subnet(xs) for v, subnet in zip(vs, subnets)]))
plt.title("Dropout version with semi-spaz subnets")
plt.show()
