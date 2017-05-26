'''
We consider a scenario where we're trying to fit data sampled from an abs()
function, For simplicity we assume that we didn't get any points in [-1,1].
We're trying to find the network with one hidden layer of two neurons that fits
these data and exhibits some randomness to reflect the uncertainty about what
happens in [-1,1].
'''

import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(-5,5,100)

# ReLU
r = np.vectorize(lambda x: x if x > 0. else 0.)

# Plot a flattened abs, with the flat bottom at s.
def plot_it(s):
    plt.plot(xs,r(xs-s)+r(-xs-s)+s)
    
# Sample the flat bottom from N(0.5,0.2). Note that the weighting given to the
# randomness will determine the right standard deviation here. If we favour
# randomness then we'll get a high-ish standard deviation, even if it means we
# sometimes miss data points. If we favour data will get a lower standard
# deviation.
plt.plot(xs,abs(xs))
for _ in range(5):
    s = np.random.normal(0.5,0.2)
    plot_it(s)
plt.show()

# Indeed, the above is actually sub-optimal regardless of our relative
# weighting, because we can get a higher standard deviation by centering at 0.
# This suggests that the "most likely" fit is absolute value, which is quite
# interesting.
plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0.,0.5)
    plot_it(s)
plt.show()

plt.plot(xs,abs(xs))
for _ in range(100):
    s = np.random.normal(0.,0.5)
    plot_it(s)
plt.show()

