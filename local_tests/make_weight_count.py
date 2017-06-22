import matplotlib.pyplot as plt
import numpy as np
import glob

def func(m,L):
    return 11*L + (m-1)*L*L

for m in range(2,10):
    xs = []
    ys = []
    for L in range(150):
        try:
            with open("3_3_" + str(m) + "layers" + str(L) + "each.dat", "r") as f:
                vals = [float(l.strip()) for l in f]
                xs.append(func(m,L))
                ys.append(np.median(vals))
                print("(%d, %d): %d %f" % (m,L,xs[-1],ys[-1]))
        except FileNotFoundError:
            pass
    plt.scatter(xs,ys,label=str(m))

plt.xlabel("Number of weights")
plt.ylabel("Score")
plt.legend()
plt.show()
