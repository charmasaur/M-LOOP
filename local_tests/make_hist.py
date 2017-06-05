import matplotlib.pyplot as plt
import numpy as np
import glob

ext = ".dat"
fns = glob.glob("*" + ext)
names = []
bests = []
for fn in fns:
    with open(fn) as f:
        names.append(fn.strip(ext))
        bests.append([float(l.strip()) for l in f])
        print(names[-1] + ": " + str(np.mean(bests[-1])) + " +/- " + str(np.std(bests[-1])))

plt.hist(bests, label=names)

plt.legend()
plt.show()
