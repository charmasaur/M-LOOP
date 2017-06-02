import matplotlib.pyplot as plt
import glob

ext = ".dat"
fns = glob.glob("*" + ext)
names = []
bests = []
for fn in fns:
    with open(fn) as f:
        names.append(fn.strip(ext))
        bests.append([float(l.strip()) for l in f])

plt.hist(bests, label=names)

plt.legend()
plt.show()
