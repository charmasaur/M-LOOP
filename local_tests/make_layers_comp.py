import glob
import numpy as np
import matplotlib.pyplot as plt

fns = glob.glob("*layers64each.dat")
names = []
dems = []
for fn in fns:
    with open(fn) as f:
        names.append(fn.strip(".dat"))
        dems.append([float(l.strip()) for l in f])

nums = [int(n[4:].split("l")[0]) for n in names]
dems = np.array(dems)
nums = np.array(nums)

for n,d,name in zip(nums, dems, names):
    plt.scatter([n]*len(d), d, label=name)
plt.legend()

plt.plot(nums,np.percentile(dems,50,axis=1))
plt.plot(nums,np.percentile(dems,90,axis=1))
plt.plot(nums,np.percentile(dems,10,axis=1))

plt.show()
