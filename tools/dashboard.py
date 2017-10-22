import tools.net_inspector as ni
import mloop.visualizations as mlv
import numpy as np
import numpy.linalg as nl
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sp
import matplotlib.patches as patches
import tools.dragplot as dp
from matplotlib.widgets import Slider

date = '2017-08-14_14-53'
curvature_filename = './curvature_at_best.npy'
num_params = 63
num_nets = 3
titles = ["Trapping", "Repump", "Magnetic field"]
human_ratio = 3 # OD improvement over human run
PCA_index = 0 # which net to use for PCA components

ratio = 0.5 # use this ratio of params for PCA
num_types = len(titles)
if not num_params % num_types == 0:
    raise ValueError
num_params_per_type = int(num_params / num_types)

learner = ni.get_learner('./M-LOOP_archives/learner_archive_' + date + '.txt')
cvis = mlv.ControllerVisualizer('./M-LOOP_archives/controller_archive_' + date + '.txt',file_type='txt')
costs = np.array(cvis.in_costs)
raw_parameters = np.array(cvis.out_params)
if not len(raw_parameters[0]) == num_params:
    raise ValueError

costs_and_params = [(costs[i],raw_parameters[i]) for i in range(len(costs))]
costs_and_params.sort()
# (n_components, num_params)
pca_components = None
pca_components_weights = None

try:
    curvatures = np.load(curvature_filename)
    expected_shape = (num_nets, num_params, num_params)
    if not curvatures.shape == expected_shape:
        raise ValueError("Wrong shape for curvatures. Got " + str(curvatures.shape) + ", expected " + str(expected_shape))
    eigenresult = nl.eig(0.5*(curvatures[PCA_index] + np.transpose(curvatures[PCA_index])))
    pca_components_weights = eigenresult[0]

    pca_components = np.transpose(eigenresult[1])
except Exception as e:
    print("Couldn't find curvature_at_best: " + str(e))
    pass

if pca_components is None:
    # Fall back to PCA
    srp = np.array([d[1] for d in costs_and_params]) # ("srp" means "sorted raw params")
    srpt = srp[:int(ratio*len(srp))] # ("srpt" is "sorted raw params truncated")
    ssp = sp.scale(srpt, with_mean=True, with_std=False) # ("sorted scaled params")
    pca = sd.PCA()
    pca.fit(ssp)
    pca_components = pca.components_
    pca_components_weights = pca.explained_variance_ratio_

fig,ax = plt.subplots(max(num_types,3),2)
#fig.subplots_adjust(bottom=0.25)
#pcan = 0
#nax = 2 + pcan
#usedax = 0
#slids = []
#for i in range(pcan):
#    slid_ax = fig.add_axes([0.1,usedax*0.25/nax,0.8,0.12/nax])
#    slids.append(Slider(slid_ax, 'V' + str(i), -10, 10, 0))
#    usedax += 1
#fig.add_axes([0.1,usedax*0.25/nax,0.8,0.12/nax])
#usedax += 1
#pca_axes = fig.add_axes([0.1,usedax*0.25/nax,0.8,0.12/nax])
#usedax += 1
cost_ax = ax[0,1]
pca_weight_ax = ax[1,1]
base_ax = ax[2,1]
type_ax = ax[:,0]
pca_ax = pca_weight_ax.twinx()

def reset(base):
    global base_params,pca_params,mutable_params,pca_factors
    base_params = costs_and_params[base][1]
    pca_factors = np.zeros(len(pca_components))
    pca_params = np.zeros(num_params)
    mutable_params = np.zeros(num_params)

def mutable_listener(j, i, y):
    global base_params, pca_params, mutable_params
    mutable_params[j+i] = y - (base_params[j+i] + pca_params[j+i])
    update()

def base_listener(v):
    global base_params,pca_params,mutable_params
    reset(int(round(base_slid.val)))
    update()

def pca_listener(i, y):
    global pca_factors, pca_params
    pca_factors[i] = y
    pca_params = np.dot(np.transpose(pca_components), pca_factors)
    update()

def update():
    global base_params, pca_params, mutable_params
    params = base_params + pca_params + mutable_params
    minc = mincost
    maxc = maxcost
    cost = ni.get_cost(learner, params)
    for i in range(num_types):
        ts[i].update(params[num_params_per_type*i:num_params_per_type*(i+1)])
    for i in range(num_nets):
        costs_history[i].append(cost[i])
        cost_plots[i].set_xdata(np.arange(len(costs_history[i])))
        cost_plots[i].set_ydata(np.array(costs_history[i]))
        minc = min(minc, min(costs_history[i]))
        maxc = max(maxc, max(costs_history[i]))
    cost_ax.set_xlim([max(0,len(costs_history[0])-100),len(costs_history[0])])
    cost_ax.set_ylim([minc,maxc])
    pca_plot.update(pca_factors)
    fig.canvas.draw()

ts = []
costs_history = []
cost_plots = []
mincost = min([cp[0] for cp in costs_and_params])
maxcost = max([cp[0] for cp in costs_and_params])
humancost = maxcost/np.power(maxcost/mincost, 1./human_ratio)

cost_ax.set_ylabel("Predicted costs")

reset(0)

for i in range(num_types):
    ts.append(dp.DragPlot(type_ax[i],base_params[num_params_per_type*i:num_params_per_type*(i+1)],lambda t,y,j=num_params_per_type*i: mutable_listener(j,t,y)))
    type_ax[i].set_ylabel(titles[i])
for i in range(num_nets):
    costs_history.append([ni.get_cost(learner, costs_and_params[0][1])[i]])
    cost_plots.append(cost_ax.plot(costs_history[i])[0])
    cost_ax.axhline(y=mincost,c='k')
    cost_ax.axhline(y=maxcost,c='k')
    cost_ax.axhline(y=humancost,c='k',ls='--')

pca_plot = dp.DragPlot(pca_ax, pca_factors, pca_listener)
pca_ax.set_ylim([-10,10])
pca_ax.set_xlabel("Principal component")
pca_ax.set_ylabel("Weight")
pca_weight_ax.plot(pca_components_weights)

base_slid = Slider(base_ax, '', 0, len(costs_and_params)-1, valinit=0, valfmt='%0.0f')
base_slid.on_changed(base_listener)
base_ax.set_xlabel("Base parameter")

plt.show()
