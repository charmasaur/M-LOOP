#!/usr/bin/env python
import numpy as np
import sys

def _check_params(learner, params):
    if not len(params) == learner.num_params:
        raise ValueError("Expected %d params, got %d" % (learner.num_params, len(params)))

def get_learner(fn):
    import mloop.learners as mll
    ft = fn.split(".")[-1]
    net = mll.NeuralNetLearner(nn_training_filename = fn,
            nn_training_file_type = ft,
            update_hyperparameters = False)
    net.import_neural_net()
    return net

def get_cost(learner, params):
    _check_params(learner, params)
    results = []
    for i in range(learner.num_nets):
        results.append(learner.predict_cost(params, i))
    return np.array(results)

def get_gradient(learner, params):
    _check_params(learner, params)
    results = []
    for i in range(learner.num_nets):
        results.append(learner.predict_cost_gradient(params, i))
    return np.array(results)

def get_curvature(learner, params):
    _check_params(learner, params)
    results = []
    for i in range(learner.num_nets):
        results.append(learner.get_curvature(params, i))
    return np.array(results)

def get_nearest(learner, params):
    _check_params(learner, params)
    results = []
    for i in range(learner.num_nets):
        results.append(learner.find_nearest_minimum(params))
    return np.array(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the neural network at a particular set of parameters. Must be run from the same directory as M-LOOP was run.')
    parser.add_argument("learner_filename")
    parser.add_argument("-c","--cost",action="store_true",help="get the cost")
    parser.add_argument("-g","--gradient",action="store_true",help="get the gradient")
    parser.add_argument("-u","--curvature",action="store_true",help="get the curvature (second derivatives)")
    parser.add_argument("-n","--nearest",action="store_true",help="get the nearest (predicted) minimum")
    #parser.add_argument("-c","--curvature",nargs='?',help="get the curvature (second derivatives) for a particular param set")
    parser.add_argument("-f","--file",type=argparse.FileType('r'),default=sys.stdin,help="File from which to take params (default stdin, type your params then Ctrl+D/EOF)")
    args = parser.parse_args()

    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-c", "--costs", help="print all costs", action="store_true")
    #group.add_argument("-i", "--index", help="get particular params", type=int)

    if args.cost or args.gradient or args.curvature or args.nearest:
        rs = []
        for line in args.file:
            for s in line.split(" "):
                rs.append(float(s))
        learner = get_learner(args.learner_filename)
        if args.cost:
            print("Cost")
            print(get_cost(learner, rs))
        if args.gradient:
            print("Gradient")
            print(get_gradient(learner, rs))
        if args.curvature:
            print("Curvature")
            print(get_curvature(learner, rs))
        if args.nearest:
            print("Nearest")
            print(get_nearest(learner, rs))
