#!/usr/bin/env python
import sys

def _get_neural_net_learner(fn):
    import mloop.learners as mll
    ft = fn.split(".")[-1]
    net = mll.NeuralNetLearner(nn_training_filename = fn,
            nn_training_file_type = ft,
            update_hyperparameters = False)
    net.import_neural_net()
    return net

def get_curvature(fn, params):
    learner = _get_neural_net_learner(fn)
    if not len(params) == learner.num_params:
        raise ValueError("Expected %d params, got %d" % (learner.num_params, len(params)))
    return learner.get_curvature(params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the neural network at a particular set of parameters. Must be run from the same directory as M-LOOP was run.')
    parser.add_argument("learner_filename")
    parser.add_argument("-c","--curvature",action="store_true",help="get the curvature (second derivatives)")
    #parser.add_argument("-c","--curvature",nargs='?',help="get the curvature (second derivatives) for a particular param set")
    parser.add_argument("-f","--file",type=argparse.FileType('r'),default=sys.stdin,help="File from which to take params (default stdin, type your params then Ctrl+D/EOF)")
    args = parser.parse_args()

    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-c", "--costs", help="print all costs", action="store_true")
    #group.add_argument("-i", "--index", help="get particular params", type=int)

    if args.curvature:
        rs = []
        for line in args.file:
            for s in line.split(" "):
                rs.append(float(s))
        for p in get_curvature(args.learner_filename, rs):
            print(p)
