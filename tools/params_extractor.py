#!/usr/bin/env python
import numpy as np
import mloop.utilities as mlu
import sys

def _get_controller_dict(fn):
    ft = fn.split(".")[-1]
    d = mlu.get_dict_from_file(fn, ft)
    
    if not 'archive_type' in d:
        raise ValueError("Unknown archive type")
    
    if not d['archive_type'] == 'controller':
        raise ValueError("Wrong archive type: " + d['archive_type'])

    return d

# Returns the index'th params
def get_params(fn, index):
    return np.array(_get_controller_dict(fn)['out_params'][index])

# Returns a list of all costs by default, or a list of (cost, learner type) if
# include_type is True.
def get_all_costs(fn, include_type=False):
    d = _get_controller_dict(fn)

    out_params = np.array(d['out_params'])
    out_type = [x.strip() for x in list(d['out_type'])]
    in_costs = np.squeeze(np.array(d['in_costs']))

    if not len(out_params) == len(out_type):
        raise ValueError("Num out params (%d) and num out types (%d) don't match" % (len(out_params), len(out_type)))

    if len(out_params) < len(in_costs):
        raise ValueError("More in costs (%d) than out params (%d)" % (len(in_costs), len(out_params)))

    if len(out_params) > len(in_costs):
        print("Warning, more out params (%d) than in costs (%d), so I'm dropping the extra params" % (len(out_params), len(in_costs)))

    ct = len(in_costs)
    return zip(in_costs[:ct], out_type[:ct]) if include_type else in_costs[:ct]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract parameters from the controller archive.')
    parser.add_argument("controller_filename")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--costs", help="print all costs", action="store_true")
    group.add_argument("-i", "--index", help="get particular params", type=int)
    args = parser.parse_args()
    if args.costs:
        costs = get_all_costs(args.controller_filename, include_type=True)
        for i,(c,t) in enumerate(costs):
            print("%d %f %s" % (i,c,t))
    elif not args.index is None:
        for p in get_params(args.controller_filename, args.index):
            print(p)
