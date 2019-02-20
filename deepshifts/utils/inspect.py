from deepshifts.h5handler import H5Handler
import sys

with H5Handler(sys.argv[1]) as dl:
    groups = dl.count_groups(sys.argv[2])
    print('total number of conformations =', sum(groups.values()))
    print('total number of groups =', len(groups))
    for k,v in groups.items():
        print(k, '=', v)

