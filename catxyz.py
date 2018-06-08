#!/usr/bin/python3
# just concatenate some XYZ files
#
from ips_subs import *
#
molname = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
concat_xyz(molname, start, end)
