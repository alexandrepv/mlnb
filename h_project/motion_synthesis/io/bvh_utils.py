from bvh import Bvh
import bvhtoolbox

def load_bvh(bvh_fpath):
    with open(bvh_fpath) as f:
        mocap = Bvh(f.read())

    return mocap