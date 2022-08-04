# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# from collections import defaultdict
# import h5py
# import numpy as np
import os

# from lutil import pax, print_rank, print_seq

from .timer import Timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError as e:
        pass
    return path

def make_sub_dir(top_path, sub_name):
    sub_path = os.path.join(top_path, sub_name)
    mkdir(sub_path)
    return sub_path

# eof
