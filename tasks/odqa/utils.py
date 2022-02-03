
import os
from pathlib import Path



def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    ans_dict = {}
    for path in files:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id, answer = line.split('\t')
                ans_dict[int(id)] = answer
        path.unlink()

    with open(output_path, 'w') as outfile:
        for id, ans in sorted(ans_dict.items()):
            outfile.write(ans)
            
    glob_path.rmdir()
    
    return True
