# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# import glob
import os
from tqdm import tqdm

from lutil import pax

PROJECTS_DIR = "/lustre/fsw/portfolios/adlr/users/lmcafee/retro/projects"
SRC_PROJECT_DIR = os.path.join(PROJECTS_DIR, "next-llm")
DST_PROJECT_DIR = os.path.join(PROJECTS_DIR, "next-llm-core")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def softlink_dirs(top_dir):
#     # print(top_dir)
#     for root, sub_keys, sub_files in os.walk(top_dir):
#         # for sub_key in sub_keys:
#         #     softlink_dirs(os.path.join(top_dir, sub_key))
#         # pax("root, dirs, files")
#         print("root = [%d, %d] ... '%s'." % (len(sub_keys), len(sub_files), root))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # >>>
    # # src_files = list(glob.glob(src_project_dir + "/*"))
    # # src_files = list(glob.glob(src_project_dir + "/db/**", recursive=True))
    # src_files = list(glob.glob(src_project_dir + "/index/**", recursive=True))
    # # src_files = list(glob.glob(src_project_dir + "/index/faiss_par_add/**", recursive=True))

    # n_src_files = len(src_files)
    # pax("src_project_dir, dst_project_dir, n_src_files")
    # +++
    # softlink_dirs(SRC_PROJECT_DIR)
    # +++
    dir_items = list(os.walk(SRC_PROJECT_DIR))
    for dir_idx, (src_dir, sub_keys, sub_files) in enumerate(dir_items):

        dst_dir = src_dir.replace(SRC_PROJECT_DIR, DST_PROJECT_DIR)
        sub_files = [ f for f in sub_files if not f.endswith("~") ]

        os.makedirs(dst_dir, exist_ok=True)
        
        print("dir %2d/%2d ... [%2d, %6d] ... '%s'." % (
            dir_idx,
            len(dir_items),
            len(sub_keys),
            len(sub_files),
            dst_dir,
        ))

        for sub_file in tqdm(sub_files, desc="symlink"):
            src_path = os.path.join(src_dir, sub_file)
            dst_path = os.path.join(dst_dir, sub_file)
            if not os.path.islink(dst_path):
                os.symlink(src_path, dst_path)
            # pax("src_path, dst_path")

        # if src_dir.replace(SRC_PROJECT_DIR, "") not in (
        #     "",
        #     "/db",
        #     "/db/individual",
        # ):
        #     pax("dst_dir")

    # <<<

# eof
