from merge import load_data, dump_data
import json
import os

def load_neighbours(neighbours_file):

    examples = []
    with open(neighbours_file, "r") as f:
        for fn in f:
            examples.append(json.loads(fn))

    return examples

def rewrite_ctx(source_rows, retriever_rows):

    assert len(source_rows) == len(retriever_rows)
    new_rows = []
    for row_s, row_n in zip(source_rows, retriever_rows):
        assert len(row_n) == 2
        assert row_s["question"] == row_n[0]
        row_s["ctxs"] = row_n[1]
        new_rows.append(row_s)
    return new_rows

def main(source_file, neighbours_file, target_file, shuffle=True):

    source_rows = load_data(source_file)
    rows = load_neighbours(neighbours_file)
    new_rows = rewrite_ctx(source_rows, rows)
    dump_data(new_rows, target_file, shuffle)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="combine data")
    parser.add_argument('--split', type=str, default=None,
                       help='split')
    parser.add_argument('--brand', type=str, default=None,
                       help='brand')
    args = parser.parse_args()
    brand = args.brand
    split = args.split
    retriever_name = "tasb"

    neighbours_file = "/mnt/fsx-main/pengx/projects/inform-re2/inform-retriever/data/{}/{}_retriever_top10_{}.json".format(brand, retriever_name, split).replace("dev", "valid")
    rows = load_neighbours(neighbours_file)
    # print(dpr_rows[0])
    source_file = "/mnt/fsx-main/pengx/projects/inform-re2/inform-retriever/data/{}/{}.json".format(brand, split).replace("dev", "valid")
    target_dir = "/mnt/fsx-outputs-chipdesign/pengx/projects/retro/data/{}_{}_retrieved/".format(brand, retriever_name)
    target_file = target_dir + "{}.json".format(split)
    # target_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/benz_dpr_finetuned/test.json"
    
    path = target_dir
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
    
    main(source_file, neighbours_file, target_file)
    # print(source_file, dpr_neighbours_file, target_file)

