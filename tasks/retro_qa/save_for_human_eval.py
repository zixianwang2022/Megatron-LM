import json

def load_groundtruth_file(data_file):

    with open(data_file, "r") as f:
        nq_examples = json.load(f)

    gcs, qs, sps = [], [], []
    for instance in nq_examples:
        gcs.append(instance["answers"][0])
        qs.append(instance["question"])
        sps.append(instance["sub-paragraphs"])

    return gcs, qs, sps

def load_prediction(data_file):

    data = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data.append(line.strip())

    return data

def save_to_txt(output_file, qs, answers, gcs, sps):

    with open(output_file, "w") as f:
        for q, a, g, sp in zip(qs, answers, gcs, sps):
            f.write(q + "\n")
            f.write("Model: " + a + "\n")
            f.write("GroundTruth: " + g + "\n")
            f.write("sub-paragraphs: "+ repr(sp) + "\n")
            f.write("\n")

def test():

    ground_truth_file = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/landrover/test.json"
    gcs, qs, sps = load_groundtruth_file(ground_truth_file)
    
    model_output1 = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_ft_same_format_ctx1_8.3b_8_1e-6_0.0/generate_8.3b_test_greedy_0_250_292.txt"
    model_output2 = "/lustre/fsw/adlr/adlr-nlp/pengx/retro/checkpoints/applications/landrover_plus_benz_tasb_retrieved_retro_ft_same_format_ctx1_8.3b_8_1e-6_0.0_4_384/generate_8.3b_test_greedy_0_250_4_438.txt"
    model_output = model_output1
    answers = load_prediction(model_output)
    output_file = model_output.split("/")[-2] + ".eval.txt"
    print(output_file)
    
    assert len(qs) == len(answers)
    save_to_text(output_file, qs, answers, gcs, sps)

def main(ground_truth_file, model_output, output_file):

    gcs, qs, sps = load_groundtruth_file(ground_truth_file)
    answers = load_prediction(model_output)
    assert len(qs) == len(answers)
    save_to_txt(output_file, qs, answers, gcs, sps)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="create data for evaluation")
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()
    main(args.ground_truth, args.predict_file, args.output_file)
