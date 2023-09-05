import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_tokenizer, get_args
import glob
import json

# def get_zero_scroll_template(task, paragraph, question=None):
# 
#     instructions= {}
#     instructions["gov_report"] = "You are given a report by a government agency. Write a one-page summary of the report."
#     instructions["summ_screen_fd"] = "You are given a script of a TV episode. Summarize the episode in a paragraph."
#     instructions["qmsum"] = "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences."
#     instructions["squality"] = "" 
#     instructions["qasper"] = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".'
#     instructions["narrative_qa"] = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible."
#     instructions["quality"] = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D)."
#     instructions["musique"] = ""
#     instructions["space_digest"] = ""
#     instructions["book_sum_sort"] = ""
# 
#     instruction = instructions[task]
#     if task == "quality":
#         template="{}\n\nStory:\n{}\n\nQuestion and Possible Answers:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
#     elif task == "qasper":
#         template="{}\n\nArticle:\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
#     elif task == "narrative_qa":
#         template="{}\n\nStory:\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
#     elif task == "musique":
#         template="{}\n\n{}\n\nQuestion:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
#     elif task == "gov_report":
#         template="{}\n\nReport:\n{}\n\nSummary:\n".format(instruction, paragraph)
#     elif task == "qmsum":
#         template="{}\n\nTranscript:\n{}\n\nQuery:\n{}\n\nAnswer:\n".format(instruction, paragraph, question)
#     elif task == "summ_screen_fd":
#         template="{}\n\nEpisode Script:\n{}\n\nSummary:\n".format(instruction, paragraph)
#     else:
#         raise ValueError('invalid task name for zero_scroll') # , choose from "qasper", "narrative_qa", "musique", "quality"
# 
#     return template

def get_zero_scroll_template_flan(task, paragraph, question, max_len, tokenizer, output_len):

    instructions= {}
    instructions["gov_report"] = "You are given a report by a government agency. Write a one-page summary of the report."
    instructions["summ_screen_fd"] = "You are given a script of a TV episode. Summarize the episode in a paragraph."
    instructions["qmsum"] = "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences."
    instructions["squality"] = ""
    instructions["qasper"] = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".'
    instructions["narrative_qa"] = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible."
    instructions["quality"] = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D)."
    instructions["musique"] = ""
    instructions["space_digest"] = ""
    instructions["book_sum_sort"] = ""

    instruction = instructions[task]
    # paragraph = paragraph.replace("\n", " ")
    if task == "quality":
        template="{} Story: {}".format(instruction, paragraph)
    elif task == "qasper":
        template="{} Article: {}".format(instruction, paragraph)
    elif task == "narrative_qa":
        template="{} Story: {}".format(instruction, paragraph)
    elif task == "gov_report":
        template="{} Report: {}".format(instruction, paragraph)
    elif task == "qmsum":
        template="{} Transcript: {}".format(instruction, paragraph)
    elif task == "summ_screen_fd":
        template="{} Episode Script: {}".format(instruction, paragraph)
    else:
        raise ValueError('invalid task name for zero_scroll') # , choose from "qasper", "narrative_qa", "musique", "quality"
    
    if question is not None:
        template = tokenizer.detokenize(tokenizer.tokenize(template)[:max_len - len(tokenizer.tokenize(question)) - 1 - output_len])
        template = template + " " + question
    else:
        template = tokenizer.detokenize(tokenizer.tokenize(template)[:max_len - output_len])

    return template

def format_answer(answer):
    return " {}".format(answer)

def preprocess(data_file, inference_only=False, retrieved_neighbours=False):

    args = get_args()
    assert args.ft_neighbours > 0 
    if args.longform_answer:
        nq_examples = []
        with open(data_file, "r") as f:
            for fn in f:
                nq_examples.append(json.loads(fn))
    else:
        nq_examples = []
        for my_data_file in sorted(glob.glob(data_file)):
            with open(my_data_file, "r", encoding='utf-8') as f:
                nq_examples.extend(json.load(f))
    
    data = []
    for instance in nq_examples:
        question = instance["question"]
        if retrieved_neighbours:
            contexts = instance["ctxs"]
            neighbours = [ctx["text"] for ctx in contexts] 
        else:
            if "sub-paragraphs" in instance:
                neighbours = [instance["sub-paragraphs"]]
            else:
                raise ValueError("need to get sub-paragraphs key")

        if inference_only:
            data.append((question, None, neighbours))
        else:
            if "answers" in instance:
                answers = instance["answers"]
            elif "answer" in instance:
                if type(instance["answer"]) is str:
                    answers = [instance["answer"]]
                elif type(instance["answer"]) is list:
                    answers = instance["answer"]
                else:
                    answers = [str(instance["answer"])]
            else:
                raise ValueError("need to have answer or answers")
            if len(answers) < 1:
                continue
                # answers = ["This question cannot be answered based on the given information."]
            else:
                ## only take answer 0
                if type(answers[0]) is dict:
                    answers = [answers[0]["text"].strip()]
                elif type(answers[0]) is str:
                    answers = [answers[0]]
                else:
                    raise ValueError("unsupported type for answer(s)")

            for answer in answers:
                answer = format_answer(answer)
                data.append((question, answer, neighbours))
    
    return data


def format_query(task, neighbours, question, max_len, tokenizer, output_len):

    paragraph = "\n\n".join(neighbours)
    # input_text = get_zero_scroll_template(task, paragraph, question)
    input_text = get_zero_scroll_template_flan(task, paragraph, question, max_len, tokenizer, output_len)

    ### to do
    # chunk to fit into sequence length

    # return tokens to chunk based on max_len in the future
    input_tokens = tokenizer.tokenize(input_text)
    assert len(input_tokens) <= 16384
    
    return input_tokens
