import random
from flask import request, jsonify, Response
from App import app
from App.cgap.api_cgap import init_all, cgap
import os
import logging
import sys
import argparse
from App.cgap.api_utils import get_tasks_args

parser = argparse.ArgumentParser(description='process LIGHT data')
args = get_tasks_args(parser)
args.random_seed = random.randrange(1000000)

args.nq_prompt_file='/raid/dansu/datasets/open_domain_data/NQ/train.json'
args.tqa_prompt_file='/raid/dansu/datasets/open_domain_data/TQA/train.json'
args.nq_encoded_ctx_file='/raid/dansu/datasets/open_domain_data/NQ/encoded_ctx_files_all_multisetdpr_queryctx.pickle'
args.tqa_encoded_ctx_file='/raid/dansu/datasets/open_domain_data/TQA/encoded_ctx_files_all_multisetdpr_queryctx.pickle'
args.megatron_api_url='http://10.14.74.235:5000/api'

retriever, megatron_tokenizer = init_all(args)
margin_number = 4
ctx_len = 64


logger = logging.getLogger(__name__)

def init_logger(filename=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger

logger = init_logger('run_api.log')

@app.route('/post_input', methods=['POST'])
def process_input():
    input_ = request.form.get('input')
    input = {"question": input_}
    result = cgap(input, margin_number, ctx_len,retriever, megatron_tokenizer, args)
    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/get_input', methods=['GET'])
def get_input():
    input_ = request.args.get('input')
    input = {"question": input_}
    result = cgap(input, margin_number, ctx_len,retriever, megatron_tokenizer, args)
    logger.info('-------------------------------------------------')
    logger.info(f'{str(input_)} \n {result}')
    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")

    return response

@app.route('/index', methods=['GET'])
def metrics():  # pragma: no cover
    content = get_file('front/QA1.html')
    return Response(content, mimetype="text/html")

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        return open(src).read()
    except IOError as exc:
        return str(exc)

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))
