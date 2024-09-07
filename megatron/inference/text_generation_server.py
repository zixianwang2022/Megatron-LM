# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron.training import get_args
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process
import os 

GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_generate():
        choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.tensor([BEAM_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
    
    def put(self):
        args = get_args()
       
        args.global_counter_cnt += 1 
        print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/megatron/inference/text_generation_server.py FUNC=MegatronGenerate.put line 35')
       
        if not "prompts" in request.get_json():
            return "prompts argument required", 400
        
        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        
        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400
        
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400
        
        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        
        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"
        
        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        
        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        
        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        
        top_p_decay = 0.0
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        
        top_p_bound = 0.0
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (type(top_p_bound) == float):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"
        
        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"
        
        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"
        
        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token=50256
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"
        
        length_penalty = 1 
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"
        
        
        # Updating flags to retrieve/insert states based on API request
        if "retrieving_mamba_states" in request.get_json():
            api_retrieving_mamba_states = request.get_json()["retrieving_mamba_states"]
            if not isinstance(api_retrieving_mamba_states, bool):
                return "retrieving_mamba_states must be a boolean"
            args.retrieving_mamba_states = api_retrieving_mamba_states
            print (f"----args.retrieving_mamba_states={args.retrieving_mamba_states}")
        else:
            args.retrieving_mamba_states = False 
            
        if args.retrieving_mamba_states: 
            if "retrieved_mamba_states_path" in request.get_json():
                api_retrieved_mamba_states_path = request.get_json()["retrieved_mamba_states_path"]
                if not isinstance(api_retrieved_mamba_states_path, str):
                    return "retrieved_mamba_states_path must be a string path"
                if not ".pkl" in api_retrieved_mamba_states_path:
                    return "retrieved_mamba_states_path must be a pickle file path"
                args.retrieved_mamba_states_path = api_retrieved_mamba_states_path
                print (f"----args.retrieved_mamba_states_path={args.retrieved_mamba_states_path}")
        else: 
            args.retrieved_mamba_states_path = None 
            
        
        if "inserting_mamba_states" in request.get_json():
            api_inserting_mamba_states = request.get_json()["inserting_mamba_states"]
            if not isinstance(api_inserting_mamba_states, bool):
                return "inserting_mamba_states must be a boolean"
            args.inserting_mamba_states = api_inserting_mamba_states
            print (f"----args.inserting_mamba_states={args.inserting_mamba_states}")
        else:
            args.inserting_mamba_states = False 
        
        if args.inserting_mamba_states: 
            if "inserted_mamba_states_path" in request.get_json():
                api_inserted_mamba_states_path = request.get_json()["inserted_mamba_states_path"]
                if not isinstance(api_inserted_mamba_states_path, str):
                    return "inserted_mamba_states_path must be a string path"
                if not ".pkl" in api_inserted_mamba_states_path:
                    return "inserted_mamba_states_path must be a pickle file path"
                args.inserted_mamba_states_path = api_inserted_mamba_states_path
                print (f"----args.inserted_mamba_states_path={args.inserted_mamba_states_path}")
        else:
            args.inserted_mamba_states_path = None 
        
        with lock:  # Need to get lock to keep multiple threads from hitting code
            
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                print(json.dumps(request.get_json()),flush=True)
                print("start time: ", datetime.datetime.now())
            
            try:
                if beam_width is not None:
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size = beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon
                        )
                    
                    return jsonify({"text": response,
                        "segments": response_seg,
                        "scores": response_scores})
                else:
                    print (f"Printing from Megatron-LM/megatron/inference/text_generation_server.py line 206")
                    
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    response, response_seg, response_logprobs, _, all_layers_states_dict = \
                        generate_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        return_output_log_probs=logprobs,
                        top_k_sampling=top_k,
                        top_p_sampling=top_p,
                        top_p_decay=top_p_decay,
                        top_p_bound=top_p_bound,
                        temperature=temperature,
                        add_BOS=add_BOS,
                        use_eod_token_for_early_termination=True,
                        stop_on_double_eol=stop_on_double_eol,
                        stop_on_eol=stop_on_eol,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        random_seed=random_seed)
                        
                    returned_states=""
                    
                    # Storing returned states 
                    if (args.retrieving_mamba_states):
                        try:
                            # Try to save the data
                            os.makedirs(os.path.dirname(args.retrieved_mamba_states_path), exist_ok=True)
                            torch.save(all_layers_states_dict, args.retrieved_mamba_states_path)
                            print("Retrieved states saved successfully.")
                            returned_states = args.retrieved_mamba_states_path
                        except Exception as e:
                            # Handle other exceptions
                            print (f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            print(f"An error occurred when storing inserted states: {e}")
                            print (f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            returned_states = f"Failed: {e}"
                    
                    print (f"Printing from Megatron-LM/megatron/inference/text_generation_server.py line 227")
                    print (f"--args.inserting_mamba_states: {args.inserting_mamba_states}; \n\
                             --args.retrieving_mamba_states: {args.retrieving_mamba_states}; \n\
                             --args.retrieved_mamba_states_path:{args.retrieved_mamba_states_path}; \n\
                             --args.inserted_mamba_states_path:{args.inserted_mamba_states_path}")

                    args.global_counter_cnt += 1 
                    print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/megatron/inference/text_generation_server.py FUNC=MegatronGenerate.put line 243')
       
                    
                    return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs, 
                        "returned_states": returned_states,})

            except ValueError as ve:
                return ve.args[0]
            print("end time: ", datetime.datetime.now())
        

class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model])
        
    def run(self, url, port): 
        print (f"printing from Megatron-LM/megatron/inference/text_generation_server.py line 246")
        print (f"--Trying to run Megatron Server")
        args = get_args()
        args.global_counter_cnt += 1 
        print (f'GLOBAL_CNT={args.global_counter_cnt} at Megatron-LM/megatron/inference/text_generation_server.py FUNC=MegatronServer.run line 259')
        
        # server.run("0.0.0.0",port=5000)
        self.app.run(url, threaded=True, debug=False, port=port)
        print (f"--After running Megatron Server")
