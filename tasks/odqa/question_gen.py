
from readline import replace_history_item
from megatron.text_generation import generate_and_post_process, beam_search_and_post_process
from megatron import get_args
from megatron import mpu


def context_filtering(context):
    '''remove the incomplete sentences from the context'''
    filtered_context=context

    return filtered_context

def post_process_generations(generations, min_token_length=5, sep='\n'):
    # return the first string that has length longer than 5
    generations_split = generations.split(sep)
    for each in generations_split:
        if len(each.strip()) >= min_token_length:
            return each.strip()
    
    return "No proper answer!"


def construct_qg_prompt(context, prompt_list=None, num_prompt_examples=0):

    prompt =''
    if prompt_list is not None:
        for each in prompt_list[:num_prompt_examples]:
            prompt += 'Context: ' + each['ctxs']['title'] + ' ' + each['ctxs']['text'] + '\n' + 'Question: ' + each['question'] + '\n\n'

    # test zero-shot for question generation first
    qg_prompt = 'Context: ' + context['wikipedia_title'] + ' ' + context['text'] + '\n' + 'Question:'

    qg_prompt = prompt + qg_prompt

    len_qg_prompt = len(qg_prompt)

    return qg_prompt, len_qg_prompt
        

def generate_question(data, gen_model, prompt_list=None, num_prompt_examples=0):
    '''generate the question for the given context, and the prompt_list[:num_prompt_examples]'''
    args = get_args()

    topk_contexts = data['output'][0]['provenance']
    qg_prompt_list=[]
    len_qg_prompt_list=[]
    prompts_plus_generations_list=[]

    for each in topk_contexts:
        qg_prompt, len_qg_prompt = construct_qg_prompt(each, prompt_list, num_prompt_examples)

        qg_prompt_list.append(qg_prompt)
        len_qg_prompt_list.append(len_qg_prompt)

    qg_outputs = generate_and_post_process(
                model=gen_model, 
                prompts=qg_prompt_list, 
                tokens_to_generate=48,
                top_k_sampling=0,
                top_p_sampling=0.9,
                temperature = args.temperature,
                random_seed=args.random_seed
                )
    prompts_plus_generations_list = qg_outputs[0]

    generation_list = []
    if mpu.get_tensor_model_parallel_rank() == 0:
        if mpu.is_pipeline_first_stage():
            for prompts_plus_generations, len_qg_prompt in zip(prompts_plus_generations_list, len_qg_prompt_list):
                generations = prompts_plus_generations[len_qg_prompt:].strip()
                generations_str = post_process_generations(generations, min_token_length=5, sep='\n')
                generation_list.append(generations_str)
               
    question_context_pairs=[]
    scores = []
    for question_gen, context in zip(generation_list, topk_contexts):
        ctx={}
        ctx['title'] = context['wikipedia_title']
        ctx['text'] = context['text']
        question_context_pairs.append(
            {
            "ctxs": ctx,
            "question": question_gen.strip().replace('?','') + '?',
            }
        )
        scores.append(context['score'])
    return question_context_pairs, scores
    

    
        







