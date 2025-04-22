# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess Hellaswag dataset.

"""

import re
import os
import datasets
import pdb
from verl.utils.hdfs_io import copy, makedirs
import argparse


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/hellaswag')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'ast'
    t_1_dataset=datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/gorilla/berkeley-function-call-leaderboard/data/bfcl_live_simple') #mul_sim_par_pm')
    t_2_dataset = datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/gorilla/berkeley-function-call-leaderboard/data/bfcl_live_simple') #mul_sim_par_pm') #lucasmccabe/logiqa', trust_remote_code=True)
    train_data_num=len(t_1_dataset['train'])
    train_dataset = t_1_dataset['train'] #.select(range(int(train_data_num*0.8)))
    val_dataset = t_2_dataset['test'] #.select(range(int(train_data_num*0.8),train_data_num))
    test_dataset = t_2_dataset['test']
    instruction="""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

Your output consists of <think>…</think> and <code>…</code>. You put your thinking process in the format of  <think>…</think>. Then if you decide to invoke any of the function(s), you MUST put it in the format of <code>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]</code>,
You should avoid these errors: using wrong functions, lack necessary functions, missing requirement parameters, using unexpected parameters, parameter type unmatched, set wrong parameter value.

Here are some examples:
Question 1:Find me the best Italian restaurants in New York City with average customer ratings of more than 4 and accepts credit cards.
Here is a list of functions in JSON format that you can invoke.
"function": [{"name": "restaurant_search", "description": "Locates top rated restaurants based on specific criteria such as type of cuisine, ratings, and facilities.", "parameters": {"type": "dict", "properties": {"location": {"type": "string", "description": "The city and state, e.g. New York City, NY"}, "cuisine": {"type": "string", "description": "Preferred type of cuisine e.g., Italian, Indian, American, etc."}, "rating": {"type": "integer", "description": "Minimum average customer rating out of 5"}, "accepts_credit_cards": {"type": "boolean", "description": "If the restaurant should accept credit cards."}}, "required": ["location", "cuisine", "rating", "accepts_credit_cards"]}}]

Model response 1: <think> the thinking process""" 


    instruction+=""" To find the best Italian restaurants in New York City with an average customer rating of more than 4 and that accept credit cards, we need to use the `restaurant_search` function. This function is designed to locate top-rated restaurants based on specific criteria, which aligns perfectly with the requirements of the question.
The function requires the following parameters:
* location: The city and state (e.g., "New York City, NY")
* cuisine: The type of cuisine (e.g., "Italian")
* rating: The minimum average customer rating (e.g., 4)
* accepts_credit_cards: Whether the restaurant accepts credit cards (e.g., true)
Given the question, we have all the necessary parameters:
* Location: "New York City, NY"
* Cuisine: "Italian"
* Rating: 4 (since the question specifies "more than 4", we use 4 as the minimum)
* Accepts Credit Cards: true
We will construct the function call with these parameters to retrieve the desired list of restaurants. """ 

    instruction+="""</think>
<code> [{"restaurant_search": {"location": "New York City", "cuisine": "Italian", "rating": 4, "accepts_credit_cards": True}}]</code>

Current Question: """
    #instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct option here </answer>. '

    def make_map_fn(split):

        def process_fn(doc, idx):
            #ctx = #doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            import pdb
            query = instruction+str(eval(doc['input'])[0][0]['content']) #+' '+doc['query']+ ' ' #preprocess(doc["activity_label"] + ": " + ctx)
            #choices = [str(chr(id+ord('A')))+': '+option for id,option in enumerate(doc['options'])] #[preprocess(ending) for ending in doc["endings"]]
            query+=' ' +"Here is a list of functions in JSON format that you can invoke."+doc['function']+ ' '+"Model response:" #.join(choices)
            
            gold = doc['output'] #chr(int(doc['correct_option'])+ord('A'))
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": query,
                }],
                "ability": "nlp",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gold
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
#            pdb.set_trace()
            return data

        return process_fn
    # filter data that doesn't have a label
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('validation'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
 #   pdb.set_trace()
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
