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

    data_source = 'mednli'
    data_num=1
    dataset = datasets.load_dataset('/dccstor/obsidian_llm/yiduo/summary/src/mednli') #, trust_remote_code=True)
    dataset_gen=datasets.load_from_disk(f'/dccstor/obsidian_llm/yiduo/summary/src/mednli_{data_num}_500')
    train_dataset = dataset['train'].select(range(data_num))
    train_data=[]
    
    for data in train_subset:
        train_data.append({'input':data['input'],'output':data['output'],'mode_rate':1.0})
    train_dataset = datasets.concatenate_datasets([Dataset.from_list(train_data), dataset_gen['train']])   
    val_dataset = dataset['test']
    test_dataset = dataset['test']

    instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct answer here </answer>. '

    def make_map_fn(split):

        def process_fn(doc, idx):
            #ctx = #doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            query = doc['input']+ ' ' #preprocess(doc["activity_label"] + ": " + ctx)
            #choices = [str(chr(id+ord('A')))+': '+option for id,option in enumerate(doc['options'])] #[preprocess(ending) for ending in doc["endings"]]
            #query+=' '.join(choices)
            query+= ' '+instruction
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
