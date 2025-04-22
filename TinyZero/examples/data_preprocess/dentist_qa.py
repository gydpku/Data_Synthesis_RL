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
import torch
from BM25_retriever import create_bm, search_relevant_texts_case #search_relevant_texts
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    data_source = 'dentist_qa'

    dataset = datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/dentist_qa_rag_2048') #dataset('hippocrates/MedNLI_test') #, trust_remote_code=True)
    train_dataset = dataset['train']
    val_dataset = datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')['test']
    test_dataset = datasets.load_from_disk('/dccstor/obsidian_llm/yiduo/summary/src/medical_qa')['test']

    instruction = 'Your output thinking process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> thinking process here </think> <answer> the correct answer here </answer>. '
    '''
    dentist_book=torch.load('dentist_book.pt')
    bm25=create_bm(dentist_book)
    vectorizer = TfidfVectorizer(stop_words='english')  # You can change 'english' to None for no stop words removal
    tfidf_matrix = vectorizer.fit_transform(dentist_book)
    def td_get(documents,query,K=10):
        global vectorizer,tfidf_matrix
        query_vector = vectorizer.transform([query])

    # Step 3: Compute cosine similarity between the query and the documents
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Step 4: Get top K documents
        top_k_indices = cosine_similarities.argsort()[-K:][::-1]  # Get indices of top K highest similarities

    # Step 5: Return top K documents and their similarity scores
        top_k_results = [documents[idx] for idx in top_k_indices]
    
        return top_k_results
    from vllm import LLM,SamplingParams
    # inference param
    sampling_params = SamplingParams(temperature=0,max_tokens=30, top_p=0.95)
    model_name='/dccstor/obsidian_llm/yiduo/summary/src/TinyZero/model/Qwen2.5-7B-Instruct' # load model
    llm = LLM(model_name,max_model_len=8192,gpu_memory_utilization=0.95) #device="cuda:1"
    '''
    def generate(prompt,llm,sampling_params):
        response = llm.generate(
        prompt,
        sampling_params,
    )
        return response[0].outputs[0].text
    def make_map_fn(split):

        def process_fn(doc, idx):
            '''
            global dentist_book,bm25,llm,sampling_params
            keywords_prompt='Summarize the keywords of this question:{0}. Only output the keywords'.format(doc['query'][doc['query'].find('.')+1:])
            keywords=generate(keywords_prompt,llm,sampling_params)
            re_book=td_get(dentist_book,keywords,K=5) #search_relevant_texts_case(dentist_book,bm25,keywords,top_k=10)
            '''
            #ctx = #doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            try:
                query = doc['input'] #[doc['query'].find('.')+1:]+ ' ' #preprocess(doc["activity_label"] + ": " + ctx)
            except:
                query = doc['query'][doc['query'].find('.')+1:]
#            pdb.set_trace() #choices = [str(chr(id+ord('A')))+': '+option for id,option in enumerate(doc['options'])] #[preprocess(ending) for ending in doc["endings"]]
            #query+=' '.join(choices)
            query+= ' '+instruction
            query=query.strip()
            try:
                gold = doc['answer']
            except:
                gold=doc['output']
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
