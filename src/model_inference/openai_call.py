import openai
import tiktoken
import json
import os
import pdb
import requests
from openai import OpenAI
from typing import List, Dict
from vllm import LLM, SamplingParams

encoding = tiktoken.encoding_for_model("gpt-4")
openai.api_key ='xxx'
def truncate_text_with_token_count(text: str, max_tokens: int) -> str:
    num_tokens = len(encoding.encode(text))
    if num_tokens > max_tokens:
        encoded = encoding.encode(text)[:max_tokens]
        truncated_text = encoding.decode(encoded)
        return truncated_text
    return text
def query_azure_openai_chatgpt_chat(query: str, model: str="gpt-4o", temperature: float=0, n: int=1) -> str:
    truncated_input = truncate_text_with_token_count(query, 30000)

    client = OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": truncated_input}, ], temperature=temperature, max_tokens=4000,n=n)
    except:
        import time
        time.sleep(10)
        response = client.chat.completions.create(model=model, messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": truncated_input}, ], temperature=temperature, max_tokens=4000,n=n)
    responses = [choice.message.content for choice in response.choices]
    if n==1:
        return responses[0] 
    else:
        return responses
def query_openai_chatgpt(query: str,api_key: str, model: str = "gpt-4", temperature: float = 0, n: int = 1) -> str:
    truncated_input = truncate_text_with_token_count(query, 30000)
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": truncated_input}, ], temperature=temperature, max_tokens=4000,n=n)
        import time
        time.sleep(1)
    except:
        import time
        time.sleep(120)

    responses = [choice.message.content for choice in response.choices]
    if n==1:
        return responses[0] 
    else:
        return responses
def query_vllm(model_path: str, query: str, max_tokens: int = 4000, temperature: float = 0.0, n: int = 1) -> List[str]:
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        top_k=50,
        n=n
    )
    prompt = f"You are a helpful AI assistant.\n\nUser: {query}\n\nAssistant:"
    results = llm.generate([prompt], sampling_params)
    responses = [result.outputs[0].text.strip() for result in results]
    return responses[0] if n == 1 and responses else responses

# Function to query Claude models via Anthropic's API
def query_claude(api_key: str, query: str, model: str = "claude-2", temperature: float = 0, max_tokens: int = 4000) -> str:
    url = "https://api.anthropic.com/v1/complete"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": f"\n\nHuman: {query}\n\nAssistant:",
        "model": model,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "stop_sequences": ["\n\nHuman:"],
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["completion"].strip()
    else:
        raise ValueError(f"Error querying Claude: {response.status_code} - {response.text}")
def query_model(
    query: str,
    backend: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    n: int = 1,
    model_path: str = None,
    api_key: str = None
) -> str:
    if backend == "openai":
        return query_openai_chatgpt(query, model, temperature, n,api_key)
    elif backend == "vllm" and model_path:
        return query_vllm(model_path, query, temperature=temperature, n=n,max_tokens=1500)
    elif backend == "claude" and api_key:
        return query_claude(api_key, query, model, temperature)
    else:
        raise ValueError("Invalid backend specified or missing required parameters (e.g., model_path for vLLM, api_key for Claude).")
