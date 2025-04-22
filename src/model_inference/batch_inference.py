import openai
import asyncio
import aiohttp
import time
from typing import List, Dict

# Initialize your OpenAI API key here
#openai.api_key = "your_openai_api_key"
def batch_inference(prompts: List[str], temperature: int=0, model: str="gpt-4o", batch_size: str=20) -> List[str]:
    if len(prompts)<batch_size:
        return asyncio.run(async_batch_inference(prompts,model,temperature=temperature))
    else:
        results=[]
        for i in range(0,len(prompts),batch_size):
            batch_prompts=prompts[i:i+batch_size]
            results.extend(asyncio.run(async_batch_inference(batch_prompts,model,temperature=temperature)))
            time.sleep(3)
        return results

async def async_batch_inference(prompts: List[str], model: str="gpt-4o", max_tokens: int=4000, temperature: int=0):
    async def fetch_response(session, prompt):
        try:
            # Send request asynchronously
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    print(f"Request failed with status: {response.status}")
                    return None
        except Exception as e:
            print(f"Error for prompt: {prompt}. Error: {e}")
            return None

    # Main inference logic
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_response(session, prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
    
    return responses
