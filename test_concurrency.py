import asyncio
import json
import os
import time
import shortuuid
from openai import OpenAI


async def test(concurrency: int = 10, num_tests: int = 10, use_guidance: bool = False):
    print(f"###### concurrency={concurrency}, use_guidance={use_guidance} ######")
    guidance = {
        "seed": 1000,
        "guided_json": {
            "type": "object",
            "properties": {
            "steps": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                    "explanation": {
                    "type": "string"
                    },
                    "output": {
                    "type": "string"
                    }
                },
                "required": ["explanation", "output"],
                "additionalProperties": False
                }
            },
            "final_answer": {
                "type": "string"
            }
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False
        }
    }

    print(guidance if use_guidance else {})
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="empty",
    )

    folder_name = f"concurrency_{concurrency}_guidance_{use_guidance}_{shortuuid.random()}"
    os.makedirs(folder_name, exist_ok=True)

    response_times = []
    for i in range(num_tests):
        os.makedirs(f"{folder_name}/attempt_{i}", exist_ok=True)
        start = time.time()
        futures = []
        for _ in range(concurrency):
            response = client.chat.completions.create(
                model="neuralmagic/Llama-3.2-3B-Instruct-FP8",
                messages=[
                    {
                    "role": "system",
                    "content": """You are a helpful math tutor. Format your answers in JSON format, conform to the following json schema:
{
            "type": "object",
            "properties": {
            "steps": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                    "explanation": {
                    "type": "string"
                    },
                    "output": {
                    "type": "string"
                    }
                },
                "required": ["explanation", "output"],
                "additionalProperties": False
                }
            },
            "final_answer": {
                "type": "string"
            }
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False
}"""
                    },
                    {
                        "role": "user",
                        "content": "solve 8x + 31 = 2"
                    }
                ],
                n=1,
                extra_body = guidance if use_guidance else {"seed": 1000},
            )
            futures.append(response)
        responses = await asyncio.gather(*futures)
        response_time = time.time() - start
        response_times.append(response_time)
        print(f"Time taken: {response_time}s")
        
        for j, response in enumerate(responses):
            raw_text = response.choice[0].message.content
            with open(f"{folder_name}/attempt_{i}/raw_{j}.json", "w") as f:
                f.write(raw_text)

    report_dict = {'response_times': response_times, 'average_response_time': sum(response_times) / len(response_times)}
    with open(f"{folder_name}/report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

if __name__ == '__main__':
    asyncio.run(test(concurrency=30, num_tests=10, use_guidance=True))
