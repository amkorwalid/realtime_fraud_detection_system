from pathlib import Path
from dotenv import load_dotenv
import os
import requests
import time 
import httpx
import asyncio
import json
from generator import data_loader, generate_request_schedule


load_dotenv()

IP_INFERENCE_SERVER = os.getenv("IP_INFERENCE_SERVER", "localhost")
PORT_INFERENCE_SERVER = os.getenv("PORT_INFERENCE_SERVER", "8000")

# save the response to a json file in the data/responses directory
def save_response_to_file(result, filename="response.json"):
    filename = Path(__file__).parent.parent / "data" / "responses" / filename
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)

# async function to process a batch of requests
async def process_batch(idx_start, idx_end):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"http://{IP_INFERENCE_SERVER}:{PORT_INFERENCE_SERVER}/api/inference", json=data[idx_start:idx_end], timeout=5)
        response.raise_for_status()
    # save data, result, and response to a json file
    query = {
        "data": data[idx_start:idx_end],
        "result": result[idx_start:idx_end],
        "response": response.json()
    }
    save_response_to_file(query, filename=f"response_{idx_start}_{idx_end}.json")

data, result = data_loader()

TOTAL_REQUESTS = len(data)
DURATION = 3600

buckets_batchs = generate_request_schedule(total_requests=TOTAL_REQUESTS, duration=DURATION)

idx_start = 0
idx_end = 0

# dynamic two pointer approach to process the requests in batches based on the generated schedule
total_time_start = time.perf_counter()
for bucket_size in buckets_batchs:
    start_time = time.perf_counter()
    idx_start = idx_end
    idx_end = idx_start + bucket_size
    asyncio.run(process_batch(idx_start, idx_end))
    end_time = time.perf_counter()
    if (end_time - start_time ) > 1.0:
        print(f"Warning: Processing time exceeded 1 second for bucket starting at index {idx_start}. Time taken: {end_time - start_time:.2f}s")
    else:
        time.sleep(max(0, 1.0 - (end_time - start_time)))
    print(f"Timecost: {end_time - start_time:.2f}s, Bucket Size: {bucket_size}, Start Index: {idx_start}, End Index: {idx_end}")
    
print("All batches processed.")
total_time_end = time.perf_counter()
print(f"Total Timecost: {total_time_end - total_time_start:.2f}s")