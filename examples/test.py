import os
import json
import logging

from routellm.controller import Controller
from typing import Dict, List, Union

def load_json_dataset(dataset_path: str) -> Union[List[Dict], Dict]:
    """
    Loads a JSON or JSONL dataset from the given file path.

    Args:
        dataset_path (str): The path to the JSON or JSONL dataset file.

    Returns:
        Union[List[Dict], Dict]: The loaded dataset. If the file is in JSONL format, a list of dictionaries is returned.
        If the file is in JSON format, a single dictionary is returned.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        JSONDecodeError: If the dataset file is not a valid JSON or JSONL file.

    """
    assert dataset_path, "data_path is not specified"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file {dataset_path} does not exist.")

    with open(dataset_path) as json_file:
        if dataset_path.lower().endswith("jsonl"):
            return [json.loads(f) for f in list(json_file)]

        return json.load(json_file)

# main ...

client = Controller(
  routers=["mf"],
  strong_model="azure/gpt-4o",
  weak_model="azure_ai/Phi-3-mini-128k-instruct-0624",
)

# load dataset
# hacky path. not reliable
dataset = load_json_dataset("../hybrid-ml/data/mix_instruct/val_mix_instruct.jsonl")

routing_predictions = []
for item in dataset:
    query = item["instruction"] or item["input"]
    response = client.chat.completions.create(
        model="router-mf-0.11593",
        messages=[
            {"role": "user", "content": query}
        ]
    )

    routing_predictions.append(response)

    print(f"{query} -> {response}")

count_dic = {}
for candidate in routing_predictions:
    if candidate not in count_dic:
        count_dic[candidate] = routing_predictions.count(candidate)
print(f"Summary: {count_dic}")

# response = client.chat.completions.create(
#   # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
#   model="router-mf-0.11593",
#   messages=[
#     {"role": "user", "content": "My internet service is very expensive, is there some way to drastically cut the cost of connecting to the internet?"}
#   ]
# )

# print(response)