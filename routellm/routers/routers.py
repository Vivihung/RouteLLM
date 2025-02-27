import abc
import functools
import random
import re
import logging

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from litellm import completion

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


class ReasoningModelRouter(Router):
    def __init__(
        self,
        reasoning_model: str,
        api_base: str,
        api_key: str,
        few_shot_examples: list,
        model_pair: "ModelPair",
        system_prompt: str = "Choose between {strong} and {weak} for this query. Respond only with the model name.",
        max_tokens: int = 50,
        temperature: float = 0.0
    ):
        self.reasoning_model = reasoning_model
        self.api_base = api_base
        self.api_key = api_key
        self.few_shot_examples = few_shot_examples
        self.model_pair = model_pair
        self.system_prompt = system_prompt.format(
            strong=model_pair.strong,
            weak=model_pair.weak
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def _build_messages(self, prompt: str) -> list:
        return [{
            "role": "system",
            "content": self.system_prompt
        }] + self.few_shot_examples + [{
            "role": "user",
            "content": f"Query: {prompt}\nResponse:"
        }]

    def calculate_strong_win_rate(self, prompt: str) -> float:
        try:
            response = completion(
                model=self.reasoning_model,
                messages=self._build_messages(prompt),
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            logging.error(f"Reasoning model error: {str(e)}")
            return 0.5  # Fallback to neutral

    def _parse_response(self, text: str) -> float:
        text = text.lower()
        strong = self.model_pair.strong.lower()
        weak = self.model_pair.weak.lower()
        
        strong_match = re.search(rf"\b{re.escape(strong)}\b", text)
        weak_match = re.search(rf"\b{re.escape(weak)}\b", text)
        
        if strong_match and not weak_match:
            return 1.0
        elif weak_match and not strong_match:
            return 0.0
        return 0.5  # Neutral if ambiguous


@no_parallel
class CausalLLMRouter(Router):
    pass  # Placeholder to fix indentation error


@no_parallel
class BERTRouter(Router):
    pass  # Placeholder to fix indentation error


class SWRankingRouter(Router):
    pass  # Placeholder to maintain structure


@no_parallel
class MatrixFactorizationRouter(Router):
    pass  # Placeholder to maintain structure


@no_parallel
class RandomRouter(Router):
    pass  # Placeholder to maintain structure


ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
    "reasoning_model": ReasoningModelRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
