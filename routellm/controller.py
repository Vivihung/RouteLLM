from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS

GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
    "reasoning_model": {
        "reasoning_model": "o3-mini",
        "api_base": "https://api.example.com/v1",
        "api_key": "your-api-key",
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Explain quantum physics concepts"
            },
            {
                "role": "assistant", 
                "content": "gpt-4"
            }
        ],
        "system_prompt": "Select the best model between {strong} (advanced) and {weak} (basic) for this query.",
        "max_tokens": 30,
        "temperature": 0.0
    }
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            
            router_config = config.get(router, {})
            # Inject model pair for routers that need it
            if router == "reasoning_model":
                router_config["model_pair"] = self.model_pair
                
            self.routers[router] = ROUTER_CLS[router](**router_config)

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    # ... rest of existing Controller methods ...
