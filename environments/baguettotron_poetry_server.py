import random
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import re

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ServerBaseline,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

class BaguettotronPoetryEnv(BaseEnv):
    name = "baguettotron_poetry"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=False,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, ServerBaseline]:
        env_config = BaseEnvConfig(
            max_num_workers=1,
            max_eval_workers=1,
            max_num_workers_per_node=1,
            tokenizer_name="PleIAs/Baguettotron",
            group_size=1,
            use_wandb=True,
            rollout_server_url="http://127.0.0.1:8000",
            total_steps=1000,
            batch_size=1,
            steps_per_eval=1000,
            max_token_length=8192,
            wandb_name="baguettotron_poetry",
            ensure_scores_are_not_same=False,
        )
        server_config = APIServerConfig(
            server_type="vllm",
            model_name="PleIAs/Baguettotron",
            base_url="http://127.0.0.1:9004/v1",
            api_key="x",
            num_requests_for_eval=1,
            num_max_requests_at_once=1,
        )
        return env_config, server_config

    async def setup(self):
        self.train = iter(load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True).shuffle(buffer_size=10_000, seed=42))
        # override for baguettotron's separate template file
        self.tokenizer.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"

    async def collect_trajectories(
        self, item: str
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item}
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[user_message],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=0.9,
                repetition_penalty=1.05,
                min_p=0.1,
            )

        to_score = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "finish_reason": chat_completion.finish_reason,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, []

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["messages"] = []

        for item in rollout_group_data:
            messages = item["messages"]
            finish_reason = item["finish_reason"]
            # look for ⟨H≈ block
            print(messages[-1]["content"])
            reward = 1 if len(re.findall(r"⟨H≈", messages[-1]["content"])) > 0 else 0
            # score = 0 if not natural stop
            reward = 0 if finish_reason != "stop" else reward
            out_dict = tokenize_for_trainer(self.tokenizer, messages)
            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(reward)
            scores["messages"].append(messages)

        return scores

    async def get_next_item(self) -> str:
        next_item = next(self.train)
        return next_item["text"]

    async def evaluate(
        self,
    ):
        print(f"[{self.name}] Evaluate method called (placeholder).")
        # Implement evaluation logic if you have a separate test set and metrics
        pass


if __name__ == "__main__":
    BaguettotronPoetryEnv.cli()
