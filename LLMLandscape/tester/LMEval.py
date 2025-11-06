import torch
import os
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from typing import Tuple, Dict, Iterable
from models import BaseModel


__all__ = ["lm_eval_gsm8k", "lm_eval_mmlu", "lm_eval_truthfulqa", "lm_eval_humaneval"]


def lm_eval_gsm8k(
    model: BaseModel,
    task: str = "gsm8k_cot_llama",
    limit: int = 100,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=True,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["exact_match,flexible-extract"].item()
    print("GSM8k Score: ", score)
    print("-" * 10)
    return score


def lm_eval_mmlu(
    model: BaseModel,
    task: str = "mmlu_generative",
    limit: int = 10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    system_instruction: str = None,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["mmlu", "mmlu_continuation", "mmlu_generation", "mmlu_generative"]
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=10,
        apply_chat_template=True,
        gen_kwargs=dict(do_sample=False),
        system_instruction=system_instruction,
    )
    results = results["results"]
    score = sum(
        [
            result["exact_match,get_response"] if task == "mmlu_generative" else result["acc,none"]
            for task_name, result in results.items()
        ]
    )
    score /= len(results)
    print("MMLU Score: ", score)
    print("-" * 10)
    return score


def lm_eval_truthfulqa(
    model: BaseModel,
    task: str = "truthfulqa_gen",
    limit: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md
    """
    assert task in ["truthfulqa_mc1", "truthfulqa_mc2", "truthfulqa_gen"]
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        max_batch_size=128,
        apply_chat_template=True,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["bleu_acc,none"] if "gen" in task else results["results"][task]["acc,none"]
    print("TruthfulQA Score: ", score)
    print("-" * 10)
    return score


def lm_eval_humaneval(
    model: BaseModel,
    task: str = "humaneval_instruct",
    limit: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    system_instruction: str = None,
    **kwargs
) -> float:
    """
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/README.md
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # warning: Please read the warning!!!
    assert task in ["humaneval", "humaneval_64", "humaneval_instruct", "humaneval_instruct_64"]
    hf_model = HFLM(pretrained=model, tokenizer=model.tokenizer, device=device)
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=[task],
        limit=limit,
        batch_size=1,
        apply_chat_template=True,
        confirm_run_unsafe_code=True,  # warning: Please read the warning!!!
        system_instruction=system_instruction,
        gen_kwargs=kwargs,
    )
    score = results["results"][task]["pass@1,create_test"].item()
    print("Humaneval Score: ", score)
    print("-" * 10)
    return score
