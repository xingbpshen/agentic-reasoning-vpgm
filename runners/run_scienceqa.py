import argparse
import json
import os
import glob
import string

from tqdm import tqdm
from vllm import LLM, SamplingParams

from engine.lm import build_lm_input
from engine.func import get_lm_answer_prob


def get_next_run_idx(output_dir, reasoning_model, llm_name):
    pattern = os.path.join(output_dir, f"{reasoning_model}_{llm_name}_run-*.json")
    existing_files = glob.glob(pattern)

    run_indices = []
    for f in existing_files:
        base = os.path.basename(f)
        try:
            idx_str = base.split("run-")[1].split(".json")[0]
            run_indices.append(int(idx_str))
        except (IndexError, ValueError):
            continue

    return max(run_indices, default=-1) + 1  # start from 0 if none found


def info(msg):
    print(f"\033[1;94m[run_scienceqa.py]\033[0m \033[94mINFO\033[0m {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ScienceQA with vLLM.")

    # core setup
    parser.add_argument(
        "--split",
        type=str,
        default="test_2563",
        help="Dataset split name (e.g. val_1005, test_2563).",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="vpgm-n3",
        choices=["cot", "chameleon", "chameleon-plus", "vpgm-n2", "vpgm-n3", "vpgm-n4"],
        help="Reasoning strategy.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        help="Logical LLM name used in result filenames.",
    )

    # model + vLLM config
    parser.add_argument(
        "--model_path",
        type=str,
        default="/usr/local/data/Meta-Llama-3-8B-Instruct",
        help="Path or identifier for the base model.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio for vLLM.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1",
        help='Value for CUDA_VISIBLE_DEVICES (e.g. "0,1" or "2,3").',
    )

    # generation params
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="</EOR>",
        help="Stop sequence for generation.",
    )

    # data + output
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/my_scienceqa",
        help="Directory containing ScienceQA JSON splits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/my_scienceqa",
        help="Base directory to store results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples_override",
        type=int,
        default=None,
        help=(
            "Override number of samples per question. "
            "If not set, uses 3 for {chameleon-plus,vpgm-n2,vpgm-n3,vpgm-n4}, else 1."
        ),
    )

    # misc
    parser.add_argument(
        "--xdg_cache_home",
        type=str,
        default="/usr/local/data/.cache",
        help="XDG cache home directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # env setup
    os.environ["XDG_CACHE_HOME"] = args.xdg_cache_home
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # paths
    split = args.split
    reasoning_model = args.reasoning_model
    llm_name = args.llm_name

    split_path = os.path.join(args.data_dir, f"{split}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}")

    # prepare output
    split_output_dir = os.path.join(args.output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)
    run_idx = get_next_run_idx(split_output_dir, reasoning_model, llm_name)
    output_file = os.path.join(
        split_output_dir, f"{reasoning_model}_{llm_name}_run-{run_idx}.json"
    )
    info(f"Saving results to {output_file}")

    # init model
    info(
        f"Loading LLM from {args.model_path} "
        f"(tp={args.tensor_parallel_size}, gpu_mem_util={args.gpu_memory_utilization})"
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
    )

    # load data
    info(f"Loading datapoints from {split_path}")
    datapoints = json.load(open(split_path))
    inputs = [build_lm_input(dp, reasoning_model) for dp in datapoints]

    # num samples logic
    if args.num_samples_override is not None:
        num_samples = args.num_samples_override
    else:
        num_samples = (
            3 if reasoning_model in ["chameleon-plus", "vpgm-n2", "vpgm-n3", "vpgm-n4"] else 1
        )
    info(f"Using num_samples={num_samples} per question")

    batch_size = args.batch_size
    logs = []

    # generate
    for i in tqdm(range(0, len(inputs), batch_size), desc="Running ScienceQA"):
        batch_inputs = inputs[i : i + batch_size]
        if not batch_inputs:
            continue

        # collect multiple samples if needed
        batch_outputs_per_sample = []
        for _ in range(num_samples):
            # vLLM: returns a list of RequestOutput, one per input
            batch_outputs_per_sample.append(llm.generate(batch_inputs, sampling_params))

        # transpose: [num_inputs][num_samples]
        # each element is a list of RequestOutput objects
        batch_outputs = [list(sample_group) for sample_group in zip(*batch_outputs_per_sample)]

        for dp, output_list in zip(datapoints[i : i + batch_size], batch_outputs):
            pid = dp["pid"]
            gt_answer = ["A", "B", "C", "D", "E"][dp["answer"]]
            choices = dp["choices"]
            option_list = list(string.ascii_uppercase[: len(choices)])

            generated_text_list = [o.outputs[0].text for o in output_list]
            lm_answer_prob_dict = get_lm_answer_prob(
                generated_text_list, option_list, reasoning_model
            )

            logs.append(
                {
                    "pid": pid,
                    "choices": choices,
                    "gt_answer": gt_answer,
                    "lm_response": generated_text_list,
                    "lm_answer": lm_answer_prob_dict,
                }
            )

        # save incrementally
        with open(output_file, "w") as f:
            json.dump(logs, f, indent=2)

    info(f"Done. Total examples: {len(logs)}. Results written to {output_file}")


if __name__ == "__main__":
    main()
