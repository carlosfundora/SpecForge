import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT_DIR / "cache"
DEFAULT_DATASET_DIR = DEFAULT_CACHE_DIR / "dataset"
DEFAULT_COMPILED_KERNEL_DIR = DEFAULT_CACHE_DIR / "compiled_kernels"
DEFAULT_MODEL_REGISTRY = Path("/home/local/Projects/models/registry")
THOTH_ROOT = ROOT_DIR.parent
SGLANG_PYTHON_DIR = THOTH_ROOT / "sglang" / "python"
DEFAULT_THOTH_ARTIFACT_ROOT = THOTH_ROOT / "artifacts" / "models" / "local"


PROFILES = {
    "bonsai17": {
        "target_model_path": "prism-ml/Bonsai-1.7B-unpacked",
        "target_model_note": "Public unpacked Bonsai-1.7B target for HF/Transformers training.",
        "chat_template": "qwen",
        "max_length": 512,
        "embedding_key": "model.embed_tokens.weight",
        "target_model_backend": "hf",
        "attention_backend": "sdpa",
        "train_mask_hidden_only": False,
        "default_k_train": 5,
        "default_ttt_length": 5,
        "default_num_epochs": 20,
        "default_max_num_steps": None,
        "default_sample_limit": None,
        "warm_start_ckpt": str(
            DEFAULT_MODEL_REGISTRY / "local" / "Bonsai-1.7B-EAGLE3-local" / "weights"
        ),
        "output_dir": str(
            DEFAULT_MODEL_REGISTRY / "local" / "Bonsai-1.7B-P-EAGLE-local"
        ),
        "train_data_path": str(DEFAULT_DATASET_DIR / "sharegpt_train.jsonl"),
        "dataset_name": "sharegpt",
    },
    "bonsai17_smoke": {
        "target_model_path": "prism-ml/Bonsai-1.7B-unpacked",
        "target_model_note": (
            "Public unpacked Bonsai-1.7B target for a fast smoke run. "
            "Uses a shorter context and lower K_train to prove the stack can train/save."
        ),
        "chat_template": "qwen",
        "max_length": 256,
        "embedding_key": "model.embed_tokens.weight",
        "target_model_backend": "hf",
        "attention_backend": "sdpa",
        "train_mask_hidden_only": True,
        "default_k_train": 5,
        "default_ttt_length": 5,
        "default_num_epochs": 1,
        "default_max_num_steps": 500,
        "default_sample_limit": 1024,
        "warm_start_ckpt": str(
            DEFAULT_MODEL_REGISTRY / "local" / "Bonsai-1.7B-EAGLE3-local" / "weights"
        ),
        "output_dir": str(
            DEFAULT_THOTH_ARTIFACT_ROOT / "Bonsai-1.7B-P-EAGLE-local-smoke"
        ),
        "train_data_path": str(DEFAULT_DATASET_DIR / "sharegpt_train.jsonl"),
        "dataset_name": "sharegpt",
    },
    "opcoder15": {
        "target_model_path": str(
            DEFAULT_MODEL_REGISTRY / "infly" / "OpenCoder-1.5B-Instruct" / "weights"
        ),
        "target_model_note": "Local HF-style OpenCoder weights checkout.",
        "chat_template": "llama3",
        "max_length": 4096,
        "embedding_key": "model.embed_tokens.weight",
        "target_model_backend": "hf",
        "attention_backend": "flex_attention",
        "train_mask_hidden_only": False,
        "default_k_train": 8,
        "default_ttt_length": 7,
        "default_num_epochs": 5,
        "default_max_num_steps": None,
        "default_sample_limit": None,
        "warm_start_ckpt": str(
            DEFAULT_MODEL_REGISTRY / "local" / "OpenCoder-1.5B-EAGLE3-local" / "weights"
        ),
        "output_dir": str(
            DEFAULT_MODEL_REGISTRY / "local" / "OpenCoder-1.5B-P-EAGLE-local"
        ),
        "train_data_path": str(DEFAULT_DATASET_DIR / "sharegpt_train.jsonl"),
        "dataset_name": "sharegpt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch P-EAGLE fine-tuning from an existing local EAGLE-3 head."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="bonsai17",
        help="Preset training profile. bonsai17 is the THOTH default target.",
    )
    parser.add_argument("--target-model-path", type=str, default=None)
    parser.add_argument("--eagle3-head", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--draft-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.0025)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--train-mask-hidden-only", action="store_true")
    parser.add_argument("--k-train", type=int, default=None)
    parser.add_argument("--ttt-length", type=int, default=None)
    parser.add_argument("--cod-retention", type=float, default=0.8)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--max-num-steps", type=int, default=None)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optionally write and use only the first N JSONL samples from train-data-path.",
    )
    parser.add_argument(
        "--prepare-data-if-missing",
        action="store_true",
        help="Generate the preset dataset locally with scripts/prepare_data.py if train-data-path is missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    return parser.parse_args()


def maybe_prepare_dataset(dataset_name: str, train_data_path: str, cache_dir: str, dry_run: bool):
    if os.path.exists(train_data_path):
        return

    output_path = Path(cache_dir) / "dataset"
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "prepare_data.py"),
        "--dataset",
        dataset_name,
        "--output-path",
        str(output_path),
    ]
    if dry_run:
        print("DATASET CMD:", " ".join(cmd))
        return

    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


def maybe_materialize_sampled_dataset(
    source_path: str, cache_dir: str, sample_limit: int | None, dry_run: bool
) -> str:
    if sample_limit is None:
        return source_path

    source = Path(source_path)
    sampled_dir = Path(cache_dir) / "dataset"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    sampled_path = sampled_dir / f"{source.stem}.sample-{sample_limit}{source.suffix}"

    if sampled_path.exists():
        return str(sampled_path)

    if dry_run:
        print(f"SAMPLED DATASET: {sampled_path} (first {sample_limit} lines from {source_path})")
        return str(sampled_path)

    kept = 0
    with source.open("r", encoding="utf-8") as src, sampled_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            json.loads(line)
            dst.write(line)
            kept += 1
            if kept >= sample_limit:
                break
    if kept == 0:
        raise ValueError(f"No JSONL samples were written to {sampled_path} from {source_path}")
    return str(sampled_path)


def main():
    args = parse_args()
    profile = PROFILES[args.profile]

    target_model_path = args.target_model_path or profile["target_model_path"]
    eagle3_head = args.eagle3_head or profile["warm_start_ckpt"]
    output_dir = args.output_dir or profile["output_dir"]
    train_data_path = args.train_data_path or profile["train_data_path"]
    dataset_name = args.dataset_name or profile["dataset_name"]
    max_length = args.max_length or profile["max_length"]
    attention_backend = args.attention_backend or profile["attention_backend"]
    train_mask_hidden_only = (
        True if args.train_mask_hidden_only else bool(profile.get("train_mask_hidden_only", False))
    )
    num_epochs = args.num_epochs or profile["default_num_epochs"]
    k_train = args.k_train or profile["default_k_train"]
    ttt_length = (
        args.ttt_length
        if args.ttt_length is not None
        else profile.get("default_ttt_length", k_train)
    )
    max_num_steps = (
        args.max_num_steps
        if args.max_num_steps is not None
        else profile["default_max_num_steps"]
    )
    sample_limit = (
        args.sample_limit if args.sample_limit is not None else profile["default_sample_limit"]
    )

    if args.prepare_data_if_missing:
        maybe_prepare_dataset(
            dataset_name=dataset_name,
            train_data_path=train_data_path,
            cache_dir=args.cache_dir,
            dry_run=args.dry_run,
        )
    train_data_path = maybe_materialize_sampled_dataset(
        source_path=train_data_path,
        cache_dir=args.cache_dir,
        sample_limit=sample_limit,
        dry_run=args.dry_run,
    )

    env = os.environ.copy()
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DEFAULT_COMPILED_KERNEL_DIR))
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/thoth-pycache")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(ROOT_DIR), str(SGLANG_PYTHON_DIR)]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    train_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(args.num_gpus),
        str(ROOT_DIR / "scripts" / "train_eagle3.py"),
        "--target-model-path",
        target_model_path,
        "--ckpt-dir",
        eagle3_head,
        "--speculative-algorithm",
        "P_EAGLE",
        "--parallel-drafting",
        "--train-data-path",
        train_data_path,
        "--output-dir",
        output_dir,
        "--num-epochs",
        str(num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--draft-accumulation-steps",
        str(args.draft_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-length",
        str(max_length),
        "--attention-backend",
        attention_backend,
        *(
            ["--train-mask-hidden-only"]
            if train_mask_hidden_only
            else []
        ),
        "--chat-template",
        profile["chat_template"],
        "--cache-dir",
        args.cache_dir,
        "--embedding-key",
        profile["embedding_key"],
        "--tp-size",
        str(args.tp_size),
        "--target-model-backend",
        profile["target_model_backend"],
        "--build-dataset-num-proc",
        str(args.build_dataset_num_proc),
        "--k-train",
        str(k_train),
        "--ttt-length",
        str(ttt_length),
        "--cod-retention",
        str(args.cod_retention),
        "--log-interval",
        str(args.log_interval),
        "--save-interval",
        str(args.save_interval),
        "--eval-interval",
        str(args.eval_interval),
    ]
    if max_num_steps is not None:
        train_cmd.extend(["--max-num-steps", str(max_num_steps)])

    print(f"Profile: {args.profile}")
    print(f"Target model path: {target_model_path}")
    print(f"Target model note: {profile['target_model_note']}")
    print(f"Warm-start EAGLE3 head: {eagle3_head}")
    print(f"Train data path: {train_data_path}")
    print(f"Sample limit: {sample_limit}")
    print(f"Attention backend: {attention_backend}")
    print(f"Train mask_hidden only: {train_mask_hidden_only}")
    print(f"TTT length: {ttt_length}")
    print(f"Output dir: {output_dir}")
    print("TRAIN CMD:", " ".join(train_cmd))

    if args.dry_run:
        return

    subprocess.run(train_cmd, check=True, cwd=str(ROOT_DIR), env=env)


if __name__ == "__main__":
    main()
