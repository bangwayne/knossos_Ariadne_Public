#!/usr/bin/env python3


from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_TASKS = (
    "one_step_simple_skill",
    "node_inventory_skill",
    "node_centric_relationship_skill",
    "tgp_foodweb_node_centric",
    "tgp_foodweb_edge_reasoning",
    "tgp_foodweb_edge_trace5_group",
    "tgp_foodweb_edge_v6_traceaware",
    "tgp_knossos_node_centric",
    "tgp_knossos_node_inventory",
    "tgp_knossos_edge_trace5_typed_multigraph",
)

DEFAULT_MODEL_PATH = (
    ""
    "speculative-verdict/models/GLM-4.1V-9B-Thinking"
)

DEFAULT_LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_VISION_TARGET_SUFFIXES = (
    "qkv",
    "proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_BRIDGE_KEYWORDS = (
    "model.visual.merger",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GLM-4.1V-9B-Thinking with LoRA/QLoRA on Knossos SFT data."
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("sft_data/manifests_v1"),
        help="Root folder containing per-task JSONL manifests.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional explicit train JSONL path.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional explicit eval JSONL path.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training even if an eval manifest is provided.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Base folder used to resolve relative image paths in the manifest.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local GLM-4.1V-9B-Thinking model folder.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional value such as sdpa or flash_attention_2.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-mode",
        choices=("text_only", "text_vision", "all_linear"),
        default="text_only",
    )
    parser.add_argument(
        "--lora-target-suffixes",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_SUFFIXES),
    )
    parser.add_argument(
        "--vision-target-suffixes",
        nargs="+",
        default=list(DEFAULT_VISION_TARGET_SUFFIXES),
    )
    parser.add_argument(
        "--bridge-keywords",
        nargs="+",
        default=list(DEFAULT_BRIDGE_KEYWORDS),
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--init-adapter-path",
        default=None,
        help="Optional existing LoRA adapter folder to load as the starting point for continued tuning.",
    )
    parser.add_argument("--save-final-merged", action="store_true")
    parser.add_argument(
        "--optim",
        default=None,
        help="Optional TrainingArguments optimizer override.",
    )
    return parser.parse_args()


@dataclass
class ManifestExample:
    sample_id: str
    image_path: str
    prompt: str
    target: Any
    system_prompt: str
    task: str
    assistant_text: Optional[str] = None

    @property
    def answer_text(self) -> str:
        if self.assistant_text is not None:
            return self.assistant_text
        return json.dumps(self.target, ensure_ascii=False, indent=2)


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        max_samples: Optional[int] = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.data_root = data_root.resolve()
        self.examples: List[ManifestExample] = []
        self._load(max_samples=max_samples)

    def _load(self, max_samples: Optional[int]) -> None:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and idx >= max_samples:
                    break
                row = json.loads(line)
                self.examples.append(
                    ManifestExample(
                        sample_id=row["sample_id"],
                        image_path=self._resolve_image_path(row["image_path"]),
                        prompt=row["prompt"],
                        target=row.get("target", row.get("final_graph", {})),
                        system_prompt=self._extract_system_prompt(row),
                        task=row["task"],
                        assistant_text=(
                            row.get("assistant_text")
                            or row.get("answer")
                            or row.get("response")
                        ),
                    )
                )

    def _resolve_image_path(self, image_path: str) -> str:
        path = Path(image_path)
        if path.is_absolute():
            return str(path)
        return str((self.data_root / path).resolve())

    @staticmethod
    def _extract_system_prompt(row: Dict[str, Any]) -> str:
        messages = row.get("messages") or []
        if not messages:
            return ""
        content = messages[0].get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                    text = str(part.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        return ""

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ManifestExample:
        return self.examples[idx]


class Glm41vThinkingCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.pad_token_id = int(self.tokenizer.pad_token_id)

    @staticmethod
    def _build_prefix_messages(example: ManifestExample) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if example.system_prompt:
            messages.append({"role": "system", "content": example.system_prompt})
        return messages

    def _build_user_messages(self, example: ManifestExample) -> List[Dict[str, Any]]:
        return self._build_prefix_messages(example) + [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example.prompt},
                ],
            }
        ]

    def _build_full_messages(self, example: ManifestExample) -> List[Dict[str, Any]]:
        return self._build_user_messages(example) + [
            {
                "role": "assistant",
                "content": example.answer_text,
            }
        ]

    @staticmethod
    def _load_images(features: Sequence[ManifestExample]) -> List[Image.Image]:
        return [Image.open(feature.image_path).convert("RGB") for feature in features]

    def __call__(self, features: Sequence[ManifestExample]) -> Dict[str, Any]:
        images = self._load_images(features)
        full_texts = [
            self.processor.apply_chat_template(
                self._build_full_messages(feature),
                tokenize=False,
                add_generation_prompt=False,
            )
            for feature in features
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._build_user_messages(feature),
                tokenize=False,
                add_generation_prompt=True,
            )
            for feature in features
        ]
        full_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = full_inputs["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : int(prompt_len)] = -100

        full_inputs["labels"] = labels
        return full_inputs


def infer_train_eval_manifests(args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    train_manifest = (
        args.train_manifest.resolve()
        if args.train_manifest is not None
        else (args.manifest_dir / args.task / "train.jsonl").resolve()
    )
    eval_manifest = (
        args.eval_manifest.resolve()
        if args.eval_manifest is not None
        else (args.manifest_dir / args.task / "val.jsonl").resolve()
    )

    if not train_manifest.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
    if args.no_eval:
        return train_manifest, None
    if not eval_manifest.exists():
        eval_manifest = None
    return train_manifest, eval_manifest


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Glm4vForConditionalGeneration

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=False,
    )

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "trust_remote_code": False,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if world_size > 1:
        model_kwargs["low_cpu_mem_usage"] = True
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("?????? --load-in-4bit ?????????????????? CUDA ?????????")
        model_kwargs["device_map"] = {"": local_rank}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolve_torch_dtype(args.bnb_4bit_compute_dtype),
        )

    model = Glm4vForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    print(
        f"[rank={os.environ.get('RANK', '0')}] "
        f"loaded model on cuda:{local_rank}" if torch.cuda.is_available() else
        f"[rank={os.environ.get('RANK', '0')}] loaded model on cpu"
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, processor


def _classify_linear_module(module_name: str) -> str:
    lowered = module_name.lower()
    if lowered.startswith("model.visual."):
        if ".merger." in lowered or lowered.endswith(".merger"):
            return "bridge"
        return "vision"
    return "text"


def _collect_lora_targets(model: Any, args: argparse.Namespace) -> Dict[str, List[str]]:
    import torch

    grouped_targets: Dict[str, List[str]] = {
        "text": [],
        "vision": [],
        "bridge": [],
    }
    blocked_suffixes = ("lm_head", "output_layer")
    seen: set[str] = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module_name in seen:
            continue
        leaf_name = module_name.split(".")[-1]
        if leaf_name in blocked_suffixes:
            continue
        module_group = _classify_linear_module(module_name)

        if args.lora_target_mode == "text_only":
            if module_group != "text":
                continue
            if leaf_name not in args.lora_target_suffixes:
                continue
        elif args.lora_target_mode == "text_vision":
            if module_group == "text":
                if leaf_name not in args.lora_target_suffixes:
                    continue
            elif module_group == "vision":
                if leaf_name not in args.vision_target_suffixes:
                    continue
            elif module_group == "bridge":
                if not any(keyword in module_name.lower() for keyword in args.bridge_keywords):
                    continue
                if leaf_name not in args.vision_target_suffixes:
                    continue
        elif args.lora_target_mode == "all_linear":
            pass

        grouped_targets[module_group].append(module_name)
        seen.add(module_name)

    total_targets = sum(len(items) for items in grouped_targets.values())
    if not total_targets:
        raise ValueError(
            "????????????????????? LoRA ?????????????????????????????? --lora-target-mode all_linear???"
        )
    for key in grouped_targets:
        grouped_targets[key] = sorted(grouped_targets[key])
    return grouped_targets


def pick_lora_targets(model: Any, args: argparse.Namespace) -> Tuple[List[str], Dict[str, List[str]]]:
    grouped_targets = _collect_lora_targets(model, args)
    target_modules: List[str] = []
    for group_name in ("text", "vision", "bridge"):
        target_modules.extend(grouped_targets[group_name])
    return target_modules, grouped_targets


def attach_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules, grouped_targets = pick_lora_targets(model, args)
    print(f"LoRA target mode: {args.lora_target_mode}")
    for group_name in ("text", "vision", "bridge"):
        group_items = grouped_targets[group_name]
        if not group_items:
            continue
        preview = ", ".join(group_items[:12])
        suffix = "" if len(group_items) <= 12 else f" ... (+{len(group_items) - 12} more)"
        print(f"LoRA {group_name} targets ({len(group_items)}): {preview}{suffix}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def maybe_load_initial_adapter(model: Any, args: argparse.Namespace) -> Any:
    if not args.init_adapter_path:
        return attach_lora(model, args)

    from peft import PeftModel

    adapter_path = Path(args.init_adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Initial adapter path not found: {adapter_path}")

    print(f"Loading initial adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
    )
    model.print_trainable_parameters()
    return model


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    optim_name = args.optim or ("paged_adamw_8bit" if args.load_in_4bit else "adamw_torch")

    strategy_value = "steps" if has_eval else "no"
    common_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else [],
        dataloader_num_workers=args.dataloader_num_workers,
        logging_strategy="steps",
        save_strategy="steps",
        optim=optim_name,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        seed=args.seed,
    )
    signature = inspect.signature(TrainingArguments.__init__)
    supported_names = set(signature.parameters.keys())

    optional_kwargs: Dict[str, Any] = {}
    if "overwrite_output_dir" in supported_names:
        optional_kwargs["overwrite_output_dir"] = True
    if "eval_strategy" in supported_names:
        optional_kwargs["eval_strategy"] = strategy_value
    elif "evaluation_strategy" in supported_names:
        optional_kwargs["evaluation_strategy"] = strategy_value

    final_kwargs: Dict[str, Any] = {}
    for key, value in {**common_kwargs, **optional_kwargs}.items():
        if value is None:
            continue
        if key in supported_names:
            final_kwargs[key] = value

    unsupported = sorted(
        key
        for key, value in {**common_kwargs, **optional_kwargs}.items()
        if value is not None and key not in supported_names
    )
    if unsupported:
        print("TrainingArguments will ignore unsupported keys:", ", ".join(unsupported))

    return TrainingArguments(**final_kwargs)


def dump_run_config(
    args: argparse.Namespace,
    train_manifest: Path,
    eval_manifest: Optional[Path],
    train_size: int,
    eval_size: int,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "model_path": args.model_path,
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest) if eval_manifest is not None else None,
        "train_size": train_size,
        "eval_size": eval_size,
        "load_in_4bit": args.load_in_4bit,
        "torch_dtype": args.torch_dtype,
        "lora_target_mode": args.lora_target_mode,
        "lora_target_suffixes": list(args.lora_target_suffixes),
        "vision_target_suffixes": list(args.vision_target_suffixes),
        "bridge_keywords": list(args.bridge_keywords),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "init_adapter_path": args.init_adapter_path,
    }
    with open(args.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train_manifest, eval_manifest = infer_train_eval_manifests(args)
    train_dataset = ManifestDataset(
        manifest_path=train_manifest,
        data_root=args.data_root,
        max_samples=args.max_train_samples,
    )
    eval_dataset = (
        ManifestDataset(
            manifest_path=eval_manifest,
            data_root=args.data_root,
            max_samples=args.max_eval_samples,
        )
        if eval_manifest is not None
        else None
    )

    dump_run_config(
        args=args,
        train_manifest=train_manifest,
        eval_manifest=eval_manifest,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset) if eval_dataset is not None else 0,
    )

    model, processor = load_model_and_processor(args)
    model = maybe_load_initial_adapter(model, args)
    data_collator = Glm41vThinkingCollator(processor)
    training_args = build_training_arguments(args, has_eval=eval_dataset is not None)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.save_final_merged:
        if args.load_in_4bit:
            raise ValueError("4bit ????????????????????? merge LoRA ?????????")
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
