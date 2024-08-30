import math
from typing import List, Optional, Tuple

import datasets
import torch
import torch.nn as nn
from datasets.iterable_dataset import IterableDataset
from omegaconf import open_dict
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    T5ForConditionalGeneration,
)

from .copied_utils import (
    DataCollatorForNI,
    DataCollatorForT5MLM,
    compute_input_and_target_lengths,
    tokenize_function,
)
from .t5_model import MyT5


def get_model(args, config):
    klass = {
        "hf_t5": T5ForConditionalGeneration,
        "local_t5": MyT5,
    }[args.model.klass]

    if args.model.checkpoint_path:
        model = klass(config)
        model.load_state_dict(torch.load(args.model.checkpoint_path))
    elif args.model.random_init:
        model = klass(config)
    else:
        assert (
            klass == T5ForConditionalGeneration
        ), "To load HFs weights you need to use HF model"
        model = klass.from_pretrained(
            args.model.name,
            config=config,
        )

    with open_dict(args):
        args.n_all_param = sum([p.nelement() for p in model.parameters()])

    return model


def get_config(args, tokenizer: AutoTokenizer):
    config = AutoConfig.from_pretrained(args.model.name)

    # Update config with new vocab size and relevant special tokens
    config.vocab_size = math.ceil(len(tokenizer) / 128) * 128  # ensure divisible by 128

    if config.pad_token_id is None or config.pad_token_id != tokenizer.pad_token_id:
        config.pad_token_id = tokenizer.pad_token_id
    if (
        config.decoder_start_token_id is None
        or config.decoder_start_token_id != tokenizer.pad_token_id
    ):
        config.decoder_start_token_id = tokenizer.pad_token_id
    if config.eos_token_id is None or config.eos_token_id != tokenizer.eos_token_id:
        config.eos_token_id = tokenizer.eos_token_id

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    return config


def get_tokenizer(args):
    # access args.tokenizer, defaulting to None if it doesn't exist
    tokenizer_config = getattr(args, "tokenizer", None)

    # If tokenizer config exists and has a name, use it; otherwise, use the model name
    tokenizer_name = getattr(tokenizer_config, "name", None) or args.model.name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    # check for pad token and eos token
    assert (
        tokenizer.pad_token_id is not None and tokenizer.eos_token_id is not None
    ), "Tokenizer should have pad_token_id and eos_token_id"

    # check to make sure T5 special tokens are in tokenizer
    t5_mask_tokens = [f"<extra_id_{i}>" for i in range(100)]
    _vocab = tokenizer.get_vocab()
    assert all(
        token in _vocab for token in t5_mask_tokens
    ), "T5 special tokens are not in tokenizer"

    return tokenizer


def load_dataset_splits(args):
    if args.mode == "pt":
        ds_fw = datasets.load_dataset(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", streaming=True
        )
        ds_c4 = datasets.load_dataset(
            "c4", "en", streaming=True, trust_remote_code=True
        )

        ds_fw = ds_fw.remove_columns(["id", "metadata"])
        ds_c4 = ds_c4.remove_columns(["timestamp", "url"])

        dataset_splits = {
            "train": ds_fw["train"],
            "test": ds_c4["validation"],
        }

        # assert (
        #     dataset["train"].n_shards == 1024
        # ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"
    elif args.mode == "ft":
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task,
        )
    else:
        raise NotImplementedError

    return dataset_splits


def process_dataset(dataset_splits, args, tokenizer):
    if args.mode == "pt":
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():
            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "in_length": before_mask_input_length,
                },
                remove_columns=["text"],
            )

            dataset_split = dataset_split.shuffle(buffer_size=10_000, seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == "ft":
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets


def get_data_collator(tokenizer, config, args):
    if args.mode == "pt":
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
        )
    elif args.mode == "ft":
        data_collator = DataCollatorForNI(
            tokenizer,
            padding="longest",
            max_source_length=args.data.max_seq_len,
            max_target_length=args.data.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            add_task_name=args.data.add_task_name,
            add_task_definition=args.data.add_task_definition,
            num_pos_examples=args.data.num_pos_examples,
            num_neg_examples=args.data.num_neg_examples,
            add_explanation=args.data.add_explanation,
            tk_instruct=args.data.tk_instruct,
        )
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(tokenizer, config, args):
    dataset_splits = load_dataset_splits(args)
    dataset = process_dataset(
        dataset_splits=dataset_splits, args=args, tokenizer=tokenizer
    )
    data_collator = get_data_collator(tokenizer=tokenizer, config=config, args=args)

    is_iterable = isinstance(dataset["train"], IterableDataset)

    dataloaders = {}

    for split in ["train", "test"]:
        batch_size = args.optim.batch_size // args.optim.grad_acc

        shuffle = (split == "train") and not is_iterable

        if args.mode == "ft" and split == "train":
            assert shuffle is True
        else:
            assert shuffle is False

        dataloaders[split] = DataLoader(
            dataset[split],
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders["train"])
            args.data.test_batches = len(dataloaders["test"])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = (
                len(dataloaders["train"]) // args.optim.grad_acc
            ) * args.optim.epochs

        args.eval.corrected_steps = args.eval.steps

    return dataloaders["train"], dataloaders["test"]


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == "adamw":
        from transformers import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == "adamwscale":
        from .copied_utils import AdamWScale

        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == "adafactor":
        from transformers import Adafactor

        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps],
        )
    elif args.optim.lr_scheduler == "legacy":
        import math

        from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: (
                min(1e-2, 1.0 / math.sqrt(step)) / args.optim.base_lr
                if step
                else 1e-2 / args.optim.base_lr
            ),
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1],
        )
    elif args.optim.lr_scheduler == "constant":
        from transformers import get_scheduler

        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler


def model_summary(
    model: PreTrainedModel, max_depth: int = 4, show_input_size: bool = False
) -> None:
    """
    Prints an accurate summary of the model, avoiding double-counting of parameters.

    :param PreTrainedModel model: torch model to summarize
    :param int max_depth: maximum depth of the model to print, defaults to 4
    :param bool show_input_size: whether to show input size for each layer, defaults to False
    """

    def format_params(num_params: int) -> str:
        return f"{num_params:,}" if num_params > 0 else "--"

    def format_size(size: Optional[List[int]]) -> str:
        return "x".join(str(x) for x in size) if size else "N/A"

    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        return total_params, trainable_params

    def recursive_summarize(
        module: nn.Module, depth: int, idx: List[int], prefix: str = ""
    ) -> List[Tuple[str, int, int, int, Optional[List[int]], nn.Module]]:
        summary = []

        total_params, trainable_params = count_parameters(module)

        if depth <= max_depth:
            layer_name = f"{prefix}{type(module).__name__}"
            layer_index = ".".join(map(str, idx))
            param_shape = next(
                (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
                None,
            )
            summary.append(
                (layer_name, depth, total_params, trainable_params, param_shape, module)
            )

            for i, (name, child) in enumerate(module.named_children(), 1):
                child_summary = recursive_summarize(
                    child, depth + 1, idx + [i], prefix + "  "
                )
                summary.extend(child_summary)

        return summary

    summary = recursive_summarize(model, 1, [1])

    max_name_length = max(len(name) for name, _, _, _, _, _ in summary)
    max_shape_length = max(len(format_size(shape)) for _, _, _, _, shape, _ in summary)

    print("=" * (max_name_length + 50))
    header = f"{'Layer (type:depth-idx)':<{max_name_length}} {'Output Shape':>{max_shape_length}} {'Param #':>12} {'Trainable':>10}"
    print(header)
    print("=" * (max_name_length + 50))

    for name, depth, num_params, trainable_params, shape, _ in summary:
        shape_str = format_size(shape) if show_input_size else ""
        print(
            f"{name:<{max_name_length}} {shape_str:>{max_shape_length}} {format_params(num_params):>12} {str(trainable_params > 0):>10}"
        )

    total_params, trainable_params = count_parameters(model)
    print("=" * (max_name_length + 50))
    print(f"Total params: {format_params(total_params)}")
    print(f"Trainable params: {format_params(trainable_params)}")
    print(f"Non-trainable params: {format_params(total_params - trainable_params)}")
    print("=" * (max_name_length + 50))
