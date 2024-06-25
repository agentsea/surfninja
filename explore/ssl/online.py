import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch
import transformers
from accelerate import dispatch_model
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


# Add this function at the top of your script
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "attention.wqkv",
            "attention.wo",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def auto_configure_device_map(num_gpus):
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus
    device_map = {
        "vit": 0,
        "vision_proj": 0,
        "model.tok_embeddings": 0,
        "model.norm": num_gpus - 1,
        "output": num_gpus - 1,
    }
    used = 3
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f"model.layers.{i}"] = gpu_target
        used += 1
    return device_map


class FeedbackDataset(Dataset):
    def __init__(self, tokenizer, img_size=224):
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.data = []

    def add_example(self, query, image, response, feedback):
        self.data.append(
            {
                "text_input": [f"<ImageHere>{query}", feedback],
                "image": image,
                "data_type": "image",
            }
        )

    def __getitem__(self, idx):
        return {"samples": self.data[idx]}

    def __len__(self):
        return len(self.data)

    def get_last_example(self):
        return self.data[-1] if self.data else None


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    if trainer.args.should_save:
        # check if zero3 mode enabled
        if trainer.is_deepspeed_enabled and trainer.deepspeed.zero3_enabled():
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        else:
            if trainer.args.use_lora:
                state_dict = get_peft_state_maybe_zero_3(
                    trainer.model.named_parameters(), bias
                )
            else:
                state_dict = trainer.model.state_dict()

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer._save(output_dir, state_dict=state_dict)


@dataclass
class DataCollatorForSupervisedDataset:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [instance["samples"] for instance in instances]
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ("text_input", "data_type")
        )
        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image"] = images
        return dict(samples=batch)


def initialize_or_load_peft_model(base_model, peft_model_path, lora_args):
    if os.path.exists(peft_model_path):
        print(f"Loading existing PEFT model from {peft_model_path}")
        model = PeftModel.from_pretrained(base_model, peft_model_path)
    else:
        print("Initializing new PEFT model")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    return model


def online_fine_tune(
    model,
    tokenizer,
    dataset,
    num_gpus,
    peft_model_path,
    lora_args,
    learning_rate=1e-5,
    num_epochs=1,
):
    if num_gpus > 1:
        device_map = auto_configure_device_map(num_gpus)
        model = dispatch_model(model, device_map=device_map)
    else:
        model = model.cuda()

    training_args = TrainingArguments(
        output_dir="./online_fine_tuned_model",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    data_collator = DataCollatorForSupervisedDataset()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    safe_save_model_for_hf_trainer(trainer, peft_model_path, bias=lora_args.lora_bias)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--dtype", default="fp16", type=str)
    parser.add_argument("--peft_model_path", default="./peft_model", type=str)
    parser.add_argument(
        "--immediate_train",
        action="store_true",
        help="Train immediately after each feedback",
    )
    parser.add_argument(
        "--accumulate_threshold",
        default=5,
        type=int,
        help="Number of examples to accumulate before training",
    )

    # Add LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=[
            "attention.wqkv",
            "attention.wo",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ],
    )
    parser.add_argument("--lora_bias", type=str, default="none")
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        device = "cpu"
        args.dtype = "fp32"  # Force fp32 if CUDA is not available
    else:
        device = "cuda"

    # Initialize model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        "internlm/internlm-xcomposer2-vl-7b",
        trust_remote_code=True,
    )
    config.use_cache = False

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        "internlm/internlm-xcomposer2-vl-7b",
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "internlm/internlm-xcomposer2-vl-7b", trust_remote_code=True
    )

    if device == "cuda":
        if args.dtype == "fp16":
            base_model = base_model.half()
        base_model = base_model.cuda()
    else:
        base_model = base_model.float()

    # We don't need to manually move the model to device here
    # The dispatch_model function in online_fine_tune will handle this

    # Create LoraArguments
    lora_args = LoraArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_bias=args.lora_bias,
    )

    # Initialize or load PEFT model
    model = initialize_or_load_peft_model(base_model, args.peft_model_path, lora_args)

    # Initialize dataset
    dataset = FeedbackDataset(tokenizer)

    try:
        while True:
            try:
                # Get user input
                text = input("Enter your query (or 'quit' to exit): ")
                if text.lower() == "quit":
                    break

                image_path = input("Enter image path: ")
                if not os.path.exists(image_path):
                    print(f"Image file not found: {image_path}")
                    continue

                # Preprocess the image
                image_tensor = preprocess_image(image_path)

                # Move image to the same device and dtype as the model
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                image_tensor = image_tensor.to(device=device, dtype=dtype)

                # Generate response
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        response, _ = model.chat(
                            tokenizer,
                            query=text,
                            image=image_tensor,
                            history=[],
                            do_sample=False,
                        )
                print("Model response:", response)

                # Get user feedback
                feedback = input("Enter your feedback: ")

                # Add to dataset
                dataset.add_example(
                    query=text, image=image_tensor, response=response, feedback=feedback
                )

                # Determine if we should train
                should_train = (
                    args.immediate_train or len(dataset) >= args.accumulate_threshold
                )

                if should_train:
                    print("Fine-tuning model...")
                    if args.immediate_train:
                        # Create a temporary dataset with just the last example
                        temp_dataset = FeedbackDataset(tokenizer)
                        last_example = dataset.get_last_example()
                        if last_example:
                            temp_dataset.add_example(**last_example)
                        train_dataset = temp_dataset
                    else:
                        train_dataset = dataset

                    model = online_fine_tune(
                        model,
                        tokenizer,
                        train_dataset,
                        args.num_gpus,
                        args.peft_model_path,
                        lora_args=lora_args,
                    )
                    print("Fine-tuning complete.")

                    # Clear the dataset if we're not doing immediate training
                    if not args.immediate_train:
                        dataset = FeedbackDataset(tokenizer)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Continuing to next iteration...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting gracefully...")
    finally:
        # Perform any cleanup or final saves here
        print("Saving final model state...")
        model.save_pretrained(args.peft_model_path)
        print("Exiting program.")
        sys.exit(0)


if __name__ == "__main__":
    main()
