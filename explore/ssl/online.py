import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch
import torchvision.transforms as transforms
from accelerate import dispatch_model
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments


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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []

    def add_example(self, query, image, response, feedback):
        print(
            f"Adding example to dataset: query={query}, response={response}, feedback={feedback}"
        )
        self.data.append(
            {"query": query, "image": image, "response": response, "feedback": feedback}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["query"] + " " + item["feedback"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        print(
            f"Processed inputs for item {idx}: input_ids shape={inputs['input_ids'].shape}"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze(),
            "image": item["image"],
        }

    def get_last_example(self):
        if self.data:
            return self.data[-1]
        return None


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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSupervisedDataset(),
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Save the updated PEFT model
    model.save_pretrained(peft_model_path)
    print(f"Model saved at {peft_model_path}")

    return model


class DataCollatorForSupervisedDataset:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([instance["input_ids"] for instance in instances])
        attention_masks = torch.stack(
            [instance["attention_mask"] for instance in instances]
        )
        labels = torch.stack([instance["labels"] for instance in instances])
        images = torch.cat(
            [instance["image"] for instance in instances], dim=0
        )  # Adjust this line if necessary
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "image": images,
        }


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
    base_model = AutoModel.from_pretrained(
        "internlm/internlm-xcomposer2-vl-7b", trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "internlm/internlm-xcomposer2-vl-7b", trust_remote_code=True
    )

    if device == "cuda":
        if args.dtype == "fp16":
            base_model = base_model.half().cuda()
        elif args.dtype == "fp32":
            base_model = base_model.cuda()
    else:
        base_model = base_model.to(device)

    print(f"Model and tokenizer loaded on {device}")

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
    print("PEFT model initialized or loaded")

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

                # Generate response
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        response, _ = model.chat(
                            tokenizer,
                            query=text,
                            image=image_path,
                            history=[],
                            do_sample=False,
                        )
                print("Model response:", response)

                # Get user feedback
                feedback = input("Enter your feedback: ")

                # Add to dataset
                image_tensor = model.encode_img(
                    image_path
                )  # Use model's image encoding
                dataset.add_example(text, image_tensor, response, feedback)

                # Determine if we should train
                should_train = (
                    args.immediate_train or len(dataset) >= args.accumulate_threshold
                )

                if should_train:
                    print("Fine-tuning model...")
                    if args.immediate_train:
                        # Create a temporary dataset with just the last example
                        temp_dataset = FeedbackDataset(tokenizer)
                        temp_dataset.add_example(**dataset.get_last_example())
                        train_dataset = temp_dataset
                    else:
                        train_dataset = dataset

                    model = online_fine_tune(
                        model,
                        tokenizer,
                        train_dataset,
                        args.num_gpus,
                        args.peft_model_path,
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
        # Add code to save the model state if needed
        print("Exiting program.")
        sys.exit(0)


if __name__ == "__main__":
    main()
