import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Constants
MODEL_NAME = "internlm/internlm-xcomposer2-4khd-7b"
TEST_DATA_PATH = "test_data_with_images.json"  # Path to the test JSON file with images
OUTPUT_DIR = "./fine_tuned_model"
LOG_DIR = "./logs"
HD_NUM = 55  # Set hd_num to a positive integer as required

# Image transformation (no resizing)
image_transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet means and stds
    ]
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
).cuda()
model = model.eval()

# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Adjust the rank as needed
    lora_alpha=32,
    target_modules=[
        "attention.wqkv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    ],  # Adjust target modules based on model architecture
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir=LOG_DIR,
    logging_steps=10,  # Log every 10 steps
    save_steps=10,
    save_total_limit=2,
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    bf16=True,  # Enable mixed precision training if your hardware supports it
    report_to="tensorboard",  # Report to TensorBoard
)


def load_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0).cuda()


def inference(model, tokenizer, prompt: str, image_path: str) -> str:
    image_tensor = load_image(image_path)
    with torch.cuda.amp.autocast():
        response, _ = model.chat(
            tokenizer,
            query=prompt,
            image=image_tensor,
            hd_num=HD_NUM,
            history=[],
            do_sample=False,
            num_beams=3,
        )
    return response


def fine_tune(model, tokenizer, data_path, training_args):
    with open(data_path, "r") as f:
        data = json.load(f)

    dataset = Dataset.from_dict({"samples": data})

    def data_collator(data):
        input_ids = []
        labels = []
        images = []
        for sample in data["samples"]:
            for conv in sample["conversations"]:
                input_ids.append(tokenizer.encode(conv["value"], truncation=True))
                labels.append(tokenizer.encode(conv["value"], truncation=True))
            images.append([load_image(img) for img in sample["image"]])
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "images": images,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()


def main():
    # Test before fine-tuning
    prompt = "<ImageHere> What is in this image?"
    image_path = "path/to/image1.jpg"
    print("Before fine-tuning:")
    print(inference(model, tokenizer, prompt, image_path))

    # Fine-tune the model
    fine_tune(model, tokenizer, TEST_DATA_PATH, training_args)

    # Test after fine-tuning
    print("After fine-tuning:")
    print(inference(model, tokenizer, prompt, image_path))

    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
