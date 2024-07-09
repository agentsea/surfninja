# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import neptune
import torch
import transformers
from accelerate.utils import DistributedType
from data_mix import Mix_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Trainer, deepspeed
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import evaluate
from tqdm import tqdm


def custom_evaluate(model, eval_dataset, tokenizer, metric_name="rouge"):
    model.eval()
    metric = evaluate.load(metric_name)
    results = []

    # Determine model dtype
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Model dtype: {model_dtype}")

    logger.info(f"Starting evaluation with {len(eval_dataset)} samples")
    logger.info(f"First sample keys: {eval_dataset[0].keys()}")
    logger.info(f"First sample content: {eval_dataset[0]}")

    for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        logger.info(f"Processing eval sample {idx}")
        # logger.info(f"Sample keys: {sample.keys()}")
        # logger.info(f"Sample content: {sample}")

        try:
            # Access the nested 'samples' dictionary
            sample_data = sample["samples"]
            query = sample_data["text_input"][0]  # Assuming it's a list with one item
            logger.info(f"Query: {query}")
        except KeyError as e:
            logger.error(f"KeyError in sample {idx}: {str(e)}")
            logger.error(f"Sample content: {sample}")
            continue

        image = sample_data.get("image")
        if image is not None:
            # Convert image to model dtype
            image = image.to(model_dtype)
        logger.info(f"Image present: {image is not None}")

        # Extract ground truth from the query
        ground_truth = query.split("[UNUSED_TOKEN_146]assistant\n")[-1].split(
            "[UNUSED_TOKEN_145]"
        )[0]
        logger.info(f"Ground truth: {ground_truth}")

        try:
            # Use the chat method for inference
            with torch.cuda.amp.autocast(dtype=model_dtype):
                response, _ = model.chat(
                    tokenizer,
                    query,
                    image=image,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                )
            logger.info(f"Model response: {response}")
        except Exception as e:
            logger.error(f"Error during model.chat for sample {idx}: {str(e)}")
            logger.exception("Stack trace:")
            continue

        results.append({"prediction": response, "reference": ground_truth})

    logger.info(f"Evaluation completed with {len(results)} valid results")

    # Compute metrics
    metric_results = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results],
    )

    logger.info(f"Metric results: {metric_results}")

    return metric_results, results


def inspect_dataset(dataset, name):
    print(f"Inspecting {name} dataset:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, (list, tuple)):
                print(f"    First element: {type(value[0])}")
                print(
                    f"    Value: {value[:100]}..."
                )  # Print first 100 characters if it's a long string
            elif isinstance(value, dict):
                print(f"    Keys: {value.keys()}")
            else:
                print(f"    Value: {value}")
        print()


from transformers import Trainer, TrainingArguments


class DebugTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float("-inf")

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     logger.debug(f"Computing loss with inputs: {inputs.keys()}")
    #     outputs = model(**inputs)
    #     loss = outputs.loss if return_outputs else outputs[0]
    #     return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self.model.eval()

        logger.info("Starting evaluation loop")
        logger.info(f"Eval dataset type: {type(self.eval_dataset)}")
        logger.info(f"Eval dataset length: {len(self.eval_dataset)}")
        logger.info(f"First eval sample: {self.eval_dataset[0]}")

        try:
            metric_results, eval_results = custom_evaluate(
                self.model, self.eval_dataset, self.tokenizer
            )
        except Exception as e:
            logger.error(f"Error during custom_evaluate: {str(e)}")
            logger.exception("Stack trace:")
            raise

        # Log results
        logger.info(f"Evaluation results: {metric_results}")
        for key, value in metric_results.items():
            self.log({f"{metric_key_prefix}_{key}": value})

        # Save best model
        if metric_results.get("rouge1", 0) > self.best_metric:
            self.best_metric = metric_results["rouge1"]
            logger.info(f"New best model with rouge1 score: {self.best_metric}")
            self.save_model(os.path.join(self.args.output_dir, "best_model"))

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metric_results,
            num_samples=len(self.eval_dataset),
        )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default="data.txt", metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    given_num: bool = False
    img_size: int = 224
    batch_size: int = 4
    hd_num: int = -1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    label_names: List[str] = field(default_factory=lambda: ["samples"])


@dataclass
class LoraArguments:
    lora_r: int = 256
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


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
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
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # logger.info(f"Collating {len(instances)} instances")
        instances = [instance["samples"] for instance in instances]
        text_input, data_type = tuple(
            [instance[key] for instance in instances]
            for key in ("text_input", "data_type")
        )

        # logger.info(f"Text input type: {type(text_input)}, Data type: {type(data_type)}")

        if "image" not in instances[0]:
            text_input = [instance["text_input"][0] for instance in instances]

        batch = dict(
            text_input=text_input,
            data_type=data_type,
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image"] = images

        # logger.info(f"Batch keys: {batch.keys()}")
        # logger.info(f"Batch['text_input'] type: {type(batch['text_input'])}")
        # logger.info(f"Batch['data_type'] type: {type(batch['data_type'])}")

        # Ensure all values are tensors
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                if isinstance(value[0], (int, float)):
                    batch[key] = torch.tensor(value)
                elif isinstance(value[0], str):
                    # You might need to tokenize strings or handle them differently
                    logger.debug(f"String value found in batch for key {key}")

        return dict(samples=batch)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")
    if data_args.data_path.endswith("json"):
        train_json = json.load(open(data_args.data_path))
        train_json = {data_args.data_path: train_json}
    elif data_args.data_path.endswith("txt"):
        train_json = {}
        with open(data_args.data_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            line = line.split(" ")
            with open(line[0]) as f:
                temp = json.load(f)
            if data_args.given_num:
                assert len(line) == 2
                num = int(float(line[1]) * 1000)
                if len(temp) > num:
                    temp = random.sample(temp, num)
                else:
                    ex_temp = []
                    for i in range(num - len(temp)):
                        ex_temp.append(random.choice(temp))
                    temp.extend(ex_temp)
            else:
                if len(line) == 2:
                    ratio = float(line[1])
                    new_len = int(len(temp) * ratio)
                    if ratio < 1:
                        temp = random.sample(temp, new_len)
                    elif ratio > 1:
                        ex_temp = []
                        for i in range(new_len - len(temp)):
                            ex_temp.append(random.choice(temp))
                        temp.extend(ex_temp)
            rank0_print(f"Load {len(temp)} samples from {line}")
            train_json[line[0]] = temp
    train_dataset = Mix_dataset(
        train_json,
        data_args.batch_size,
        img_size=data_args.img_size,
        hd_num=data_args.hd_num,
        local_rank=local_rank,
    )
    print(str(len(train_dataset)) + " train samples loaded")

    # Load evaluation dataset similarly (assume data_args.eval_data_path)
    if hasattr(data_args, "eval_data_path") and data_args.eval_data_path:
        with open(data_args.eval_data_path) as f:
            eval_json = json.load(f)
            eval_json = {data_args.eval_data_path: eval_json}
        logger.info(
            f"Loaded {len(eval_json[data_args.eval_data_path])} evaluation samples"
        )
        logger.info(f"First eval sample: {eval_json[data_args.eval_data_path][0]}")
        eval_dataset = Mix_dataset(
            eval_json,
            data_args.batch_size,
            img_size=data_args.img_size,
            hd_num=data_args.hd_num,
            local_rank=local_rank,
        )
    else:
        eval_dataset = None
        logger.warning("No evaluation dataset provided")

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def inspect_dataset(dataset, name):
    print(f"Inspecting {name} dataset:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, (list, tuple)):
                print(f"    First element: {type(value[0])}")
                print(
                    f"    Value: {value[:100]}..."
                )  # Print first 100 characters if it's a long string
            elif isinstance(value, dict):
                print(f"    Keys: {value.keys()}")
            else:
                print(f"    Value: {value}")
        print()


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # Set up model
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    if data_args.img_size != 336:
        model.vit.resize_pos()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    if training_args.fix_vit:
        model.vit.requires_grad_(False)
    else:
        model.vit.requires_grad_(True)
        model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity()

    if training_args.fix_sampler:
        model.vision_proj.requires_grad_(False)
    else:
        model.vision_proj.requires_grad_(True)

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    logger.info("Data module created")
    logger.info(f"Train dataset length: {len(data_module['train_dataset'])}")
    logger.info(f"Eval dataset length: {len(data_module['eval_dataset'])}")
    logger.info(f"First train sample: {data_module['train_dataset'][0]}")
    logger.info(f"First eval sample: {data_module['eval_dataset'][0]}")

    # Initialize our Trainer
    trainer = DebugTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
    )

    logger.info("Trainer initialized")

    # Train the model
    logger.info("Starting training")
    trainer.train()

    # Save the final model
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias if training_args.use_lora else "none",
    )

    # Evaluate the final model
    # if training_args.do_eval:
    #     final_metrics, _ = custom_evaluate(trainer.model, data_module['eval_dataset'], tokenizer)
    #     print("Final Evaluation Metrics:", final_metrics)

    print("Evaluating model...")
    final_metrics, _ = custom_evaluate(
        trainer.model, data_module["eval_dataset"], tokenizer
    )
    print("Final Evaluation Metrics:", final_metrics)


##
### ONLINE ###
##

import base64
import io
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

from agentdesk import Desktop
from mllm import RoleMessage, RoleThread, Router
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class ClickTarget(BaseModel):
    """A target which the mouse could be moved to and clicked"""

    description: str = Field(
        description="A long detailed description of the target e.g. A round blue button with the text 'login'"
    )
    purpose: str = Field(
        description="A general purpose of the target e.g. 'log the user in' or 'search for a product'"
    )
    expectation: str = Field(
        description="An expectation on what will happen when you click this target e.g. 'A login screen will appear'"
    )
    near: str = Field(
        description="Describe what is near the target e.g. the 'Get' button is near the text 'Awesome App'"
    )


def describe_location(
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Describe the current location of the mouse"""

    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    size = img.size
    coords = desktop.mouse_coordinates()

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    image.save(img_path)

    thread = RoleThread()
    cropped = crop_image(img_path, bbox)
    img = Image.open(img_path)

    thread.post(
        role="user",
        msg=f"""I'm going to provide you with two images. The first is a picture of a desktop UI with a bounding box around an element. 
    The second is a cropped portion of the first image containing just the bounding box.
    Please describe the element in the bounding box using the schema {ClickTarget.model_json_schema()}.
    Please return just raw json. For example if you see the mouse above the chromium icon then 
    you would return {{"description": "A blue chromium icon with the text 'chromium' beneath it", "purpose": "to open chromium browser", "expectation": "a web browser will open and be visible on the screen"}}.
    """,
        images=[img, cropped],
    )

    resp = router.chat(thread, model=model, expect=ClickTarget)

    if not resp.parsed:
        raise ValueError("No click area found")

    prompt = f"<ImageHere> describe the current mouse cursor position. Return a JSON object adhearing to the schema {ClickTarget.model_json_schema()}"

    return resp.parsed, {
        "id": str(uuid.uuid4()),
        "image": [img_path],
        "conversations": [
            {
                "from": "user",
                "value": prompt,
            },
            {
                "from": "assistant",
                "value": resp.parsed.model_dump_json(),
            },
        ],
    }


def image_to_b64(img: Image.Image, image_format="PNG") -> str:
    """Converts a PIL Image to a base64-encoded string with MIME type included.

    Args:
        img (Image.Image): The PIL Image object to convert.
        image_format (str): The format to use when saving the image (e.g., 'PNG', 'JPEG').

    Returns:
        str: A base64-encoded string of the image with MIME type.
    """
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    image_data = buffer.getvalue()
    buffer.close()

    mime_type = f"image/{image_format.lower()}"
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def b64_to_image(base64_str: str) -> Image.Image:
    """Converts a base64 string to a PIL Image object.

    Args:
        base64_str (str): The base64 string, potentially with MIME type as part of a data URI.

    Returns:
        Image.Image: The converted PIL Image object.
    """
    # Strip the MIME type prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


class MouseCoordsPrediction(BaseModel):
    x: int = Field(description="The current X coordinate")
    y: int = Field(description="The current Y coordinate")


def predict_current_coords(
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    size = img.size
    coords = desktop.mouse_coordinates()

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    image.save(img_path)

    coords_prompt = f"<ImageHere> return the mouse coordinates for the GUI image of size [{size[0]}, [{size[1]}] in the form [x, y]"

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=coords_prompt,
                image=img_path,
                history=[],
                do_sample=False,
            )
    print("model response: ", response)

    return {
        "id": str(uuid.uuid4()),
        "image": [img_path],
        "conversations": [
            {
                "from": "user",
                "value": coords_prompt,
            },
            {"from": "assistant", "value": f"[{coords[0]}, [{coords[1]}]"},
        ],
    }


class ExpectationValidation(BaseModel):
    valid: bool = Field(
        description="Whether to expectation is validated by the current environment"
    )


def predict_click(
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    coords = desktop.mouse_coordinates()
    size = img.size

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    image.save(img_path)

    click_prompt = f"<ImageHere> I've provided you with an image of an ubuntu desktop GUI with the size [{size[0]}, [{size[1]}]. In detail, predict the state of the screen after you click the mouse at its current coordinates [{coords[0]}, [{coords[1]}]"

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=click_prompt,
                image=img_path,
                history=[],
                do_sample=False,
            )
    print("model response: ", response)

    print("clicking")
    desktop.click()

    thread = RoleThread()

    print("sleeping for 5 second")
    time.sleep(5)

    b64_img_new = desktop.take_screenshot()
    img_new = b64_to_image(b64_img)
    coords_new = desktop.mouse_coordinates()

    thread.post(
        role="user",
        msg=f"""I've provided you with two images of a desktop GUI, first is an image of the state of the GUI before the mouse was clicked. The second image is an image of the state of the GUI after the mouse was clicked. Please describe the current state of the screen in detail, paying attention to any changes in the screen that may have occured from the click""",
        images=[img, img_new],
    )

    resp = router.chat(thread, model=model)

    return {
        "id": str(uuid.uuid4()),
        "image": [img_path],
        "conversations": [
            {
                "from": "user",
                "value": click_prompt,
            },
            {
                "from": "assistant",
                "value": resp.msg.text,
            },
        ],
    }


class PredictedURL(BaseModel):
    url: str = Field(description="The URL the browser is currently navigated to")


def recognize_url(
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    coords = desktop.mouse_coordinates()

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    image.save(img_path)

    url_prompt = f"<ImageHere> I've provided you with an image of an ubuntu desktop GUI with a chromium browser open. What URL is the browser currently navigated to?"

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=click_prompt,
                image=img_path,
                history=[],
                do_sample=False,
            )
    print("model response: ", response)

    thread = RoleThread()
    thread.post(
        role="user",
        msg=f"""I've provided you with an image of an ubuntu desktop GUI with a chromium browser open. What URL is the browser currently navigated to? Please return just the raw JSON object conforming to the schema {PredictedURL.model_json_schema()}. For example if you see the browser is navigated to the URL 'https://www.airbnb.com/host/homes' then you would return {{"url": 'https://www.airbnb.com/host/homes'}}""",
        images=[img],
    )

    resp = router.chat(thread, model=model)

    if not resp.parsed:
        raise ValueError("no parsed response")

    return resp.parsed.url, {
        "id": str(uuid.uuid4()),
        "image": [img_path],
        "conversations": [
            {
                "from": "user",
                "value": url_prompt,
            },
            {
                "from": "assistant",
                "value": resp.parsed.url,
            },
        ],
    }


class NextTargetSelector(BaseModel):
    current_url: str = Field(
        description="The current URL of the browser e.g. https://google.com"
    )
    x: int = Field(description="The current X coordinate")
    y: int = Field(description="The current Y coordinate")


from typing import Any, Dict


class State(BaseModel):
    image: Image.Image
    description: str


class Memory(BaseModel):
    url: Optional[str] = None
    action: str
    parameters: Dict[str, Any]
    before_state: State
    after_state: State


import json
import time
from typing import List

from .memory import PageMemory, Session, find_memories_by_url


class ClickTargets(BaseModel):
    targets: List[ClickTarget] = Field(description="A list of click targets")


def get_targets(desktop: Desktop, router: Router) -> ClickTargets:
    """Generate targets from a desktop screenshot"""
    print("get_targets()")

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    thread.post(
        role="user",
        msg=f"""I've provided you with an image of a desktop UI. Please describe all the possible targets that you can interact with.
    Please return a JSON object that conforms to the schema {ClickTargets.model_json_schema()}.
    Please be exhaustive, describing all possibilities on the screenshot.
    Please return just raw json. For example {{"targets": [{{"description": "A green button resembling a user", "purpose": "open user settings", "expectation": "user settings will open", "near": "a big red logout button is to the left"}}]}}
    """,
        images=[img],
    )
    resp = router.chat(thread, expect=ClickTargets, namespace="get_targets")

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


def pick_next_target(
    current_url: str,
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> ClickTarget:
    """Pick the next target to explore in the app"""

    thread = RoleThread()

    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    coords_new = desktop.mouse_coordinates()

    session = Session()

    page_memories = find_memories_by_url(current_url, session)

    target = None

    if not page_memories:
        targets = get_targets(desktop, router)

        for tgt in targets:
            mem = PageMemory(
                url=current_url, click_target=json.dumps(tgt), created=time.time()
            )
            session.add(mem)

        session.commit()

        target = targets.targets[0]

    # TODO: validate page memory

    # find any targets that haven't been clicked
    no_result = []
    for memory in page_memories:
        if not memory.result:
            no_result.append(memory)

    if no_result:
        target_model = no_result[0]
        target = target_model.click_target

    else:
        tgts = []

        for mem in page_memories:
            tgts.append(mem.click_target)

        targets = ClickTargets(targets=tgts)
        thread.post(
            role="user",
            msg=f"""I've provided you with an image of a desktop UI. Please describe all the possible targets that you can interact with 
                which are not currently included in our known targets:
                {targets.model_dump_json()}
                
                Please return a JSON object that conforms to the schema {ClickTargets.model_json_schema()}.
                Please only return targets the current list is missing, if there are no new targets return an empty array.
                Please return just raw json. For example {{"targets": [{{"description": "A green button resembling a user", "purpose": "open user settings", "expectation": "user settings will open", "near": "a big red logout button is to the left"}}]}}
                """,
            images=[img],
        )
        resp = router.chat(thread, expect=ClickTargets, model=model)

        if not resp.parsed:
            raise ValueError("no parsed resp")

        if resp.parsed.targets:
            for tgt in resp.parsed.targets:
                mem = PageMemory(
                    url=current_url, click_target=json.dumps(tgt), created=time.time()
                )
                session.add(mem)

            session.commit()

            target = resp.parsed.targets[0]

        else:
            target = tgts[0]

    return target


def move_towards_or_away_from_center(
    x, y, max_width, max_height, factor=0.05, move_away_factor=0.05
):
    """
    Move the point (x, y) a little bit towards the center of the screen,
    or if the point is exactly at the center, move it slightly away.

    Parameters:
    - x (int): The x-coordinate of the point.
    - y (int): The y-coordinate of the point.
    - max_width (int): The maximum width of the screen.
    - max_height (int): The maximum height of the screen.
    - factor (float): The factor determining how much to move towards the center.
                      Default is 0.1, which means 10% closer to the center.
    - move_away_factor (float): The factor determining how much to move away from the center
                                if the point is exactly at the center. Default is 0.1.

    Returns:
    - (new_x, new_y) (tuple): The new coordinates of the point.
    """
    center_x = max_width / 2
    center_y = max_height / 2

    if x == center_x and y == center_y:
        # Move the point slightly away from the center
        new_x = x + (x * move_away_factor)
        new_y = y + (y * move_away_factor)
    else:
        # Calculate the difference between the point and the center
        dx = center_x - x
        dy = center_y - y

        # Move the point closer to the center by the given factor
        new_x = x + dx * factor
        new_y = y + dy * factor

    return [new_x, new_y]


def predict_delta_mouse_coords(
    target: ClickTarget,
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    coords = desktop.mouse_coordinates()
    size = img.size

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    img.save(img_path)

    url_prompt = f"""<ImageHere> I've provided you with an image of an ubuntu desktop GUI, our goal is to click on the target: 
    
    {target.model_dump_json()} 
    
    The screen size is [{size[0]}, [{size[1]}] and the mouse is located at [{coords[0]}, {coords[1]}] Please provide the mouse coordinates of the center of the target in the form [x, y]"""

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer, query=url_prompt, image=img_path, history=[], do_sample=False
            )
    print("model response: ", response)

    def apply_learn(trgt, coords):
        prompt = f"""<ImageHere> I've provided you with an image of an ubuntu desktop GUI, our goal is to click on the target: 
        
        {trgt.model_dump_json()} 
        
        The screen size is [{size[0]}, [{size[1]}] and the mouse is located at [{coords[0]}, {coords[1]}] Please provide the mouse coordinates of the center of the target in the form [x, y]"""

        example = {
            "id": str(uuid.uuid4()),
            "image": [img_path],
            "conversations": [
                {
                    "from": "user",
                    "value": prompt,
                },
                {
                    "from": "assistant",
                    "value": json.dumps(coords),
                },
            ],
        }
        return example

    try:
        coords = json.loads(response)
        return coords, apply_learn
    except Exception as e:
        print("exception parsing coords: ", e)
        return (
            move_towards_or_away_from_center(
                coords[0],
                coords[1],
                size[0],
                size[1],
                factor=0.05,
                move_away_factor=0.05,
            ),
            apply_learn,
        )


def describe_delta_mouse_coords(
    new_coords: List[int],
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    coords = desktop.mouse_coordinates()
    size = img.size

    action_id = str(uuid.uuid4())
    img_path = f"online_img/{action_id}.png"
    img.save(img_path)

    delta_prompt = f"""<ImageHere> I've provided you with an image of an ubuntu desktop GUI, the screen size is [{size[0]}, [{size[1]}] and the mouse is currently located at [{coords[0]}, {coords[1]}]. We are moving the mouse to {new_coords}, please predict what the mouse will be located above in that position. Please be descriptive."""

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            response, _ = model.chat(
                tokenizer,
                query=delta_prompt,
                image=img_path,
                history=[],
                do_sample=False,
            )
    print("model response: ", response)

    def apply_learn(description):
        prompt = f"""<ImageHere> I've provided you with an image of an ubuntu desktop GUI, the screen size is [{size[0]}, [{size[1]}] and the mouse is currently located at [{coords[0]}, {coords[1]}]. We are moving the mouse to {new_coords}, please predict what the mouse cursor will be located over in that position. Please be descriptive."""

        example = {
            "id": str(uuid.uuid4()),
            "image": [img_path],
            "conversations": [
                {
                    "from": "user",
                    "value": prompt,
                },
                {
                    "from": "assistant",
                    "value": description,
                },
            ],
        }
        return example

    return response, apply_learn


def gather_data(
    base_url: str,
    desktop: Desktop,
    router: Router,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> List[dict]:
    data = []

    # choose a direction to move
    print("choosing the next target...")
    next_target = pick_next_target(base_url, desktop, router, tokenizer, model)
    if not next_target:
        raise ValueError("No next target")
    print("next target: ", next_target)

    # predict the change in coords and describe where it will be
    print("predicting delta coords...")
    new_coords, delta_mouse_apply = predict_delta_mouse_coords(
        next_target, desktop, router, tokenizer, model
    )
    print("new coords: ", new_coords)

    print("describing delta mouse coords..")
    description, describe_apply = describe_delta_mouse_coords(
        new_coords, desktop, router, tokenizer, model
    )
    print("delta mouse description")

    print("moving mouse...")
    desktop.move_mouse(new_coords[0], new_coords[1])

    # describe the current location
    print("describing current location")
    current_target, example = describe_location(desktop, router, tokenizer, model)
    data.append(example)
    print("current target: ", current_target)
    print("example: ", example)

    # learn from the changes
    print("generating training examples from environment changes...")
    coords = desktop.mouse_coordinates()
    example = delta_mouse_apply(current_target, list(coords))
    data.append(example)
    print("delta mouse example: ", example)

    example = describe_apply(current_target.description)
    data.append(example)
    print("describe example: ", example)

    # predict the mouse coords
    print("predicting current mouse coords...")
    example = predict_current_coords(desktop, router, tokenizer, model)
    data.append(example)
    print("predicted mouse coords: ", example)

    # predict what will happen when we click
    print("predicting the click...")
    example = predict_click(desktop, router, tokenizer, model)
    data.append(example)
    time.sleep(2)
    print("click example: ", example)

    # predict the current browser URL
    print("recognizing the URL...")
    current_url, example = recognize_url(desktop, router, tokenizer, model)
    data.append(example)
    print("url: ", current_url)
    print("url example: ", example)

    # check if we have left the site
    if base_url not in current_url:
        print("we have left the base url, reopening")
        desktop.open_url(base_url)
        time.sleep(5)

    print("returning data: ", data)
    return data


def make_online_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    train_data: List[dict],
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")
    train_json = {"online": train_data}

    train_dataset = Mix_dataset(
        train_json,
        data_args.batch_size,
        img_size=data_args.img_size,
        hd_num=data_args.hd_num,
        local_rank=local_rank,
    )
    print(str(len(train_dataset)) + " train samples loaded")

    data_collator = DataCollatorForSupervisedDataset()
    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def online_train_single():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = (
        parser.parse_args_into_dataclasses()
    )

    # Set up model
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model.tokenizer = tokenizer

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = DataCollatorForSupervisedDataset()

    # Initialize Trainer
    trainer = DebugTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    vm = Desktop.find(name="vibrant-perlman")
    desktop = Desktop.from_vm(vm[0])

    router = Router.from_env()

    base_url = "https://airbnb.com"

    while True:
        print("train loop...")
        # Gather and process new data
        # gather_data(base_url: str, desktop: Desktop, router: Router, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel)
        new_data = gather_data(base_url, desktop, router, tokenizer, model)
        print("got new data: ", new_data)

        data_module = make_online_supervised_data_module(
            tokenizer=tokenizer, data_args=data_args, train_data=new_data
        )

        trainer = DebugTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=data_module["train_dataset"],
            data_collator=data_module["data_collator"],
        )

        print("training...")
        # Perform incremental update
        trainer.train()

        print("saving model...")
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            bias=lora_args.lora_bias if training_args.use_lora else "none",
        )

        # Perform evaluation if needed
        # if training_args.do_eval:
        #     final_metrics, _ = custom_evaluate(trainer.model, trainer.eval_dataset, tokenizer)
        #     print("Evaluation Metrics:", final_metrics)

        # Pause for a moment before gathering the next piece of data
        # time.sleep(1)
        print("looping again...")


if __name__ == "__main__":
    train()
