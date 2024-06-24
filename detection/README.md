# UI element detection

Agents need to be able to semantically detect UI elements.

## Current state

SOTA MLLMs can perform detection, but they do not perform well in the context of web UIs. Notebook `notebooks/bound.ipynb` validates this hypothesis with large closed models like Gemini Pro 1.5 and GPT-4o and an open small model, PaliGemma. Further work needs to be done to validate this with new releases too, like Claude 3.5 and Florence. However, so far the results are consistent with the hypothesis that models have not been trained on web UIs and therefore do not perform well in this context.

Other vision models, like YOLOv8 can perform detection on web UIs, but, unlike MLLMs, they do not provide a semantic understanding of the elements, and, thus, cannot perform zero-shot detection.

## WebUI

Open datasets, such as the WebUI dataset provide a key element to improve the performance of MLLMs in the context of web UIs. The WebUI dataset is a collection of web UI elements and bounding-box data, among other things. Specifically, the WebUI release provides the "Web-7k-Resampled", a "small, higher quality split for experimentation. Web-7k-Resampled was generated using a class-balancing sampling technique, and we removed screens with possible visual defects (e.g., very small, occluded, or invisible elements)". Nevertheless, the dataset has multiple limitations:

- It does not come out-of-the-box with a image-box tuples. It needs preprocessing for segregation.
- It contains repeated sites.
- It has superfluous boxes/data (at least for this specific purpose).

These limitations are addressed in `scripts/process_webui.py`, and the notebook `notebooks/filter.ipynb`.

The output of each of these steps are saved in Google Cloud Storage, as checkpoints.

## Automated annotations

In order for the WebUI dataset to be useful in the object detection task, good annotations/labels are needed.

WebUI's default annotations are not nearly enough for the task at hand. This implies the necessity of automated annotations. The notebook `notebooks/annotation_test.ipynb` explores this by taking a random sample of 100 boxes from the dataset and annotating them with two different methods:

### OpenAI's GPT-4o model

- GPT-4o model is the current top (?) multimodal model.
- Based on [tokenizer](https://platform.openai.com/tokenizer) and [pricing info](https://openai.com/api/pricing/) the cost per request is ~ $0.00213. At 68k datapoints, the total cost would be roughly $145.
- Advantages
    - Potentially faster.
    - Potentially more accurate/informative.
- Disadvantages
    - Cost is slightly higher.
    - Closed model.

### PaliGemma WidgetCap FT

- The PaliGemma model family is a family of "smol" models (at ~3B parameters).
- The WidgetCap FT is a fine-tuned version of the model on the WidgetCap dataset, for captioning UI widgets.
- Advantages
    - Cost is lower.
    - Open model.
- Disadvantages
    - Potentially slower, depending on how many GPUs we want to use.
    - Potentially less accurate/informative.
- Technical details
    - Model can run inference on a single T4 Nvidia GPU.
    - It supports 4-bit quantization.
    - It supports bfloat16/float16 precision.
    - A 100-datapoint sample takes ~240 seconds on 2 L4 Nvidia GPUs, so running the whole 68k dataset would take ~40 hours with this same setup, which is about $80 in cost (at $1 per GPU per hour).

The results of the annotation test are saved in Google Cloud Storage, as checkpoints.

## Next steps

- Pipelines are ready. We only need to decide on GPT-4o or PaliGemma WidgetCap FT for the automated annotations.
- Run PaliGemma fine-tune on annotated dataset.
- Build a benchmark and compare the performance of top models available.
- Share this with the community!
    - Share the dataset.
    - Share findings.
    - Open PaliGemma fine-tune.
    - Make it available for agents.






