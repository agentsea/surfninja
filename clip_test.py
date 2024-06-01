import clip
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

# TODO: this doesn't work?

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Load your image
image_path = "beautiful-golden-retriever-dog-lie-260nw-1570104319.jpg"
image = Image.open(image_path)
image_tensor = preprocess(image).unsqueeze(0).to(device)  # type: ignore

# Define the text description of the icon you are looking for
text_description = "dog"
text_tokens = clip.tokenize([text_description]).to(device)

# Generate candidate bounding boxes
num_boxes = 100
min_box_size = 0.1  # Minimum box size as a fraction of the image size
max_box_size = 0.5  # Maximum box size as a fraction of the image size

# Generate random box sizes
box_sizes = torch.rand(num_boxes, 2) * (max_box_size - min_box_size) + min_box_size

# Generate random box positions
box_positions = torch.rand(num_boxes, 2)

# Calculate box coordinates
boxes = torch.zeros(num_boxes, 4)
boxes[:, 0] = (box_positions[:, 0] * (1 - box_sizes[:, 0])) * image.width
boxes[:, 1] = (box_positions[:, 1] * (1 - box_sizes[:, 1])) * image.height
boxes[:, 2] = boxes[:, 0] + box_sizes[:, 0] * image.width
boxes[:, 3] = boxes[:, 1] + box_sizes[:, 1] * image.height

# Clip bounding boxes to image dimensions
boxes[:, 0] = boxes[:, 0].clamp(0, image.width)
boxes[:, 1] = boxes[:, 1].clamp(0, image.height)
boxes[:, 2] = boxes[:, 2].clamp(0, image.width)
boxes[:, 3] = boxes[:, 3].clamp(0, image.height)

# Convert boxes to integer coordinates
boxes = boxes.round().long()

# Crop and preprocess image regions
image_regions = []
for box in boxes:
    x1, y1, x2, y2 = box.tolist()
    region = image.crop((x1, y1, x2, y2))
    region_tensor = preprocess(region).unsqueeze(0).to(device)  # type: ignore
    image_regions.append(region_tensor)

image_regions = torch.cat(image_regions, dim=0)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_regions)
    text_features = model.encode_text(text_tokens)

# Calculate similarity scores and apply softmax
logits_per_image = image_features @ text_features.T
probs = logits_per_image.softmax(dim=0).detach().cpu().numpy()

# Find the top 5 bounding boxes with the highest probabilities
top_indices = np.argsort(probs[:, 0])[-5:]
top_boxes = boxes[top_indices]

# Draw the top 5 bounding boxes on the image
draw = ImageDraw.Draw(image)
for box in top_boxes:
    x1, y1, x2, y2 = box.tolist()
    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)  # type: ignore

# Draw the best bounding box in a different color
best_box = top_boxes[-1]
x1, y1, x2, y2 = best_box.tolist()
draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)  # type: ignore

# Print the best bounding box
print(
    f"Best bounding box for '{text_description}': {best_box.tolist()} with confidence {probs[top_indices[-1], 0]:.2f}"
)

# Save the image with bounding boxes
output_path = "output_image.png"
image.save(output_path)
print(f"Image with bounding boxes saved as '{output_path}'")
