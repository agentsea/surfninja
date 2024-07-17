import json
import os
from datasets import load_dataset
from PIL import Image
import io

# Load the dataset
ds = load_dataset("agentsea/wave-ui")



example0 = ds['train'][0]
example0


# Create directory to save images
output_image_dir = "waves_img"
os.makedirs(output_image_dir, exist_ok=True)

# Define the path to save the JSONL file
output_file = "train_formated_wavesui.jsonl"

# Helper function to serialize image data and save locally
def save_image_and_get_path(image_data, image_id):
    image_path = os.path.join(output_image_dir, f"image_{image_id}.jpg")
    
    # Convert image to RGB mode if it's RGBA
    if image_data.mode == 'RGBA':
        image_data = image_data.convert('RGB')
    
    image_data.save(image_path)
    return image_path


def label_bounding_box(image_resolution, bounding_box):
    x, y = image_resolution
    x0, y0, x1, y1 = bounding_box

    # Calculate the center of the bounding box
    bx_center_x = (x0 + x1) / 2
    bx_center_y = (y0 + y1) / 2

    # Calculate the center of the image
    img_center_x = x / 2
    img_center_y = y / 2

    # Determine the horizontal position
    if bx_center_x < img_center_x / 2:
        horiz_pos = 'left'
    elif bx_center_x > img_center_x * 1.5:
        horiz_pos = 'right'
    else:
        horiz_pos = 'center'

    # Determine the vertical position
    if bx_center_y < img_center_y / 2:
        vert_pos = 'top'
    elif bx_center_y > img_center_y * 1.5:
        vert_pos = 'bottom'
    else:
        vert_pos = 'center'

    # Combine the horizontal and vertical positions
    if vert_pos == 'center' and horiz_pos == 'center':
        position = 'center'
    else:
        position = f'{vert_pos}-{horiz_pos}'

    return position


label_bounding_box(example0['resolution'], example0['bbox'])


# Prepare dataset in the desired JSON structure
formatted_data = []
for idx, example in enumerate(ds['train']):  # Assuming 'train' split, adjust as per your dataset splits
    # Save image locally and get the path
    if 'image' in example:
        image_data = example['image']  # Assuming 'image' key exists in your dataset
        image_path = save_image_and_get_path(image_data, idx)
        out = {}
        out['id'] = idx
        out['image'] = [image_path]

        resolution = example['resolution']
        description = example['description']
        purpose = example['purpose']
        expectation = example['expectation']

        location = label_bounding_box(example['resolution'], example['bbox'])
        
        # Format conversations
        conversations = [
            {
                "from": "user",
                "value": f"<ImageHere> detect bounding box for <Resolution> {resolution} </Resolution> <Description> {description} </Description> <Location> {location} </Location> <Purpose> {purpose} </Purpose> <Expectation> {expectation} </Expectation>"
            },
            {
                "from": "assistant",
                "value": f"{example['bbox']}"
            }
        ]  # Replace with actual conversations data
        
        out['conversations'] = conversations
        
        # Add formatted example to the list
        formatted_data.append(out)

# Write formatted data to JSONL file
with open(output_file, 'w', encoding='utf-8') as f_out:
    for example in formatted_data:
        json_line = json.dumps(example)
        f_out.write(json_line + '\n')

print(f"Dataset saved to {output_file}")


import json

# Define the input and output file paths
input_file = 'train_formated_wavesui.jsonl'  # Replace with your actual JSONL file path
output_file = 'train_formated_wavesui.json'  # Replace with your desired JSON file path

# Read the JSONL file and load each line as a JSON object
with open(input_file, 'r', encoding='utf-8') as f_in:
    json_array = [json.loads(line) for line in f_in]

# Write the JSON array to the output file
with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(json_array, f_out, indent=4)

print(f"Dataset saved to {output_file}")