import json
import shutil
import os
import zipfile
import hashlib
from collections import defaultdict

from pathlib import Path
from PIL import Image, ImageDraw

from huggingface_hub import snapshot_download

Path("downloads").mkdir(parents=True, exist_ok=True)

print("=====================================")
print("DOWNLOAD")
print("=====================================")
snapshot_download(repo_id="biglab/webui-7kbal", repo_type="dataset", local_dir="downloads")
print("Downloaded dataset ✅")
os.system("cat downloads/balanced_7k.zip.* > downloads/combined_balanced_7k.zip")
print("Combined files into one zip ✅")
os.system("unzip -q downloads/combined_balanced_7k.zip -d downloads/")
print("Unzipped dataset ✅")
os.system("cd downloads/balanced_7k/ && find . -type f -name '*.gz' -exec gunzip {} +")
print("Unzipped all .gz files ✅")
print("=====================================")

devices = {"default_1280-720"}
base_output_path = Path("downloads/balanced_7k_processed/")
input_path = Path("downloads/balanced_7k/")
labels = False
output_zip_path = Path('downloads/balanced_7k_processed.zip')
draw_boxes = False
print("=====================================")
print("CONFIG")
print("=====================================")
print(f"Using devices: {devices}")
print(f"Output path: {base_output_path}")
print(f"Input path: {input_path}")
print(f"Labels: {labels}")
print(f"Output zip path: {output_zip_path}")
print(f"Draw boxes: {draw_boxes}")
print("=====================================")


def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_hash = hashlib.sha256(image_data).hexdigest()
    return image_hash

def find_duplicates(path):
    hash_to_sites = defaultdict(set)
    site_to_hashes = defaultdict(set)
    
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.endswith('.webp'):
                image_path = os.path.join(subdir, file)
                site_id = os.path.basename(os.path.dirname(subdir))
                image_hash = get_image_hash(image_path)
                hash_to_sites[image_hash].add(site_id)
                site_to_hashes[site_id].add(image_hash)

    duplicates = {image_hash: sites for image_hash, sites in hash_to_sites.items() if len(sites) > 1}

    return duplicates

def delete_duplicates(duplicates, path):
    for _, sites in duplicates.items():
        sites = list(sites)
        _ = sites.pop(0)
        for site_id in sites:
            shutil.rmtree(path / site_id)


print("=====================================")
print("PROCESSING")
print("=====================================")
for device in devices:
    output_path = base_output_path / device
    for site in input_path.iterdir():
        if site.is_dir():
            bbs_dir = output_path / site.stem / "bounding_boxes"
            bbs_dir.mkdir(parents=True, exist_ok=True)
            images = output_path / site.stem / "images"
            images.mkdir(parents=True, exist_ok=True)
            input_img_path = input_path / site.stem / f"{device}-screenshot-full.webp"
            bbs = json.load(open(site / "default_1280-720-bb.json", "r"))
            tags = json.load(open(site / "default_1280-720-class.json", "r"))
            shutil.copy(input_img_path, images / "full-screenshot.webp")
            if draw_boxes:
                base_img = Image.open(input_img_path)
                all_bbs_img = base_img.copy()
                all_bbs_draw = ImageDraw.Draw(all_bbs_img)
            (filtered_keys, bb) = zip(*[(k, v) for k, v in bbs.items() if v is not None])
            for i in range(len(filtered_keys)):
                k = filtered_keys[i]
                single_bb = {"tags": ', '.join(str(v) for _, v in tags.get(str(k), {}).items()), "bounding_box": bb[i]}
                json.dump(single_bb, (bbs_dir / f"{i}.json").open(mode='w'))
                if draw_boxes:
                    x = single_bb[str(k)]["bounding_box"]['x']
                    y = single_bb[str(k)]["bounding_box"]['y']
                    labels = single_bb[str(k)]["tags"]
                    box_coords = (x, y, x + single_bb[str(k)]["bounding_box"]['width'], y + single_bb[str(k)]["bounding_box"]['height'])
                    single_bb_img = base_img.copy()
                    single_bb_draw = ImageDraw.Draw(single_bb_img)
                    single_bb_draw.rectangle(box_coords, outline='red', width=2)
                    if labels:
                        single_bb_draw.text((x, y - 10), labels, fill='red')
                    single_bb_img.save(images / f"{i}.jpeg", format="JPEG")
                    all_bbs_draw.rectangle(box_coords, outline='red', width=2)
            if draw_boxes:
                all_bbs_img.save(images / "all_bounding_boxes.jpeg", format="JPEG")
print("Processed dataset ✅")

print("=====================================")
print("DETECTING DUPLICATES")
print("=====================================")
for device in devices:
    print("Deleting duplicates for device: ", device)
    path = base_output_path / device
    print("Number of sites before removing duplicates: ", len(list(path.iterdir())))
    duplicates = find_duplicates(path)
    delete_duplicates(duplicates, path)
    print("Number of sites after removing duplicates: ", len(list(path.iterdir())))


print("=====================================")
print("ZIPPING")
print("=====================================")
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_output_path):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, base_output_path))
print("Zipped processed dataset ✅")