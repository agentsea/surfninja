import os
import json
from tkinter import *
from PIL import Image, ImageTk, ImageDraw

# Define the directory where the data is stored
base_dir = 'downloads/default_1280-720/'
state_file = 'downloads/annotation_state.json'

# Create the main window
root = Tk()

# Set up the canvas for image display
canvas = Canvas(root, width=1280, height=720)
canvas.pack()

# Function to load and display an image with a specific bounding box
def display_image(site_id, json_file):
    image_path = os.path.join(base_dir, site_id, 'images', 'full-screenshot.webp')
    pil_image = Image.open(image_path)
    draw = ImageDraw.Draw(pil_image)

    # Load and draw the specified bounding box
    with open(os.path.join(base_dir, site_id, 'bounding_boxes', json_file), 'r') as file:
        data = json.load(file)
        bbox = data['bounding_box']
        draw.rectangle([(bbox['x'], bbox['y']), (bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])], outline="red")

    img = ImageTk.PhotoImage(pil_image)
    canvas.image = img
    canvas.create_image(0, 0, image=img, anchor='nw')

# Handle key press events
def on_key_press(event):
    if event.keysym == 'Up':
        vote = 'up'
    elif event.keysym == 'Down':
        vote = 'down'
    elif event.keysym == 'd':
        print(f"Skipping remaining boxes in {current_site_id} and moving to next site.")
        next_site()
        return
    else:
        return
    print(f"Vote {vote} recorded for {current_site_id} on box {json_files[0]}")
    next_bbox()

# Function to move to the next bounding box or site
def next_bbox():
    global json_files, current_site_id
    json_files.pop(0)
    if json_files:
        display_image(current_site_id, json_files[0])
    else:
        next_site()

# Function to move to the next site
def next_site():
    global current_site_id, sites, json_files
    if sites:
        current_site_id = sites.pop(0)
        json_files = sorted(os.listdir(os.path.join(base_dir, current_site_id, 'bounding_boxes')))
        next_bbox()
    else:
        print("No more sites to review.")
        root.destroy()

# Function to save the current state
def save_state():
    state = {
        'sites': sites,
        'current_site_id': current_site_id,
        'json_files': json_files
    }
    with open(state_file, 'w') as file:
        json.dump(state, file)

# Function to load the saved state
def load_state():
    global sites, current_site_id, json_files
    if os.path.exists(state_file):
        with open(state_file, 'r') as file:
            state = json.load(file)
            sites = state['sites']
            current_site_id = state['current_site_id']
            json_files = state['json_files']
    else:
        sites = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        current_site_id = None
        json_files = []

# Bind key events
root.bind('<KeyPress-Up>', on_key_press)
root.bind('<KeyPress-Down>', on_key_press)
root.bind('<KeyPress-d>', on_key_press)  # Binding the 'd' key

# Load the state when the interface is opened
load_state()

# Start with the first site if there is no current site
if current_site_id is None and sites:
    next_site()
elif current_site_id is not None:
    # Display the JSON file that was open when the interface was closed
    if json_files:
        display_image(current_site_id, json_files[0])
    else:
        next_site()

# Save the state when the interface is closed
def on_closing():
    save_state()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the GUI
root.mainloop()