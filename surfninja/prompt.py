from typing import List, Optional, Tuple

import pytesseract
from agentdesk import Desktop
from mllm import RoleThread, Router
from PIL import Image
from pydantic import BaseModel, Field

from surfninja.img import b64_to_image, crop_box_around, image_to_b64, upscale_image

router = Router.from_env()


class ClickTarget(BaseModel):
    """A target which the mouse could be moved to and clicked"""

    description: str = Field(
        description="A long description of the target e.g. A round blue button with the text 'login'"
    )
    location: str = Field(
        description="A general location of the target e.g. top-right, center, bottom-left"
    )
    purpose: str = Field(
        description="A general purpose of the target e.g. 'log the user in' or 'search for a product'"
    )
    expectation: str = Field(
        description="An expectation on what will happen when you click this target e.g. 'A login screen will appear'"
    )


class ClickTargets(BaseModel):
    targets: List[ClickTarget] = Field(description="A list of click targets")


def describe_location(desktop: Desktop) -> ClickTarget:
    """Describe the current location of the mouse"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    coords = desktop.mouse_coordinates()
    cropped = crop_box_around(img, coords[0], coords[1])

    thread.post(
        role="user",
        msg=f"""I'm going to provide you with two images. The first is a picture of a desktop UI, 
    the second is a cropped portion of the first image containing just a 100x100 portion focusing on where the mouse cursor is.
    Please describe what the mouse cursor as a JSON object conforming to the schema {ClickTarget.model_json_schema()}.
    Please return just raw json. For example if you see the mouse above the chromium icon then 
    you would return {{"is_clickable": true, "description": "A blue chromium icon with the text 'chromium' beneath it", "location": "top-right"}}.
    """,
        images=[image_to_b64(img), image_to_b64(cropped)],
    )

    resp = router.chat(thread, expect=ClickTarget)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


def get_targets(desktop: Desktop) -> ClickTargets:
    """Generate targets from a desktop screenshot"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    thread.post(
        role="user",
        msg=f"""I've provided you with an image of a desktop UI. Please describe all the possible targets that you can interact with.
    Please return a JSON object that conforms to the schema {ClickTargets.model_json_schema()}.
    Please be exhaustive, describing all possibilities on the screenshot.
    Please return just raw json. For example {{"targets": [{{"description": "A green button resembling a user", "location": "top-left", "purpose": "open user settings"}}]}}
    """,
        images=[image_to_b64(img)],
    )
    resp = router.chat(thread, expect=ClickTargets)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


class MoveDirection(BaseModel):
    current_location: str = Field(
        description="A description of the current location of the mouse cursor e.g. 'The cursor is currently over a red button in the bottom right of the image'"
    )
    reason: str = Field(
        description="Why the move was made e.g. 'The mouse cursor is in the center of the image but the target is in the top left, I need to move up and to the left'"
    )
    x: int = Field(
        description="Amount to move in the x direction. Positive values move right, negative values move left. 1 is equal to 1 pixel."
    )
    y: int = Field(
        description="Amount to move in the y direction. Positive values move down, negative values move up. 1 is equal to 1 pixel."
    )


def get_move_direction(desktop: Desktop, target: ClickTarget) -> MoveDirection:
    """Generate the next direction to move the mouse (Δx, Δy)"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    coords = desktop.mouse_coordinates()
    cropped = crop_box_around(img, coords[0], coords[1], 100)

    upscaled = upscale_image(cropped, 4)

    thread.post(
        role="user",
        msg=f"""I've provided you with two images: a screenshot of a desktop UI, and a cropped 200x200 image of the current mouse location. 
    Your goal is to navigate to '{target.description}' located in '{target.location}'. The screen size is {img.size} and the current coordinates are {coords}. 
    Please tell me which direction to move the mouse to get there. Please return a JSON object which conforms to the schema {MoveDirection.model_json_schema()}.
    Please return raw json. For example, if I want to move 12 pixels to the left, and 3 pixels up, I would return: 
    {{"reason": "The mouse is slightly below the current object and a bit to the right. I need to move the mouse up and to the left", "x": -12, "y": -3}}. You must move the mouse, 
    either 'x' or 'y' must be non-zero. The very tip of the cursor must directly over the center your desired target, if unsure, move the mouse slightly.
    YOU MUST MOVE THE MOUSE, it has already been determined that you are not in the correct location, double check that you are directly over the target, not just near it.
    The cursor will likely change to a pointer if you are over it.
    """,
        images=[image_to_b64(img), image_to_b64(upscaled)],
    )
    img.save("./.run/screenshot_move.png")
    cropped.save("./.run/cropped_move.png")
    resp = router.chat(thread, expect=MoveDirection)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


def apply_move(
    desktop: Desktop, direction: MoveDirection
) -> Tuple[Image.Image, Image.Image]:
    """Apply a mouse movement to the desktop"""

    current_coords = desktop.mouse_coordinates()
    print("current_cords: ", current_coords)

    # Calculate new absolute mouse coordinates
    new_x = current_coords[0] + direction.x
    new_y = current_coords[1] + direction.y

    print("new: ", new_x, new_y)

    if new_x == 0 and new_y == 0:
        # Bugs happen at (0, 0)
        new_x = 1
        new_y = 1

    # Move the mouse to the new coordinates
    desktop.move_mouse(x=new_x, y=new_y)

    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    coords = desktop.mouse_coordinates()
    cropped = crop_box_around(img, coords[0], coords[1])
    print("new_coords: ", coords)

    return img, cropped


class CursorType(BaseModel):
    type: str = Field(description="Can be 'default', 'text', or 'pointer'")


def det_cursor_type(desktop: Desktop) -> CursorType:
    """Detect the cursor type"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    coords = desktop.mouse_coordinates()
    cropped = crop_box_around(img, coords[0], coords[1], padding=30)

    cropped.save("./.run/cursor.png")

    composite = Image.open("./assets/cursor_composite_image.jpg")

    thread.post(
        role="user",
        msg=f"""I've provided you with two images; first is an image of a mouse cursor and the second is an image 
        displaying the different types of cursors and their names. Please return what type of cursor you see.
        Please return a json object which conforms to the schema {CursorType.model_json_schema()}.
        Please return just raw json. For example if the cursor looks like a standard pointer return {{"type": "default"}}
    """,
        images=[image_to_b64(cropped), image_to_b64(composite)],
    )
    resp = router.chat(thread, expect=CursorType)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


class CheckGoal(BaseModel):
    target: str = Field(
        description="Description of the click target in your own words e.g. 'blue_button'"
    )
    current_location: str = Field(
        description="A description of the current location of the mouse cursor e.g. 'The mouse curesor is currently over a blue button in the bottom-left of the image'"
    )
    reason: str = Field(
        description="Reasoning as to whether the cursor is over the correct location e.g. 'The cursor is over a blue button in the bottom-left but needs to be over a red button in the top-right, task is not complete'"
    )
    done: bool = Field(description="Whether the cursor is over the correct location")


def is_finished(desktop: Desktop, target: ClickTarget) -> CheckGoal:
    """Check if the target has been reached"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    coords = desktop.mouse_coordinates()
    cropped = crop_box_around(img, coords[0], coords[1], 100)

    upscaled = upscale_image(cropped, 4)

    img.save("./.run/is_finished.png")
    cropped.save("./.run/is_finished_cropped.png")

    thread.post(
        role="user",
        msg=f"""I've provided you with two images: a screenshot of a desktop UI, and a cropped 200x200 image of the current mouse location. 
    Your goal is to navigate to '{target.description}' located in '{target.location}' with the purpose of '{target.purpose}'. The screen size is {img.size} and the current coordinates are {coords}. 
    Please tell me if we have achieved that goal. Please return your response as a JSON object which conforms to the schema {CheckGoal.model_json_schema()}.
    Please return raw json. If the goal is achieved the cursor should be directly over the target and should be a pointer, then return {{"done": true}}
    """,
        images=[image_to_b64(img), image_to_b64(upscaled)],
    )
    resp = router.chat(thread, expect=CheckGoal)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed


class Coordinate(BaseModel):
    x: int
    y: int


def get_likely_targets(desktop: Desktop) -> List[Coordinate]:
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    filtered_data = data[(data.conf > 60) & (data.text.notna()) & (data.text != "")]

    out = []

    for index, row in filtered_data.iterrows():
        (x, y, w, h) = (row["left"], row["top"], row["width"], row["height"])
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        out.append(Coordinate(x=center_x, y=center_y))

    return out


class CurrentURL(BaseModel):
    url: str = Field(..., description="The current URL of the browser")


def current_url(desktop: Desktop) -> str:
    """Find the current URL of the browser"""

    thread = RoleThread()
    b64_img = desktop.take_screenshot()
    img = b64_to_image(b64_img)

    thread.post(
        role="user",
        msg=f"""I've provided you with a screenshot of a desktop UI, please return the current url of the browser.
    Please return your response as a JSON object which conforms to the schema {CurrentURL.model_json_schema()}.
    Please return raw json. If the current URL is https://airbnb.com, then return {{"url": "https://airbnb.com"}}
    """,
        images=[image_to_b64(img)],
    )
    resp = router.chat(thread, expect=CurrentURL)

    if not resp.parsed:
        raise ValueError("No click area found")

    return resp.parsed.url
