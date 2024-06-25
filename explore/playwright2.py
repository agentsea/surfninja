import sys
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync


def get_clickable_elements_bounding_boxes(url: str) -> Tuple[List[Dict[str, int]], str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Apply stealth techniques
        stealth_sync(page)

        # Navigate to the URL
        print(f"Navigating to {url}")
        page.goto(url)

        bounding_boxes: List[Dict[str, int]] = []

        # Click on the body to ensure the focus is on the page
        print("Clicking on the body element to focus the page")
        page.click("body")

        # Define a set of selectors for clickable elements
        clickable_selectors = [
            "a",
            "button",
            "input[type='button']",
            "input[type='submit']",
            "input[type='reset']",
            "input[type='image']",
            "[role='button']",
            "[role='link']",
            "div[role='menuitem']",
            "div[role='tab']",
            "div[role='treeitem']",
        ]

        # Convert selectors to a single selector string
        clickable_selector = ", ".join(clickable_selectors)
        print(f"Using clickable selectors: {clickable_selector}")

        def get_focused_element_bounding_box() -> bool:
            print("Evaluating the focused element")
            focused_element_handle = page.evaluate_handle("document.activeElement")
            focused_element = focused_element_handle.as_element()
            if not focused_element:
                print("No focused element found")
                return False
            print("Got focused element handle")
            is_clickable = focused_element.evaluate(
                "(element, selector) => element.matches(selector)", clickable_selector
            )
            print(f"Is the focused element clickable? {is_clickable}")
            if is_clickable:
                bounding_box = focused_element.evaluate(
                    """
                    element => {
                        const rect = element.getBoundingClientRect();
                        return {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        };
                    }
                """
                )
                if bounding_box:
                    print(f"Bounding box: {bounding_box}")
                    bounding_boxes.append(bounding_box)
                    return True
            return False

        # Emulate keyboard navigation
        max_tabs = 100  # Increase the number of tabs to capture more elements
        for _ in range(max_tabs):
            print("Pressing Tab key")
            page.keyboard.press("Tab")
            page.wait_for_timeout(
                200
            )  # Increase wait time to ensure the element is focused

            if get_focused_element_bounding_box():
                # Attempt arrow key navigation within widgets if an element is clickable
                for _ in range(3):  # Limit arrow key navigation within a single widget
                    print("Pressing ArrowRight key")
                    page.keyboard.press("ArrowRight")
                    page.wait_for_timeout(200)
                    if get_focused_element_bounding_box():
                        continue

                    print("Pressing ArrowLeft key")
                    page.keyboard.press("ArrowLeft")
                    page.wait_for_timeout(200)
                    if get_focused_element_bounding_box():
                        continue

        # Capture screenshot
        screenshot_path = "screenshot.png"
        print(f"Capturing screenshot and saving to {screenshot_path}")
        page.screenshot(path=screenshot_path)

        browser.close()
        print("Browser closed")
        return bounding_boxes, screenshot_path


def draw_bounding_boxes(image_path: str, bounding_boxes: List[Dict[str, int]]):
    print(f"Drawing bounding boxes on {image_path}")
    with Image.open(image_path) as im:
        draw = ImageDraw.Draw(im)
        for box in bounding_boxes:
            x, y, width, height = box["x"], box["y"], box["width"], box["height"]
            draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
        im.show()  # Display the image
        output_path = "output_with_boxes.png"
        im.save(output_path)  # Save the image with bounding boxes
        print(f"Image with bounding boxes saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <URL>")
        sys.exit(1)

    url = sys.argv[1]  # Get the URL from the command line arguments
    bounding_boxes, screenshot_path = get_clickable_elements_bounding_boxes(url)
    for box in bounding_boxes:
        print(box)
    draw_bounding_boxes(screenshot_path, bounding_boxes)
