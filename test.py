import asyncio

from PIL import Image, ImageDraw
from playwright.async_api import async_playwright


async def main(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle")

        # Sleep to allow additional time for the page to fully load
        await asyncio.sleep(10)  # Wait for 10 seconds

        selectors = [
            "a",
            "button",
            "input[type=button]",
            "input[type=submit]",
            "[onclick]",
        ]
        bounding_boxes = []
        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for element in elements:
                try:
                    if await element.is_visible():
                        await element.scroll_into_view_if_needed(timeout=5000)
                        box = await element.bounding_box()
                        if box:
                            bounding_boxes.append(box)
                    else:
                        print(
                            f"Element not visible, skipped: {await element.inner_html()}"
                        )
                except Exception as e:
                    print(f"Failed to process element {selector}: {str(e)}")

        # Take screenshot with bounding boxes
        screenshot_path = "screenshot_with_boxes.png"
        await page.screenshot(path=screenshot_path)
        image = Image.open(screenshot_path)
        draw = ImageDraw.Draw(image)
        for box in bounding_boxes:
            draw.rectangle(
                [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
                outline="red",
                width=2,
            )
        image.save(screenshot_path)
        image.show()

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main("https://www.google.com/"))
