import logging
import os

from agentdesk.device import Desktop
from mllm import Router
from rich.console import Console
from rich.json import JSON
from taskara import Task
from toolfuse import Tool, action

from .img import image_to_b64
from .prompt import (
    ClickTarget,
    apply_move,
    describe_location,
    det_cursor_type,
    get_move_direction,
    is_finished,
)

router = Router.from_env()
console = Console()

logger = logging.getLogger(__name__)
logger.setLevel(int(os.getenv("LOG_LEVEL", logging.DEBUG)))


class DesktopWithSemMouse(Tool):
    """A desktop replaces mouse click actions with semantic description rather than coordinates"""

    def __init__(
        self, task: Task, desktop: Desktop, data_path: str = "./.data"
    ) -> None:
        """
        Initialize and open a URL in the application.

        Args:
            task: Agent task. Defaults to None.
            desktop: Desktop instance to wrap.
            data_path (str, optional): Path to data. Defaults to "./.data".
        """
        super().__init__(wraps=desktop)
        self.desktop = desktop
        self.task = task

    @action
    def click_object(
        self,
        description: str,
        location: str,
        purpose: str,
        expectation: str,
        type: str,
    ) -> None:
        """Click on an object on the screen

        Args:
            description (str): A detailed description of the object, for example
                "a round dark blue icon with the text 'Home'", please be as descriptive as possible
            location (str): The location of the object on the screen, for example "top-right"
            purpose (str): The purpose for clicking on the object e.g. "to log the user in"
            expectation (str): Expectation of the user after clicking on the object e.g. "for a login screen to appear"
            type (str): Type of click, can be 'single' for a single click or
                'double' for a double click. If you need to launch an application from the desktop choose 'double'
        """
        if type != "single" and type != "double":
            raise ValueError("type must be'single' or 'double'")

        logging.debug(
            f"clicking icon with description '{description}' and purpose '{purpose}'"
        )
        max_steps = 10

        target = ClickTarget(
            description=description,
            location=location,
            purpose=purpose,
            expectation=expectation,
        )
        self.task.post_message(
            role="assistant",
            msg=f"Click target '{target.model_dump()}'",
            thread="debug",
        )

        for step in range(max_steps):
            print("\n---- checking if task is finished...")
            cursor_type = det_cursor_type(self.desktop)
            self.task.post_message(
                role="assistant",
                msg=f"Step {step} cursor type '{cursor_type}'",
                thread="debug",
            )

            print("cursor type: ", cursor_type.type)
            if cursor_type.type != "default":
                check_goal = is_finished(self.desktop, target)
                self.task.post_message(
                    role="assistant",
                    msg=f"Step {step} check goal '{check_goal.model_dump()}'",
                    thread="debug",
                )

                if check_goal.done:
                    print("task is done")
                    if type == "single":
                        self.desktop.click()
                    elif type == "double":
                        self.desktop.double_click()
                    return

                print("task is not finished but cursor is not default")
                # TODO: this should be async, take an image and calc offline
                hindsight_target = describe_location(self.desktop)
                self.task.post_message(
                    role="assistant",
                    msg=f"Step {step} extra hindsight target '{hindsight_target.model_dump()}'",
                    thread="debug",
                )
                # TODO: save as prompt
                print("created hindsight target: ", hindsight_target.model_dump())

            print("\n---- step: ", step)
            direct = get_move_direction(self.desktop, target)
            self.task.post_message(
                role="assistant",
                msg=f"Step {step} move direction '{direct.model_dump()}'",
                thread="debug",
            )

            print("\n---- move direction: ", direct.model_dump())

            new_screen, new_cursor = apply_move(self.desktop, direct)
            self.task.post_message(
                role="assistant",
                msg=f"Step {step} new screens",
                thread="debug",
                images=[
                    image_to_b64(new_screen),
                    image_to_b64(new_cursor),
                ],
            )
