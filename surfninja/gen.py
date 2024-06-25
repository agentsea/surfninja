from agentdesk import Desktop
from taskara import Task
from threadmem import RoleThread

from .prompt import ClickTarget, MoveDirection, update_target_in_thread


def generate_alternative_starts(
    task: Task,
    target: ClickTarget,
    thread: RoleThread,
    desktop: Desktop,
    move_direction: MoveDirection,
) -> None:
    """Given a click location, generate alternative starting points for the mouse"""

    starting_coords = desktop.mouse_coordinates()
