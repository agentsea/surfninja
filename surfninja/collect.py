from pydantic import BaseModel, Field


class UserFlow(BaseModel):
    """e.g. A user should be able to search for a room in a city"""

    # Is this just a task? or is there something more to it? its more generic, tasks are created from this
    # How do we define variations?
    pass


class Expectation(BaseModel):
    """e.g. When I click this button, I expect a login screen to appear"""

    pass


class TestPlan(BaseModel):
    """e.g. A test plan is a collection of test cases"""

    uri: str
