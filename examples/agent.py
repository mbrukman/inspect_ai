from typing import Callable, Generic, Optional, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel, Field
from inspect_ai._util.content import Content
from inspect_ai.tool._tool import ToolResult
from inspect_ai.util._store_model import StoreModel

T = TypeVar("T", bound=StoreModel, contravariant=True)

S = TypeVar("S", bound=StoreModel, covariant=True)
P = ParamSpec("P")


class AgentState(Protocol, Generic[S]):
    def __call__(self, namespace: str | None = None) -> S: ...


class Agent(Protocol, Generic[T]):
    async def __call__(
        self,
        state: AgentState[T],
        input: str | list[Content],
    ) -> str | list[Content]: ...


class MyAgentState(StoreModel):
    messages: int = Field(default=0)


class MyAgent(Agent[MyAgentState]):
    async def __call__(
        self,
        state: AgentState[MyAgentState],
        input: str | list[Content],
    ) -> str | list[Content]:
        return str(state().messages)
