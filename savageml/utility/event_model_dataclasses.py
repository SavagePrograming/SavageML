from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional

from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button


class MouseLocation(Enum):
    X = 'x'
    Y = 'y'


@dataclass(eq=True, frozen=True, repr=True)
class KeyEventData:
    key: Union[Key, KeyCode]
    click: bool = False


@dataclass(eq=True, frozen=True, repr=True)
class MouseEventData:
    btn: Optional[Button]
    location: Optional[MouseLocation]
    click: bool = False
