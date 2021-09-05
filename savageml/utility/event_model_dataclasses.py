from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, List

from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button


class MouseLocation(Enum):
    X = 'x'
    Y = 'y'


class Scroll(Enum):
    X = 'x'
    Y = 'y'
    dX = 'dx'
    dY = 'dy'


@dataclass(eq=True, frozen=True, repr=True)
class KeyEventData:
    key: Union[Key, KeyCode]
    released: bool = False


@dataclass(eq=True, frozen=True, repr=True)
class MouseEventData:
    location: Optional[MouseLocation] = None
    btn: Optional[Button] = None
    scroll: Optional[Scroll] = None
    released: bool = False


def build_wait_function_OR(event_list: List[Union[MouseEventData, KeyEventData]],
                           events: Union[MouseEventData, KeyEventData, List[Union[MouseEventData, KeyEventData]]]):
    index_list = []
    if not isinstance(events, list):
        events = [events]
    for event in events:
        try:
            index_list.append(event_list.index(event))
        except ValueError:
            pass
    if index_list:
        def wait_function(values):
            for index in index_list:
                if values[0, index] != 0.0:
                    return True
            return False

        return wait_function
    else:
        return None


def build_wait_function_AND(event_list: List[Union[MouseEventData, KeyEventData]],
                            events: Union[MouseEventData, KeyEventData, List[Union[MouseEventData, KeyEventData]]]):
    index_list = []
    if not isinstance(events, list):
        events = [events]

    for event in events:
        try:
            index_list.append(event_list.index(event))
        except ValueError:
            pass
    if index_list:
        def wait_function(values):
            for index in index_list:
                if values[0, index] == 0.0:
                    return False
            return True

        return wait_function
    else:
        return None

