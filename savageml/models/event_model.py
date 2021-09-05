from collections import Iterable
from typing import List, Callable

import numpy as np
import pynput

from savageml.models import BaseModel
from savageml.utility.event_model_dataclasses import *
from savageml.utility.generic_functions import wait_until


class EventModel(BaseModel):
    def __init__(self, event_list: List[Union[MouseEventData, KeyEventData]],
                 wait_function:Callable=None, wait_timeout=10.0, wait_period = 0.25, **kwargs):
        """Constructor Method"""
        super().__init__(**kwargs)
        self.event_list = event_list
        self.wait_function = wait_function
        self.wait_timeout = wait_timeout
        self.wait_period = wait_period

        mouse_event_table = {event: index for index, event in enumerate(event_list) if
                             isinstance(event, MouseEventData)}
        keyboard_event_table = {event: index for index, event in enumerate(event_list) if
                                isinstance(event, KeyEventData)}

        self.out_put = np.zeros((1, len(event_list)))

        self.mouse_listener = create_mouse_listener(mouse_event_table, self.out_put)
        self.mouse_listener.start()

        self.keyboard_listener = create_keyboard_listener(keyboard_event_table, self.out_put)
        self.keyboard_listener.start()

    def predict(self, x: Union[np.ndarray, Iterable]) -> np.ndarray:
        """

        """
        if self.wait_function is not None:
            wait_until(self.wait_function, self.wait_timeout, self.wait_period, self.out_put)

        output = np.repeat(self.out_put, x.shape[0], 0)
        self.out_put[...] = 0.0

        return output

    def clean(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()
        self.keyboard_listener.stop()
        self.keyboard_listener.join()


def create_mouse_listener(event_table, values):
    def on_move(x, y):
        mouse_x = MouseEventData(location=MouseLocation.X)
        if mouse_x in event_table:
            values[0, event_table[mouse_x]] = x

        mouse_y = MouseEventData(location=MouseLocation.Y)
        if mouse_y in event_table:
            values[0, event_table[mouse_y]] = y

    def on_click(x, y, button, pressed):
        mouse_btn = MouseEventData(btn=button, released=not pressed)
        if mouse_btn in event_table:
            values[0, event_table[mouse_btn]] += 1.0

        mouse_btn_x = MouseEventData(btn=button, location=MouseLocation.X, released=not pressed)
        if mouse_btn_x in event_table:
            values[0, event_table[mouse_btn_x]] = x

        mouse_btn_y = MouseEventData(btn=button, location=MouseLocation.Y, released=not pressed)
        if mouse_btn_y in event_table:
            values[0, event_table[mouse_btn_y]] = y

    def on_scroll(x, y, dx, dy):

        scroll_x = MouseEventData(scroll=Scroll.X)
        if scroll_x in event_table:
            values[0, event_table[scroll_x]] = x

        scroll_y = MouseEventData(scroll=Scroll.Y)
        if scroll_y in event_table:
            values[0, event_table[scroll_y]] = y

        scroll_dx = MouseEventData(scroll=Scroll.dX)
        if scroll_dx in event_table:
            values[0, event_table[scroll_dx]] = dx

        scroll_dy = MouseEventData(scroll=Scroll.dY)
        if scroll_dy in event_table:
            values[0, event_table[scroll_dy]] = dy

    return pynput.mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll)


def create_keyboard_listener(event_table, values):
    def on_press(key):
        keyEvent = KeyEventData(key=key)
        if keyEvent in event_table:
            values[0, event_table[keyEvent]] += 1.0

    def on_release(key):
        keyEvent = KeyEventData(key=key, released=True)
        if keyEvent in event_table:
            values[0, event_table[keyEvent]] += 1.0

    listener = pynput.keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    return listener
