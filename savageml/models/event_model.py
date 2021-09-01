from collections import Iterable
from ctypes import Union
from typing import List

import numpy as np

from savageml.models import BaseModel
import pynput

MOUSE_X = "mouse_x"
MOUSE_Y = "mouse_y"

CLICK_X = "click_x"
CLICK_Y = "click_y"
CLICK = "click"
CLICK_PERSISTANT = "click_persistent"

def create_mouse_listener(event_table, values):
    def on_move(x, y):
        if MOUSE_X in event_table:
            values[event_table[MOUSE_X]] = x
        if MOUSE_Y in event_table:
            values[event_table[MOUSE_Y]] = y

        print('Pointer moved to {0}'.format(
            (x, y)))

    def on_click(x, y, button, pressed):

        if MOUSE_X in event_table:
            values[event_table[MOUSE_X]] = x
        if MOUSE_Y in event_table:
            values[event_table[MOUSE_Y]] = y

        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False

    def on_scroll(x, y, dx, dy):
        print('Scrolled {0}'.format(
            (x, y)))

    return pynput.mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll)
def create_keyboard_event_handlers(event_table, case_sensitive, values):
    pass

class EventModel(BaseModel):
    def __init__(self, event_list: List[str], delay: float = 0.0, case_sensitive=False):
        """Constructor Method"""
        self.event_list = event_list
        self.delay = delay
        event_table = {event: index for index, event in enumerate(event_list)}
        self.out_put = np.zeros((1, len(event_list)))
        self.case_sensitive = case_sensitive

        self.mouse_listener = create_mouse_listener(event_table, self.out_put)
        self.mouse_listener.start()


        pass

    def clone(self):
        """Creates an exact copy of the model, in it's initial state

        Returns
        -------
        BaseModel
            The cloned copy of the class
        """
        instance = self.__class__(**self.get_params())
        return instance

    def fit(self,
            x: Union[np.ndarray, Iterable],
            y: np.ndarray = None
            ) -> None:
        """ All models must have a fit function. The fit function must support two modes:

        In the first mode, `x` and `y` are given as separate inputs, where each is an :class:`np.ndarray` .
        The first index of both arrays must be the batch index.

        In the second mode only `x` is given and it is expected to be an :class:`Iterable` . That iterable should
        produce tuples where the 0th and 1st index are both :class:`np.ndarray` representing x and y respectively

        In either case the fit function has the model learn how to predict `y` from `x`

        Parameters
        ----------
        x: np.ndarray, Iterable
            The input values to the model, or an iterable that produces (x, y, ...) tuples.

        y: np.ndarray, optional
            The expected output values if an iterable is not provided.

        """
        pass

    def predict(self, x: Union[np.ndarray, Iterable]) -> np.ndarray:
        """All models must have a predict function. This function must take an np.ndarray or
         an iterable that produces (x, ...) tuples. The function must then predict y values based on the input.

        Parameters
        ----------
        x: np.ndarray, Iterable
            The input values to the model, or an iterable that produces (x, ...) tuples.

        Returns
        -------
        np.ndarray
            The predicted output

        """
        pass

    def clean(self):
        self.mouse_listener.stop()
        self.mouse_listener.join()
        self.keyboard_listener.stop()
        self.keyboard_listener.join()