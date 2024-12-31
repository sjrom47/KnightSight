from config import *
import numpy as np


class VisualBoard:
    def __init__(self, state=None):
        self._state = state
        self._temp_state = None

    def set_initial_state(self, state):
        self._state = state
        self._playing = 1

    @property
    def state(self):
        return self._state

    @property
    def playing(self):
        return self._playing

    def make_move(self, pos1, pos2):
        start_ind = np.argmax(
            [
                self._playing * self._state[pos1[0], pos1[1]],
                self._playing * self._state[pos2[0], pos2[1]],
            ]
        )
        end_ind = 1 - start
        start = (pos1, pos2)[start_ind]
        end = (pos1, pos2)[end_ind]
        start_row, start_col = start
        end_row, end_col = end
        self._temp_state[end_row][end_col] = self._state[start_row][start_col]
        self._temp_state[start_row][start_col] = 0

    def confirm_move(self):
        self._state = self._temp_state
        self._temp_state = None
        self._playing = -self._playing

    def replace_piece(self, row, col, piece):
        self._state[row][col] = piece

    def __str__(self):
        output = ""
        for row in self._state:
            output += " ".join([str(i) for i in row]) + "\n"
        return output
