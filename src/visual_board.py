from config import *


class VisualBoard:
    def __init__(self, state):
        self._state = state

    @property
    def state(self):
        return self._state

    def make_move(self, start, end):
        start_row, start_col = start
        end_row, end_col = end
        self._state[end_row][end_col] = self._state[start_row][start_col]
        self._state[start_row][start_col] = 0

    def __str__(self):
        output = ""
        for row in self._state:
            output += " ".join([str(i) for i in row]) + "\n"
        return output
