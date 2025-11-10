"""cell class for HTM model layer"""


from .segment import Segment


class Cell:
    """Cell class for HTM model layer -- represents a single cell in a column"""

    _segments: list[Segment]

    def __init__(self):
        self._active_state = False
        self._predictive_state = False
        self._segments = []

    def __repr__(self) -> str:
        return (f"Cell(active_state={self._active_state}, "
                f"predictive_state={self._predictive_state}, "
                f"segments={self._segments})")
