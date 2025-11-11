from typing import List


class SDR:
    def __init__(self, dimensions: list[int]):
        self.size = 1
        self.dimensions_ = dimensions
        for dim in dimensions:
            self.size *= dim
        self.sparse = []

        # I think we need these for the other types of encoders like RDSE
        self.dense = []
        self.coordinates = []
        self.dense_valid = False
        self.sparse_invalid = False
        self.coordinates_valid = False

    def zero(self):
        self.sparse = []

    def getSparse(self):
        return self.sparse

    def setSparse(self, dimensions: List[int]):
        self.sparse = dimensions

    def setDense(self, value):
        assert len(value) == self.size
        self.dense, value = value, self.dense
        self.set_dense_inplace()

    def set_dense_inplace(self):
        assert len(self.dense) == self.size
        self.clear()
        self.dense_valid = True
        self.sparse = [i for i, v in enumerate(self.dense) if v != 0]
        self.do_callbacks()

    def clear(self):
        self.dense_valid = False
        self.sparse_valid = False
        self.coordinates_valid = False

    def do_callbacks(self):
        pass

    def getDense(self):
        return self.dense
