
from random import randrange
from collections.abc import Iterable

def iterable(x):
    return x if isinstance(x, Iterable) else [x]

class Hypothesis():
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, input):
        return self.solver(input)

    def test(self, inputs=[], outputs=[]):
        try:
            return all((self(i) == o).all() for (i, o) in zip(iterable(inputs), iterable(outputs)))
        except:
            return False
