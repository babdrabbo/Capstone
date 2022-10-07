from typing import List, Tuple
from abc import ABC, abstractmethod
from helpers.task import Task
from priors.primitives import Grid
from ARCSolver.hypothesis import Hypothesis

class Core(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def study_train_task(self, task: Task) -> Hypothesis:
        pass

    @abstractmethod
    def solve_test_task(self, task: Task) -> Hypothesis:
        pass

    @abstractmethod
    def do_sleep(self):
        ''' Reorganizes KB while asleep. '''
        pass
