import json
import numpy as np
from priors.primitives import Grid

class Task():
    def __init__(self, json_file):
        self.file = json_file
        self.task = json.load(open(json_file))
        self.id   = json_file.split('/')[-1].split('.')[0]
    
    def get_examples(self):
        return [(Grid(np.array(self.task['train'][i]['input'])), Grid(np.array(self.task['train'][i]['output'])))
            for i in range(len(self.task['train']))]
    
    def get_tests(self):
        return [Grid(np.array(self.task['test'][i]['input']))
            for i in range(len(self.task['test']))]

    def get_solutions(self):
        return [Grid(np.array(self.task['test'][i]['output']))
            for i in range(len(self.task['test']))]
