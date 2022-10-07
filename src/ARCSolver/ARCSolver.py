from helpers.task import Task
from ARCSolver.core import Core
from ARCSolver.hypothesis import Hypothesis

class ARCSolver:
    def __init__(self, core: Core, n_study_epochs=100, n_sessions_per_epoch=10, max_fails_per_session=10):
        self.core = core
        self.n_study_epochs = n_study_epochs
        self.n_sessions_per_epoch = n_sessions_per_epoch
        self.max_fails_per_session = max_fails_per_session
    
    def __split_into_sessions(self, tasks, n):
        s = len(tasks) // n
        sessions = [tasks[s*i : s*(i+1)] for i in list(range(n)) ]
        sessions[-1] += tasks[sum(len(s) for s in sessions):]
        return sessions

    def do_study_session(self, session_tasks=[Task]):
        n_fails = 0
        rem_tasks = session_tasks
        while(len(rem_tasks) > 0 and n_fails < self.max_fails_per_session):
            n_solved = 0
            for task in rem_tasks:
                hypothesis = self.core.study_train_task(task)
                if(hypothesis.test(task.get_tests(), task.get_solutions())):
                    rem_tasks.remove(task)
                    n_solved += 1
            n_fails = n_fails+1 if n_solved == 0 else 0

        score = (len(session_tasks) - len(rem_tasks)) / len(session_tasks)
        return score

    def do_study_epoch(self, tasks=[Task]):
        score = 0.0
        sessions = self.__split_into_sessions(tasks, self.n_sessions_per_epoch)
        for session in sessions:
            score += self.do_study_session(session)
            self.core.do_sleep()
        return score / self.n_sessions_per_epoch
    
    def do_study(self, training_tasks=[Task], evaluation_tasks=[Task]):
        score = 0.0
        scores = []
        n_epochs = 0
        study_tasks = training_tasks + evaluation_tasks

        while(n_epochs < self.n_study_epochs and score < 1.0):
            n_epochs += 1
            score = self.do_study_epoch(study_tasks)
            scores.append(score)
        
        return scores


    def solve_task(self, task:Task):
        hypothesis = self.core.solve_test_task(task)
        return [hypothesis(t) for t in task.get_tests()]

    def take_test(self, test_tasks=[Task]):
        return {task.id : self.solve_task(task) for task in test_tasks}

