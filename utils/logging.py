import pandas as pd
import os

from utils.eval_result_super import ModelEvaluationResult


class BasicLogEntry(ModelEvaluationResult):
    def __init__(self, question: str, full_answer: str, final_answer: str, correct_answer: str):
        self._question = question
        self._full_answer = full_answer
        self._final_answer = final_answer
        self._correct_answer = correct_answer

    def get_question(self) -> str:
        return self._question

    def get_full_answer(self) -> str:
        return self._full_answer

    def get_final_answer(self) -> str:
        return self._final_answer
    
    def get_correct_answer(self) -> str:
        return self._correct_answer

    def to_dict(self):
        return {
            'question': self.get_question(),
            'full_answer': self.get_full_answer(),
            'final_answer': self.get_final_answer(),
            'correct_answer': self.get_correct_answer()
        }


class StepbackLogEntry(ModelEvaluationResult):
    def __init__(self, principle_question: str, principles: str, question: str, full_answer: str, final_answer: str,
                correct_answer: str):
        self._principle_question = principle_question
        self._principles = principles
        self._question = question
        self._full_answer = full_answer
        self._final_answer = final_answer
        self._correct_answer = correct_answer

    def get_principle_question(self) -> str:
        return self._principle_question
    
    def get_principles(self) -> str:
        return self._principles

    def get_question(self) -> str:
        return self._question
    
    def get_full_answer(self) -> str:
        return self._full_answer

    def get_final_answer(self) -> str:
        return self._final_answer
    
    def get_correct_answer(self) -> str:
        return self._correct_answer

    def to_dict(self):
        return {
            'principle_question': self.get_principle_question(),
            'principles': self.get_principles(),
            'question': self.get_question(),
            'full_answer': self.get_full_answer(),
            'final_answer': self.get_final_answer(),
            'correct_answer': self.get_correct_answer()
        }


def write_log_to_file(logs: [ModelEvaluationResult], model: str, tech: str, folder: str, lower: int, bound):
    # Check for partial run
    if not (lower == 1 and bound == None):
        # encode end of dataset (bound = None) as -1, or return to upper
        if bound == None:
            upper = -1
        else:
            upper = bound + lower
    else:
        upper = -1

    # create folder if needed
    logdir = f'{folder}/logs/{model}'
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    if upper == -1:
        file = f'{logdir}/{model}_{tech}_.csv'
    else:
        file = f'{logdir}/{model}_{tech}_{lower}_{upper}.csv'
    dict_list = [log.to_dict() for log in logs]
    df = pd.DataFrame(dict_list)
    df.to_csv(file, index=False)


def write_general_log_to_file(logs: [dict], model: str, tech: str, folder: str, lower: int, bound):
    # Check for partial run
    if not (lower == 1 and bound == None):
        # encode end of dataset (bound = None) as -1, or return to upper
        if bound == None:
            upper = -1
        else:
            upper = bound + lower

    # create folder if needed
    logdir = f'{folder}/logs/{model}'
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    file = f'{logdir}/{model}_{tech}_{lower}_{upper}.csv'
    df = pd.DataFrame(logs)
    df.to_csv(file, index=False)
