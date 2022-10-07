import os


def get_all_files(folder):
    all_files = []
    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            all_files.append(f'{dirname}/{filename}')
    return all_files

def flatten_pred(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def write_out(solutions: dict, out_file):
    with open(out_file, 'w+') as file:
        file.write('output_id,output\n')
        for task_id in solutions.keys():
            for test_id, preds in enumerate(solutions[task_id]):
                file.write(
                    f'{task_id}_{test_id},{" ".join(map(flatten_pred, preds))}\n')
