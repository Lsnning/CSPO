import argparse
import json
import string
from collections import Counter
import re

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains'}


INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(str.maketrans('', '', string.punctuation))

    return prediction


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1


def get_multi_answer_em(prediction, answers):
    for answer in answers:
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_f1(prediction, answers):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)


def get_multi_answer_exact_set(prediction, answers):
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_contains(prediction, answers):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0


def get_weighted_task_score(scored_predictions):
    """Get the task overall score, weighted according to the instructions prediction frequencies."""
    id_to_counter = {}
    id_to_score = {}
    for instruction_id, instruction_data in scored_predictions.items():
        id_to_counter[instruction_id] = instruction_data['prediction_counter']
        id_to_score[instruction_id] = instruction_data['average_score']
    num_instructions = sum(list(id_to_counter.values()))
    weighted_score = 0
    for id_, count in id_to_counter.items():
        weighted_score += (id_to_score[id_] * count) / num_instructions
    return weighted_score


def cal_execution_accuracy(task_name, answers, predictions):
    if task_name == 'sentence_similarity':
        answers = [a[0].split('-')[0].strip() for a in answers]
    task_metric = TASK_TO_METRIC.get(task_name, 'em')
    scores = []
    for i in range(len(predictions)):
        if task_metric == 'f1':
            score = get_multi_answer_f1(prediction=predictions[i], answers=answers[i])
        elif task_metric == 'es':
            score = get_multi_answer_exact_set(prediction=predictions[i], answers=answers[i])
        elif task_metric == 'contains':
            score = get_multi_answer_contains(prediction=predictions[i], answers=answers[i])
        else:  # EM
            score = get_multi_answer_em(prediction=predictions[i], answers=answers[i])
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    return avg_score


def cal_accuracy(answers, predictions):
    scores = []
    for i in range(len(predictions)):
        if eval(answers[i]) == eval(predictions[i]):
            scores.append(1)
        else:
            scores.append(0)
    avg_score = sum(scores) / len(scores)
    return avg_score

def cal_accuracy2(answers, predictions):
    scores = []
    for i in range(len(predictions)):
        if str(answers[i]).lower() == str(predictions[i]).lower():
            scores.append(1)
        else:
            scores.append(0)
    avg_score = sum(scores) / len(scores)
    return avg_score


def cal_metrics(dataset, task_name, answers, predictions):
    if dataset == 'instruction-induction':
        return cal_execution_accuracy(task_name, answers, predictions)
    elif dataset == 'gms8k' or dataset == 'multi_arith':
        return cal_accuracy(answers, predictions)
    elif dataset == 'counterfactual-evaluation':
        return cal_accuracy2(answers, predictions)
    else:
        return

def is_right(dataset, task_name, answer, prediction):
    if dataset == 'instruction-induction':
        task_metric = TASK_TO_METRIC.get(task_name, 'em')
        if task_metric == 'f1':
            score = get_multi_answer_f1(prediction=prediction, answers=answer)
        elif task_metric == 'es':
            score = get_multi_answer_exact_set(prediction=prediction, answers=answer)
        elif task_metric == 'contains':
            score = get_multi_answer_contains(prediction=prediction, answers=answer)
        else:  # EM
            score = get_multi_answer_em(prediction=prediction, answers=answer)
    elif dataset == 'gms8k' or dataset == 'multi_arith':
        score = int(eval(answer) == eval(prediction))
    elif dataset == 'counterfactual-evaluation':
        score = int(str(answer).lower() == str(prediction).lower())
    else:
        score = ''
    return score
