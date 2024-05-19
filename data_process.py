import requests
import random
import json
import math
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

class DataProcessor(ABC):
    def __init__(self, data_dir, task_name, batch_size=16, is_shuffle=True):
        self.data_dir = data_dir
        self.task_name = task_name
        self.batch_size = batch_size
        self.shuffle = is_shuffle
        self.iitasks = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff',
                      'first_word_letter', 'informal_to_formal', 'larger_animal', 'letters_list', 'negation',
                      'num_to_verbal', 'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity',
                      'sentiment', 'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de',
                      'translation_en-es', 'translation_en-fr', 'word_in_context']
    def batch_generator(self, data, batch_size=16, shuffle=True):
        """
        用于划分数据集为批次的生成器函数

        参数：
        - data_list: list，包含数据的列表
        - batch_size: int，批次的大小
        - shuffle: bool，是否打乱数据列表中的数据

        返回：
        - 生成器，每次调用返回一个包含批次数据的列表
        """
        if shuffle:
            random.seed(42)
            random.shuffle(data)

        num_batches = math.ceil(len(data) / batch_size)
        data_batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            data_batches.append(dict(data[start_idx:end_idx]))
        return data_batches

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_dev_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass


class InstructionInsduction(DataProcessor):
    # 获取初始指令
    def init_inst(self):
        init_inst_dict = {
            'antonyms': 'Write a word that means the opposite of the input word.',
            'negation': 'Negate the input sentence',
            'orthography_starts_with': 'Extract the words starting with a given letter from the input sentence.',
            'rhymes': 'Write a word that rhymes with the input word.',
            'second_word_letter': 'Extract the second letter of the input word.',
            'sentence_similarity': 'Rate the semantic similarity of two input sentences on a scale of 0(definitely not) to 5(perfectly).',
            'sentiment': 'Determine whether a movie review is positive or negative.',
            'synonyms': 'Write a word with a similar meaning to the input word.',
            'taxonomy_animal': 'Write all the animals that appear in the given list.',
            'translation_en-de': 'Translate the word into German.',
            'translation_en-es': 'Translate the word into Spanish.',
            'translation_en-fr': 'Translate the word into French.',
            'word_in_context': 'Determine whether an input word has the same meaning in the two input sentences.The outputs "same" indicate that the word has the same meaning in both sentences, while the outputs "not the same" indicate that the word has different meanings in the two sentences.'
            }
        return init_inst_dict[self.task_name]

    def get_train_examples(self):
        if self.task_name not in self.iitasks:
            raise ValueError("Task not found: %s" % (self.task_name))
        with open(f'{self.data_dir}/raw/execute/{self.task_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = data['examples']
        num_examples = len(examples)

        inputs, outputs = [], []
        for i in range(num_examples):
            data = examples[str(i + 1)]
            if self.task_name == 'cause_and_effect':
                cause, effect = data['cause'], data['effect']
                # Pick an order randomly
                if random.random() < 0.5:
                    input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
                else:
                    input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
                output_ = [cause]
            elif self.task_name == 'common_concept':
                items = data['items']
                # Make comma separated list of items
                input_ = ', '.join(items[:-1])
                output_ = data['all_common_concepts']
            elif self.task_name == 'rhymes':
                input_, output_ = data['input'], data['other_rhymes']
            elif 'translation' in self.task_name:
                input_, output_ = data['input'], data['possible_translations']
            else:
                input_, output_ = data['input'], [data['output']]
            inputs.append(input_)
            outputs.append(output_)
        data = list(zip(inputs, outputs))
        batch_data = self.batch_generator(data, self.batch_size, self.shuffle)
        return batch_data

    def get_dev_examples(self):
        if self.task_name not in self.iitasks:
            raise ValueError("Task not found: %s" % (self.task_name))
        with open(f'{self.data_dir}/raw/execute/{self.task_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = data['examples']
        num_examples = len(examples)

        all_data = dict()
        for i in range(num_examples):
            data = examples[str(i + 1)]
            if self.task_name == 'cause_and_effect':
                cause, effect = data['cause'], data['effect']
                # Pick an order randomly
                if random.random() < 0.5:
                    input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
                else:
                    input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
                output_ = [cause]
            elif self.task_name == 'common_concept':
                items = data['items']
                # Make comma separated list of items
                input_ = ', '.join(items[:-1])
                output_ = data['all_common_concepts']
            elif self.task_name == 'rhymes':
                input_, output_ = data['input'], data['other_rhymes']
            elif 'translation' in self.task_name:
                input_, output_ = data['input'], data['possible_translations']
            else:
                input_, output_ = data['input'], [data['output']]
            all_data[input_] = output_
        return all_data

    def get_test_examples(self):
        if self.task_name not in self.iitasks:
            raise ValueError("Task not found: %s" % (self.task_name))
        with open(f'{self.data_dir}/raw/execute/{self.task_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = data['examples']
        num_examples = len(examples)

        all_data = dict()
        for i in range(num_examples):
            data = examples[str(i + 1)]
            if self.task_name == 'cause_and_effect':
                cause, effect = data['cause'], data['effect']
                # Pick an order randomly
                if random.random() < 0.5:
                    input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
                else:
                    input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
                output_ = [cause]
            elif self.task_name == 'common_concept':
                items = data['items']
                # Make comma separated list of items
                input_ = ', '.join(items[:-1])
                output_ = data['all_common_concepts']
            elif self.task_name == 'rhymes':
                input_, output_ = data['input'], data['other_rhymes']
            elif 'translation' in self.task_name:
                input_, output_ = data['input'], data['possible_translations']
            else:
                input_, output_ = data['input'], [data['output']]
            all_data[input_] = output_
        return all_data


class GSM8K(DataProcessor):
    # 获取初始指令
    def init_inst(self):
        return "read and solve the mathematical question."

    def get_train_examples(self):
        questions = []
        answers = []
        with open(f'{self.data_dir}/train.jsonl', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line:
                    qa = json.loads(line)
                    questions.append(qa['question'])
                    answers.append(qa['answer'].split("\n####")[-1].strip())
        data = list(zip(questions, answers))
        random.seed(42)
        batch_data = self.batch_generator(random.sample(data,100), self.batch_size, self.shuffle)
        return batch_data

    def get_dev_examples(self):
        dev_data = dict()
        with open(f'{self.data_dir}/train.jsonl', 'r', encoding='utf-8') as f:
            for line in f.readlines()[-100:]:
                if line:
                    qa = json.loads(line)
                    question = qa['question']
                    answer = qa['answer'].split("\n####")[-1].strip()
                    dev_data[question] = answer
        return dev_data

    def get_test_examples(self):
        test_data = dict()
        with open(f'{self.data_dir}/test.jsonl', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line:
                    qa = json.loads(line)
                    question = qa['question']
                    answer = qa['answer'].split("\n####")[-1].strip()
                    test_data[question] = answer
        return test_data


class MultiArith(DataProcessor):
    # 获取初始指令
    def init_inst(self):
        return "read and solve the mathematical question."

    def get_train_examples(self):
        questions = []
        answers = []
        with open(f'{self.data_dir}/MultiArith.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i in range(100):
            questions.append(data[i]["sQuestion"])
            answers.append(str(data[i]["lSolutions"][0]))
        train_data = list(zip(questions, answers))
        train_data = self.batch_generator(train_data, self.batch_size, self.shuffle)
        return train_data

    def get_dev_examples(self):
        dev_data = dict()
        with open(f'{self.data_dir}/MultiArith.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i in range(100,200):
            question = data[i]["sQuestion"]
            answer = str(data[i]["lSolutions"][0])
            dev_data[question] = answer
        return dev_data

    def get_test_examples(self):
        test_data = dict()
        with open(f'{self.data_dir}/MultiArith.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i in range(200, len(data)):
            question = data[i]["sQuestion"]
            answer = str(data[i]["lSolutions"][0])
            test_data[question] = answer
        return test_data


class CFE(DataProcessor):
    # 获取初始指令
    def init_inst(self):
        init_inst_dict = {'base8': 'Add the two numbers in base-8.',
                          'base9': 'Add the two numbers in base-9.',
                          'base11': 'Add the two numbers in base-11.',
                          'base16': 'Add the two numbers in base-16.',
                          'chess_counter_factual': 'Swap the initial positions of the knight and bishop in chess, then check if the first four moves in the input are legal. If each move is legal, output "legal"; otherwise, output "illegal."',
                          'osv': 'The structure of the input sentence is object-subject-verb. Find the main verb and the main subject in the sentence.',
                          'ovs': 'The structure of the input sentence is object-verb-subject. Find the main verb and the main subject in the sentence.',
                          'sov': 'The structure of the input sentence is subject-object-verb. Find the main verb and the main subject in the sentence.',
                          'vos': 'The structure of the input sentence is verb-object-subject. Find the main verb and the main subject in the sentence.',
                          'vso': 'The structure of the input sentence is verb-subject-object. Find the main verb and the main subject in the sentence.'
                          }
        return init_inst_dict[self.task_name]

    def get_labels(self,questions):
        labels = []
        if self.task_name[:4] == 'base':
            base = int(self.task_name[4:])
            for q in questions:
                lhs, rhs = q.split("+")
                lhs_base10 = int(lhs, base)
                rhs_base10 = int(rhs, base)
                sum_base10 = lhs_base10 + rhs_base10
                labels.append(np.base_repr(sum_base10, base))
        return labels

    def get_train_examples(self):
        if self.task_name[:4] == 'base':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)
            questions = data[:100]
            answers = self.get_labels(questions)
        elif set(self.task_name) == set('svo'):
            data = pd.read_csv(f'{self.data_dir}/{self.task_name}/deps_train.csv')
            data = data.sample(n=300,random_state=42)[:100]
            questions = list(data['original_sent'])
            answers = list(map(list,zip(data['main_subj'],data['main_verb'])))
        elif self.task_name[:5] == 'chess':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)[:100]
            questions = [line.split('*')[0].strip() for line in data]
            answers = [line.split('*')[1].strip() for line in data]
        else:
            raise ValueError(f"{self.task_name}是无效的任务")
        train_data = list(zip(questions, answers))
        train_data = self.batch_generator(train_data, self.batch_size, self.shuffle)
        return train_data

    def get_dev_examples(self):
        if self.task_name[:4] == 'base':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)
            questions = data[100:200]
            answers = self.get_labels(questions)
        elif set(self.task_name) == set('svo'):
            data = pd.read_csv(f'{self.data_dir}/{self.task_name}/deps_train.csv')
            data = data.sample(n=300,random_state=42)[100:200]
            questions = list(data['original_sent'])
            answers = list(map(list,zip(data['main_subj'],data['main_verb'])))
        elif self.task_name[:5] == 'chess':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)[100:200]
            questions = [line.split('*')[0].strip() for line in data]
            answers = [line.split('*')[1].strip() for line in data]
        else:
            raise ValueError(f"{self.task_name}是无效的任务")
        dev_data = dict(zip(questions, answers))
        return dev_data

    def get_test_examples(self):
        if self.task_name[:4] == 'base':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)
            questions = data[200:300]
            answers = self.get_labels(questions)
        elif set(self.task_name) == set('svo'):
            data = pd.read_csv(f'{self.data_dir}/{self.task_name}/deps_train.csv')
            data = data.sample(n=300,random_state=42)[200: 300]
            questions = list(data['original_sent'])
            answers = list(map(list,zip(data['main_subj'],data['main_verb'])))
        elif self.task_name[:5] == 'chess':
            with open(f'{self.data_dir}/{self.task_name}.txt', 'r', encoding='utf-8') as f:
                data = f.readlines()
            data = list(set(data))
            random.seed(42)
            data = random.sample(data, 300)[200:300]
            questions = [line.split('*')[0].strip() for line in data]
            answers = [line.split('*')[1].strip() for line in data]
        else:
            raise ValueError(f"{self.task_name}是无效的任务")
        test_data = dict(zip(questions, answers))
        return test_data


