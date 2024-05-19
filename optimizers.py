# -*- coding: utf-8 -*-
import re
import json
from abc import ABC
from tqdm import tqdm
import concurrent.futures
import utils
from evaluate import normalize_prediction, cal_metrics, is_right


class PromptOptimizer(ABC):
    def __init__(self, args, inst, max_threads=1):
        self.opt = args
        self.inst = inst
        self.max_threads = max_threads
        self.inputOrquestion = 'Question' if (self.opt["dataset"] == 'gms8k' or self.opt["dataset"] == 'multi_arith' ) else 'Input'


class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients"""

    def _sample_error_str(self, texts, answers, preds):
        """ Sample error strings from the given texts, labels, and preds"""
        if self.opt["task"] == 'sentence_similarity':
            answers = [a[0].split('-')[0].strip() for a in answers]
        error_idxs = []
        for i, (l, p) in enumerate(zip(answers, preds)):
            if is_right(self.opt["dataset"], self.opt["task"], l, p) != 1:
                error_idxs.append(i)
        error_texts = [texts[i] for i in error_idxs]
        error_answers = [answers[i] for i in error_idxs]
        error_preds = [preds[i] for i in error_idxs]
        error_strings = []
        for t, l, p in zip(error_texts, error_answers, error_preds):
            error_string = f'{self.inputOrquestion}: \"{t.strip()}\"\nOutput: {p.strip()}\nAnswer: {l}'
            error_strings.append(error_string)
        return error_strings

    def parse_res(self, text, key):
        """ Parse the returned json format to list."""
        if key == 'Output':
            match = re.search(r'({.*})', text, re.DOTALL)
        else:
            match = re.search(r'({[^{}]*})', text, re.DOTALL)
        if match:
            json_pred = match.group(1)
            json_pred = json.loads(json_pred)
            pred = json_pred[key]
        else:
            raise ("No match found!")
        if key == 'Output' and ((self.opt["task"] == 'gms8k') or (self.opt["task"] == 'multi_arith')):
            pred = re.search(r"(-?\d+(\.\d+)?)", pred).group(1)
        return pred

    def parse_prediction(self, prediction_list):
        """ Parse the main part of the prediction"""
        prediction_list = [str(p) for p in prediction_list]
        parsed_prediction_list = []
        for pred in prediction_list:
            if pred.lower().startswith('text') or pred.lower().startswith('answer') or pred.lower().startswith(
                    'input') or pred.lower().startswith('output') or pred.lower().startswith('question'):
                parsed_prediction_list.append(pred.split(":", 1)[-1].strip())
            elif pred[0].isdigit():
                parsed_prediction_list.append(pred.split(".", 1)[-1].strip())
            else:
                parsed_prediction_list.append(pred)
        return parsed_prediction_list

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        pattern = re.compile(f'{start_tag}(.*?){end_tag}', re.DOTALL)
        texts = pattern.findall(text)
        if not texts:
            texts = [text]
        finall_texts = []
        for t in texts:
            t_ = re.split(r'\d+\.', t)[1:]
            if len(t_) == 0:
                finall_texts.append(t)
            else:
                finall_texts.extend(t_)
        return finall_texts

    def init_net(self, n=1):
        """ Get initial "neural network" based on the instruction."""
        initialization_prompt = f"""
        You have a task to {self.inst}.
        List the steps to perform the task in order.
        
        Please format the steps as follows in JSON:
        {{
            "Steps": [
                "Step 1: do something",
                ...
            ]
        }}
        """
        # This prompt is designed to instruct a language model to complete a specific task.
        initialization_prompt = '\n'.join([line.lstrip() for line in initialization_prompt.split('\n')])
        while True:
            try:
                init_steps = utils.chatgpt(initialization_prompt, n=n)[0]
                init_steps = self.parse_res(init_steps, 'Steps')
                break
            except Exception as e:
                print(e)
        return '\n'.join([f'{s}' for s in init_steps])

    def prediction(self, steps, text):
        """ Predict a text."""
        output_context = f'''["output main subject", "output main subject"]''' if set(self.opt['task']) == set('svo') else f"""Answer of {self.inputOrquestion}"""
        pred_prompt = f"""
        You have a task to {self.inst}.
        
        {self.inputOrquestion}:
        {text}
        
        According to the following steps in order to {self.inputOrquestion} to perform the task:
        {steps}

        # Output format as follows in JSON
        {{
            "Perform process": ["Description of the result of performing each step"],
            "Output":"{output_context}"
        }}
        """
        pred_prompt = '\n'.join([line.lstrip() for line in pred_prompt.split('\n')])
        while True:
            try:
                pred = utils.chatgpt(pred_prompt, n=1, temperature=self.opt['temperature'])[0]
                pred = self.parse_res(pred, 'Output')
                break
            except Exception as e:
                print(e)
        return str(pred)

    def evaluate(self, steps, exs):
        """ Predict some texts and evaluate Predicted results. """
        outputs = list(exs.values())
        inputs = list(exs.keys())
        # texts_string = '\n'.join([f'{self.textOrquestion}: {text}' for i, text in enumerate(inputs)])
        preds = []
        for input in tqdm(inputs, total=len(inputs), desc='prediction...'):
            while True:
                try:
                    pred = self.prediction(steps, input)
                    preds.append(pred)
                    break
                except Exception as e:
                    print(e)
        metrics = cal_metrics(self.opt["dataset"], self.opt["task"], outputs, preds)
        return metrics, inputs, outputs, preds

    def _cal_loss(self, steps, error_string, n=1):
        """ Get "loss" for a detailed set of sequential steps based on a error string."""
        loss_prompt = f"""
        You have a task to {self.inst}.
        
        {error_string.split('Output')[0]}
        
        You perform this task on the {self.inputOrquestion} in the following sequence of steps:
        {steps}
        
        The output is: {error_string.split('Output')[1].split('Answer')[0]}
        But the real answer should be: {error_string.split('Answer')[1]}
        
        Analyze the reasons for incorrect output when the steps are followed.
        Please format the reason (or reasons) as follows in JSON:
        {{
            "Reasons": [
                "Description of reason 1",
                ...
            ]
        }}
        """
        loss_prompt = '\n'.join([line.lstrip() for line in loss_prompt.split('\n')])
        while True:
            try:
                res = utils.chatgpt(loss_prompt, n=n)[0]
                wrong_reason = self.parse_res(res, 'Reasons')
                break
            except Exception as e:
                print(e)
        return error_string, wrong_reason

    def cal_loss(self, steps, texts, answers, preds):
        """Get the reason of each wrong text according to the predicted result"""
        reasonTowrong = {}
        error_texts = self._sample_error_str(texts, answers, preds)
        if len(error_texts) != 0:
            for error_text in tqdm(error_texts, total=len(error_texts), desc='getting loss'):
                error_text, wrong_reason = self._cal_loss(steps, error_text, 1)
                for r in wrong_reason:
                    reasonTowrong[r] = error_text
        return error_texts, reasonTowrong

    def _get_gradients(self, steps, error_string, reasons, n=1):
        """ Get "gradients" base on a wrong reason."""
        gradient_prompt = f"""
        You have a task to {self.inst}.
        
        {error_string.split('Output')[0]}
        
        You perform this task on the {self.inputOrquestion} in the following sequence of steps:
        {steps}
        
        The output is: {error_string.split('Output')[1].split('Answer')[0]}
        But the real answer should be: {error_string.split('Answer')[1]}
        
        The reason for following the steps but the output error are:
        {reasons}
        
        Based on the above information, for each reason, analyze which step caused the error.And finally propose a strategy (or strategies) to improve the step.

        Please format the strategy (or strategies) as follows in JSON:
        {{  
            "Analytical process": "which step caused the error and think how to improve the step",
            "Strategies": [
                "Description of strategy 1",
            ]
        }}
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        while True:
            try:
                res = utils.chatgpt(gradient_prompt, n=n)[0]
                suggestions = self.parse_res(res, 'Strategies')
                break
            except Exception as e:
                print(e)
        return suggestions

    def get_gradients(self, steps, reasonTowrong):
        suggestionsTowrong = {}
        for error_reason in tqdm(reasonTowrong.keys(), total=len(reasonTowrong), desc='getting gradients'):
            suggestions = self._get_gradients(steps, reasonTowrong[error_reason], error_reason, 1)
            for s in suggestions:
                suggestionsTowrong[s] = reasonTowrong[error_reason]
        return suggestionsTowrong

    def update_net(self, steps, suggestions, n=1):
        update_prompt = f"""
        You have a task to {self.inst}.
        
        You list the steps to perform the task in order:
        {steps}
        
        But there are drawbacks to the steps. Combine the following improvement strategies into the steps:
        {suggestions}

        Please format the refined steps as follows in JSON:
        {{
           "Combine all improvement strategies": "Strategies after combination",
            "Modified steps": [
                "Step 1: do something",
                ...
            ]
        }}  
        """
        update_prompt = '\n'.join([line.lstrip() for line in update_prompt.split('\n')])
        while True:
            try:
                modified_steps = utils.chatgpt(update_prompt, n=n)[0]
                modified_steps = self.parse_res(modified_steps, 'Modified steps')
                break
            except Exception as e:
                print(e)
        return suggestions, '\n'.join([f'{s}' for s in modified_steps])

    def cal_lr(self, steps, sggTowrong, exs, original_metrics):
        sggTonet = {}
        for suggestion, error_string in tqdm(sggTowrong.items(), total=len(sggTowrong), desc='getting new nets'):
            suggestion, modified_steps = self.update_net(steps, suggestion, error_string, 1)
            sggTonet[suggestion] = modified_steps
        useful_suggestions = {}
        for suggestion, modified_steps in tqdm(sggTonet.items(), total=len(sggTonet), desc='evaluating suggestions'):
            metrics, _, _, _ = self.evaluate(modified_steps, exs)
            print("metrics:", metrics)
            print("original_metrics:", original_metrics)
            if metrics > original_metrics:
                useful_suggestions[suggestion] = metrics

        if len(useful_suggestions) >= self.opt['lr']:
            useful_suggestions = dict(sorted(useful_suggestions.items(), key=lambda item: item[1], reverse=True)[:self.opt['lr']])

        return list(useful_suggestions.keys())
