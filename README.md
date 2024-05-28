# CSPO
This is the official repo for "CSPO: Enhancing Large Language Model Performance through Chain-Structured Prompt Optimization". CSPO is a novel method designed to systematically enhance the performance of large language models (LLMs) by gradually improving prompt structures.CSPO decomposes tasks into a series of ordered steps, or instruction chains, and employs an iterative optimization process akin to forward and backward propagation.

<p align="center">
<img src="./images/opt_process.jpg" alt="Expert-level Prompting" title="The CSPO prompt optimization process is illustrated using the "Antonym" task as an example"/>
</p>

## Quick Start
```
python main.py --dataset instruction-induction --data_dir data/instruction-induction --task antonyms --out out/antonyms.txt --lr 5
```

This will run the antonyms subtask in the instruction induction dataset, get the optimal tips for that task, and the intermediate results will be written to the specified output file out/antonyms.txt.

For usage instructions. Some of the arguments include:

* `--dataset`: Tasks name, such as instruction-induction, gms8k, multi_arith, counterfactual-evaluation.
* `--data_dir`: Directory where the task data resides.
* `--task`: Subtask name, such as antonyms, first_word_letter, etc.
* `--out`: Output file name.
* `--lr`: Learning rate.
* `...`: Various other parameters related to optimization.
