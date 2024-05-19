# -*- coding: utf-8 -*-
import os
import time
import json
import argparse
from tqdm import tqdm
import data_process
import optimizers
from evaluate import cal_metrics


def get_dataset(dataset):
    if dataset == 'instruction-induction':
        return data_process.InstructionInsduction
    elif dataset == 'gms8k':
        return data_process.GSM8K
    elif dataset == 'multi_arith':
        return data_process.MultiArith
    elif dataset == 'counterfactual-evaluation':
        return data_process.CFE
    else:
        raise Exception(f'Unsupported dataset: {dataset}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='instruction-induction')       # 可选instruction-induction、gms8k、multi_arith、counterfactual-evaluation
    parser.add_argument('--data_dir', default='data/instruction-induction')
    parser.add_argument('--task', default='antonyms')
    parser.add_argument('--out', default='out/antonyms.txt')
    parser.add_argument('--minibatch_size', default=16, type=int)
    parser.add_argument('--is_shuffle', default=False, type=bool)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--max_threads', default=16, type=int)
    parser.add_argument('--lr', default=5, type=int)
    args = parser.parse_args()
    return args


def validation(optimizer, step, dataset, num):
    valid_metrics = []
    for i in range(num):
        v_metrics, _, _, _ = optimizer.evaluate(step, dataset)
        valid_metrics.append(v_metrics)
    return valid_metrics, sum(valid_metrics) / len(valid_metrics)

if __name__ == '__main__':
    args = get_args()
    config = vars(args)

    dataPrc = get_dataset(args.dataset)(args.data_dir, args.task, args.minibatch_size, args.is_shuffle)
    inst = dataPrc.init_inst()
    train_dataset = dataPrc.get_train_examples()
    valid_dataset = dataPrc.get_dev_examples()
    test_dataset = dataPrc.get_test_examples()

    optimizer = optimizers.ProTeGi(config, inst, args.max_threads)

    # if os.path.exists(args.out):
    #     os.remove(args.out)

    print(config)

    with open(args.out, 'a', encoding='utf-8') as outf:
        outf.write(json.dumps(config) + '\n')
        outf.write(inst + '\n')

    valid_metrics, test_metrics = [], []
    all_nets = []
    # 初始化指令链（网络结构）
    nets = optimizer.init_net()                 #

    print(nets)

    for epoch in tqdm(range(0, config['epochs'] + 1)):
        with open(args.out, 'a', encoding='utf-8') as outf:
            outf.write(f"\n======== epoch {epoch} ========\n")
        if epoch == 0:
            all_nets.append(nets)
            vm_list, vm_avg = validation(optimizer, nets, valid_dataset, 2)
            valid_metrics.append(vm_avg)

            with open(args.out, 'a', encoding='utf-8') as outf:
                outf.write(f'指令链：\n{nets}\n')
                outf.write(f'验证集指标：{vm_list}，平均：{vm_avg}\n')

        else:
            # 测minibatch并计算指标
            train_batch_metrics = []  # 存储一个epoch里每个batch的指标
            for i, train_batch in tqdm(enumerate(train_dataset)):
                print(f'\n第{i + 1}个batch的网络结构（步骤）：\n{nets}')

                # 预测一个minibatch并计算指标
                metrics, texts, labels, preds = optimizer.evaluate(nets, train_batch)
                train_batch_metrics.append(metrics)
                print(f'metrics：{train_batch_metrics[-1]}\n')

                with open(args.out, 'a', encoding='utf-8') as outf:
                    outf.write(f"\n======== batch {i+1} ========\n")
                    outf.write(f'训练集指标：{train_batch_metrics}\n')

                # 分析错误原因（计算损失）
                errors, loss = optimizer.cal_loss(nets, texts, labels, preds)
                print(f'\n错误原因：{loss}')
                if loss:
                    # 制定改进策略（获取梯度）
                    gradients = optimizer.get_gradients(nets, loss)
                    print(f'\n改进策略：{gradients}')
                    # 获取有效的策略
                    learn_gradients = optimizer.cal_lr(nets, gradients, train_batch, metrics)
                    print(f'\n有用的策略：{learn_gradients}')
                    if len(learn_gradients) != 0:
                        learn_gradients = '\n'.join([f'{i + 1}.{g}' for i, g in enumerate(learn_gradients)])
                        _, nets = optimizer.update_net(nets, learn_gradients, '\n'.join(errors))

                    with open(args.out, 'a', encoding='utf-8') as outf:
                        outf.write(f'错误原因：{loss}\n')
                        outf.write(f'改进策略：{gradients}\n')
                        outf.write(f'有用的策略：{learn_gradients}\n')

                    all_nets.append(nets)
                    vm_list, vm_avg = validation(optimizer, nets, valid_dataset, 2)
                    valid_metrics.append(vm_avg)
                else:
                    all_nets.append(nets)
                    valid_metrics.append(valid_metrics[-1])
                try:
                    with open(args.out, 'a', encoding='utf-8') as outf:
                        outf.write(f'指令链：\n{nets}\n')
                        outf.write(f'验证集指标：{vm_list}，平均：{vm_avg}\n')
                except:
                    pass

    # 测试
    best_index = valid_metrics.index(max(valid_metrics))
    best_nets = all_nets[best_index]
    test_metrics = []
    for i in range(1):
        t_metrics, _, _, _ = optimizer.evaluate(best_nets, test_dataset)
        test_metrics.append(t_metrics)
    with open(args.out, 'a', encoding='utf-8') as outf:
        outf.write(f'验证集指标：{valid_metrics}\n')
        outf.write(f'best index:{best_index}\n')
        outf.write(f'{best_nets}\n')
        outf.write(f'测试集指标：{test_metrics}，平均{sum(test_metrics) / len(test_metrics)}\n')

    print("DONE!")

