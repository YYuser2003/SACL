# -*- coding: utf-8 -*-
"""
重构后的 run_train_bert_me.py，提高可读性。

主要变更：
- 参数解析移至 get_args()
- 初始化步骤（种子、设备、数据加载器、模型、损失函数、优化器、对抗训练器）
  分解为辅助函数
- 主要的训练/测试流程移至 main() 以提高顶层清晰度
- 保留详细注释；保持原始行为
"""

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import MELDRobertaCometDataset
from model import DialogueCRN
from loss import FocalLoss

from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


# -------------------------
# 辅助函数：种子、采样器、加载器
# -------------------------
def seed_everything(seed: int = 2021):
    """设置随机种子以确保可重复性。"""
    print("随机种子:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 以速度为代价提高可重复性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid_ratio: float = 0.1):
    """根据数据集和验证比例返回训练/验证 SubsetRandomSampler。"""
    size = len(trainset)
    idx = list(range(size))
    split = int(valid_ratio * size)
    np.random.shuffle(idx)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_bert_loaders(path: str, batch_size: int = 32, classify: str = 'emotion',
                          num_workers: int = 0, pin_memory: bool = False):
    """
    使用 MELDRobertaCometDataset（预提取的 RoBERTa 特征）构建 MELD 的 DataLoader。
    返回：train_loader, valid_loader, test_loader
    """
    trainset = MELDRobertaCometDataset(path, 'train', classify)
    validset = MELDRobertaCometDataset(path, 'valid', classify)
    testset = MELDRobertaCometDataset(path, 'test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


# -------------------------
# 训练/评估循环
# -------------------------
def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, feature_type='text', target_names=None,
                        tensorboard=False, contrast_hidden_flag=False, contrast_weight=0.0, contrast_weight2=0.0,
                        adversary_flag=False, adv_trainer=None, at_method=None, at_rate=0.0, situ_rate=1.0, speaker_rate=0.0, loss_f2=None,
                        eval_cluster_flag=False, gradient_accumulation_steps=2, writer=None):
    """
    单周期的训练/评估循环。

    返回：
      avg_loss, avg_accuracy (%), avg_fscore (%), all_matrix (报告列表), [labels, preds, preds2]
    """
    assert not train_flag or optimizer is not None
    losses, preds, labels, masks = [], [], [], []
    preds2 = []  # 用于聚类评估

    model.train() if train_flag else model.eval()

    if train_flag:
        optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        # 梯度累积控制
        grad_acc_flag = step > 0 and ((step % gradient_accumulation_steps == 0) or step == len(dataloader) - 1)

        # 解包批次数据 (r1, qmask, umask, label)
        r1, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]

        # 计算每个对话的序列长度（umask 表示有效位置）
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # 前向传播
        log_prob, log_prob2 = model(r1, qmask, seq_lengths)

        # 将标签/掩码展平为话语级别
        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        umask2 = torch.cat([umask[j][:seq_lengths[j]] for j in range(len(umask))])

        # 主分类损失
        loss = loss_f(log_prob, label)

        # 监督对比学习（可选）
        if contrast_weight > 0 and loss_f2 is not None:
            # 选择对比学习的嵌入：隐藏层 (log_prob2) 或 logits (log_prob)
            cl_input = log_prob2 if contrast_hidden_flag else log_prob
            cl_loss = loss_f2(cl_input, label)
            # 原始实现乘以 len(label)（求和缩放）
            print("w: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight, loss.item(), len(label) * cl_loss.item()))
            loss = loss + cl_loss * len(label) * contrast_weight

        if eval_cluster_flag:
            preds2.append(log_prob.cpu().detach().numpy())

        preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        masks.append(umask2.view(-1).cpu().numpy())
        losses.append(loss.item())

        # ---------------------------
        # 训练：反向传播 +（可选）对抗步骤 + 优化器步骤
        # ---------------------------
        if train_flag:
            loss.backward(retain_graph=False)

            if adversary_flag and adv_trainer is not None:
                rand_rate = random.uniform(0, 1)
                at_flag = rand_rate <= at_rate
                if at_flag:
                    # 选择使用哪个对抗训练器（情境/说话人/两者）
                    if rand_rate <= at_rate * situ_rate:
                        adv_trainer_l = adv_trainer[0]  # 情境对抗
                    elif rand_rate <= at_rate * (situ_rate + speaker_rate):
                        adv_trainer_l = adv_trainer[1]  # 说话人对抗
                    else:
                        adv_trainer_l = adv_trainer[2]  # 两者都对抗

                    print("# at_flag: {}, rand_rate: {}, situ_rate:{}, speaker_rate:{}".format(at_flag, rand_rate, situ_rate, speaker_rate))

                    if at_method == "fgm":
                        adv_trainer_l.backup_grad()  # 备份梯度
                        adv_trainer_l.attack()  # 添加扰动
                        model.zero_grad()  # 清空梯度
                        # 对扰动后的模型进行前向传播
                        outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                        loss_at = loss_f(outputs_at, label)

                        if contrast_weight2 > 0 and loss_f2 is not None:
                            cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                            print("at: w: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                            loss_at = loss_at + cl_loss_at * len(label) * contrast_weight2

                        loss_at.backward(retain_graph=False)
                        adv_trainer_l.restore_grad()  # 恢复并合并梯度
                        adv_trainer_l.restore()  # 恢复参数

                    elif at_method == "pgd":
                        adv_trainer_l.backup_grad()
                        steps_for_at = 3  # PGD 步数
                        for t in range(steps_for_at):
                            adv_trainer_l.attack(is_first_attack=(t == 0))
                            model.zero_grad()
                            outputs_at, log_prob2_at = model(r1, qmask, seq_lengths)
                            loss_at = loss_f(outputs_at, label)
                            if contrast_weight2 > 0 and loss_f2 is not None:
                                cl_loss_at = loss_f2(log_prob2_at if contrast_hidden_flag else outputs_at, label)
                                print("at: w: {}, ce_loss: {}, cl_loss: {}".format(contrast_weight2, loss_at.item(), len(label) * cl_loss_at.item()))
                                loss_at = loss_at + cl_loss_at * len(label) * contrast_weight2
                            loss_at.backward()  # 累积对抗梯度
                        adv_trainer_l.restore_grad()
                        adv_trainer_l.restore()

            # TensorBoard 梯度记录（可选）
            if tensorboard and writer is not None:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)

            # 达到累积边界时执行优化器步骤
            if grad_acc_flag:
                optimizer.step()
                optimizer.zero_grad()

    # ---------------------------
    # 指标聚合
    # ---------------------------
    if not preds:
        return float('nan'), float('nan'), float('nan'), [], []

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    all_matrix = []
    all_matrix.append(metrics.classification_report(labels, preds, target_names=target_names if target_names else None, sample_weight=masks, digits=4))
    all_matrix.append(["ACC"])
    for i in range(len(target_names)):
        all_matrix[-1].append("{}: {:.4f}".format(target_names[i], accuracy_score(labels[labels == i], preds[labels == i])))

    if eval_cluster_flag:
        preds2 = np.array(np.concatenate(preds2))
    else:
        preds2 = []

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds, preds2]


# -------------------------
# 工具函数构建器
# -------------------------
def build_model_and_move(device_flag, base_model, base_layer, input_size, hidden_size, n_speakers, n_classes, dropout, reason_steps):
    """实例化 DialogueCRN 并在需要时移动到 GPU。"""
    model = DialogueCRN(base_model=base_model,
                        base_layer=base_layer,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        n_speakers=n_speakers,
                        n_classes=n_classes,
                        dropout=dropout,
                        cuda_flag=device_flag,
                        reason_steps=reason_steps)
    if device_flag:
        model.cuda()
    return model


def build_losses_and_contrast(args, class_weights):
    """构建分类损失和（可选的）对比损失。"""
    contrast_hidden_flag = args.scl_hidden_flag
    contrast_temperature = args.scl_t
    contrast_weight = args.scl_w
    contrast_weight2 = args.scl_w2

    if args.loss == 'FocalLoss':
        loss_f = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None, size_average=False)
        loss_f2 = None
        if contrast_weight > 0 or contrast_weight2 > 0:
            from pytorch_metric_learning.distances import DotProductSimilarity
            from pytorch_metric_learning.losses import NTXentLoss
            loss_f2 = NTXentLoss(temperature=contrast_temperature, distance=DotProductSimilarity())
    else:
        loss_f = nn.NLLLoss(class_weights if args.class_weight else None)
        loss_f2 = None

    return loss_f, loss_f2, contrast_hidden_flag, contrast_weight, contrast_weight2


def build_adv_trainer_if_needed(args, model):
    """根据参数构建对抗训练器（FGM/PGD）并返回。"""
    adversary_flag = args.adversary_flag
    if not adversary_flag:
        return None

    at_method = 'fgm'
    # 定义要扰动的参数名称
    emb_names = [
        'rnn.weight_ih_l0', 'rnn.bias_ih_l0',
        "rnn.weight_ih_l0_reverse", 'rnn.bias_ih_l0_reverse',
    ]
    emb_names2 = [
        "rnn_parties.weight_ih_l0", 'rnn_parties.bias_ih_l0',
        "rnn_parties.weight_ih_l0_reverse", 'rnn_parties.bias_ih_l0_reverse',
    ]
    emb_names3 = emb_names + emb_names2

    from at_training import FGM, PGD

    if at_method == "fgm":
        adv_trainer1 = FGM(model, epsilon=args.at_epsilon, emb_names=emb_names)
        adv_trainer2 = FGM(model, epsilon=args.at_epsilon, emb_names=emb_names2)
        adv_trainer3 = FGM(model, epsilon=args.at_epsilon, emb_names=emb_names3)
        return [adv_trainer1, adv_trainer2, adv_trainer3]
    else:
        # PGD 示例（单个训练器）
        return PGD(model, epsilon=args.at_epsilon, alpha=0.1, emb_names=emb_names)


# -------------------------
# 参数解析
# -------------------------
def get_args():
    """解析命令行参数并返回命名空间。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', type=str, default='train', help='可选状态: train/test/test_attack')
    parser.add_argument('--feature_type', type=str, default='text', help='特征类型, multi/text/audio')
    parser.add_argument('--data_dir', type=str, default='../data/meld/meld_features_roberta.pkl', help='数据集目录: MELD_features_raw.pkl')
    parser.add_argument('--output_dir', type=str, default='../outputs/meld/sacl_lstm_meld', help='保存模型目录')
    parser.add_argument('--load_model_state_dir', type=str, default='../sacl_lstm_best_models/meld/4/loss_sacl-lstm.pkl', help='加载模型状态目录')
    parser.add_argument('--base_model', default='LSTM', help='基础模型, LSTM/GRU/Linear')
    parser.add_argument('--base_layer', type=int, default=1, help='基础模型层数, 1/2')
    parser.add_argument('--epochs', type=int, default=1, metavar='E', help='训练周期数')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='批次大小')
    parser.add_argument('--use_valid_flag', action='store_true', default=False, help='使用验证集')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='学习率: 0.0005')
    parser.add_argument('--l2', type=float, default=0.0002, metavar='L2', help='L2 正则化权重')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout 率')
    parser.add_argument('--step_s', type=int, default=0, help='情境级推理步数,3')
    parser.add_argument('--step_p', type=int, default=1, help='说话人级推理步数,0')
    parser.add_argument('--gamma', type=float, default=1, help='gamma 0/0.5/1/2')
    parser.add_argument('--scl_hidden_flag', action='store_true', default=False, help='在隐藏层还是输出层计算对比损失')
    parser.add_argument('--scl_t', type=float, default=0.1, help='对比学习温度 0.07/0.1/0.5/1')
    parser.add_argument('--scl_w', type=float, default=0.1, help='对比损失权重 0.01/0.1/1')
    parser.add_argument('--scl_w2', type=float, default=0.05, help='对抗对比损失权重 0.01/0.1/1')
    parser.add_argument('--adversary_flag', action='store_true', default=True, help='是否使用对抗训练')
    parser.add_argument('--at_rate', type=float, default=1.0, help='对抗训练触发概率 0.1/0.5/1')
    parser.add_argument('--situ_rate', type=float, default=0.1, help='情境对抗比例 0.1/0.5/1')
    parser.add_argument('--speaker_rate', type=float, default=0.1, help='说话人对抗比例 0.1/0.5/1')
    parser.add_argument('--at_epsilon', type=float, default=5, help='对抗扰动强度 1/5')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='是否不使用 GPU')
    parser.add_argument('--loss', default="FocalLoss", help='损失函数: FocalLoss/NLLLoss')
    parser.add_argument('--class_weight', action='store_true', default=False, help='使用类别权重')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='启用 tensorboard 日志')
    parser.add_argument('--cls_type', type=str, default='emotion', help='选择情感或情绪分类')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='梯度累积步数')
    return parser.parse_args()


# -------------------------
# 主入口
# -------------------------
def main():
    args = get_args()
    print(args)

    # 基本超参数
    epochs = args.epochs
    batch_size = args.batch_size
    status = args.status
    output_path = args.output_dir
    data_path = args.data_dir
    load_model_state_dir = args.load_model_state_dir
    base_model = args.base_model
    base_layer = args.base_layer
    feature_type = args.feature_type

    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    reason_steps = [args.step_s, args.step_p]

    # 可选的 tensorboard 写入器
    writer = None
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    # -------------------------
    # 数据集/模型设置
    # -------------------------
    # MELD 特定配置
    n_speakers, hidden_size, input_size = 9, 128, None
    if args.cls_type.strip().lower() == 'emotion':
        n_classes = 7
        target_names = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
        class_weights = torch.FloatTensor(
            [1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.073096002, 1 / 0.168368836, 1 / 0.026334987, 1 / 0.117230814])
        class_weights = torch.log(class_weights)
    else:
        n_classes = 3
        target_names = ['0', '1', '2']
        class_weights = torch.FloatTensor([1.0, 1.0, 1.0])

    if feature_type == 'multi':
        input_size = 900
    elif feature_type == 'text':
        input_size = 1024
    elif feature_type == 'audio':
        input_size = 300
    else:
        print('错误: 未设置 feature_type。')
        return

    # 可重复性设置
    seed_everything(seed=args.seed)

    # 数据加载器
    train_loader, valid_loader, test_loader = get_MELD_bert_loaders(data_path,
                                                                    batch_size=batch_size,
                                                                    classify=args.cls_type,
                                                                    num_workers=0)

    # 模型
    model = build_model_and_move(cuda_flag, base_model, base_layer, input_size, hidden_size, n_speakers, n_classes, args.dropout, reason_steps)

    # 将类别权重移动到设备
    if cuda_flag:
        print('在 GPU 上运行')
        class_weights = class_weights.cuda()
    else:
        print('在 CPU 上运行')

    # 完整性检查：打印模型摘要和参数
    name = 'SACL-LSTM'
    print('{} 使用 {} 作为基础模型。'.format(name, base_model))
    print("模型总共有 {} 个参数".format(sum(x.numel() for x in model.parameters())))
    print('在 {} 特征上运行........'.format(feature_type))
    for n, p in model.named_parameters():
        print(n, p.size(), p.requires_grad)

    # 损失函数和对比学习配置
    loss_f, loss_f2, contrast_hidden_flag, contrast_weight, contrast_weight2 = build_losses_and_contrast(args, class_weights)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # 对抗训练器
    adv_trainer = build_adv_trainer_if_needed(args, model)

    # 准备输出目录
    output_path = os.path.join(output_path, '{}'.format(args.seed))
    os.makedirs(output_path, exist_ok=True)

    # -------------------------
    # 主训练/测试流程
    # -------------------------
    if status == 'train':
        all_test_fscore, all_test_acc = [], []
        best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
        patience2 = 0

        for e in range(epochs):
            start_time = time.time()
            train_loss, train_acc, train_fscore, _, _ = train_or_eval_model(
                model=model, loss_f=loss_f, dataloader=train_loader, epoch=e, train_flag=True,
                optimizer=optimizer, cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
                tensorboard=args.tensorboard, contrast_hidden_flag=contrast_hidden_flag,
                contrast_weight=contrast_weight, contrast_weight2=contrast_weight2,
                adversary_flag=args.adversary_flag, adv_trainer=adv_trainer, at_method='fgm',
                at_rate=args.at_rate, situ_rate=args.situ_rate, speaker_rate=args.speaker_rate,
                loss_f2=loss_f2, eval_cluster_flag=False, gradient_accumulation_steps=args.gradient_accumulation_steps,
                writer=writer
            )

            valid_loss, valid_acc, valid_fscore, _, _ = train_or_eval_model(
                model=model, loss_f=loss_f, dataloader=valid_loader, epoch=e, train_flag=False,
                cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
                loss_f2=loss_f2
            )

            test_loss, test_acc, test_fscore, test_metrics, label_pred = train_or_eval_model(
                model=model, loss_f=loss_f, dataloader=test_loader, epoch=e, train_flag=False,
                cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
                loss_f2=loss_f2
            )

            all_test_fscore.append(test_fscore)
            all_test_acc.append(test_acc)

            # 选择评估指标来源
            if args.use_valid_flag:
                eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
            else:
                eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore

            # 按 F1 分数保存最佳模型
            if e == 0 or best_eval_fscore < eval_fscore:
                patience = 0
                best_epoch, best_eval_fscore = e, eval_fscore
                save_model_dir = os.path.join(output_path, 'f1_{}.pkl'.format(name).lower())
                torch.save(model.state_dict(), save_model_dir)
            else:
                patience += 1

            # 按损失保存最佳模型
            if best_eval_loss is None or eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
                save_model_dir = os.path.join(output_path, 'loss_{}.pkl'.format(name).lower())
                torch.save(model.state_dict(), save_model_dir)
            else:
                patience2 += 1

            # 可选的 tensorboard 日志记录（注意：原始指标组合可能有些奇怪）
            if args.tensorboard and writer is not None:
                writer.add_scalar('train: accuracy/f1/loss', train_acc / (train_fscore + 1e-12) / (train_loss + 1e-12), e)
                writer.add_scalar('valid: accuracy/f1/loss', valid_acc / (valid_fscore + 1e-12) / (valid_loss + 1e-12), e)
                writer.add_scalar('test: accuracy/f1/loss', test_acc / (test_fscore + 1e-12) / (test_loss + 1e-12), e)
                writer.close()

            # 打印周期摘要
            print('周期: {}, 训练损失: {}, 训练准确率: {}, 训练F1: {}, 验证损失: {}, 验证准确率: {}, 验证F1: {}, 测试损失: {}, 测试准确率: {}, 测试F1: {}, 时间: {} 秒'. \
                  format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
            print(test_metrics[0])
            print(test_metrics[1])
            print("混淆矩阵: ")
            print(confusion_matrix(y_true=label_pred[0], y_pred=label_pred[1], normalize="true"))

            # 早停条件（两个耐心值都必须达到阈值）
            if patience >= args.patience and patience2 >= args.patience:
                print('早停...', patience, patience2)
                break

        # 最终摘要
        print('最终测试性能...')
        print('早停...', patience, patience2)
        print('评估指标-损失, 周期: {}, 准确率: {}, F1分数: {}'.format(best_epoch2,
                                                                                all_test_acc[best_epoch2] if best_epoch2 >= 0 else 0,
                                                                                all_test_fscore[best_epoch2] if best_epoch2 >= 0 else 0))

    elif status == 'test':
        # 加载模型并在测试集上评估（包含聚类指标）
        start_time = time.time()
        model.load_state_dict(torch.load(load_model_state_dir))
        test_loss, test_acc, test_fscore, test_metrics, label_pred = train_or_eval_model(
            model=model, loss_f=loss_f, dataloader=test_loader, epoch=0, train_flag=False,
            cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
            loss_f2=loss_f2, eval_cluster_flag=True
        )
        print('测试损失: {}, 测试准确率: {}, 测试F1: {}, 时间: {} 秒'.format(test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_metrics[0])
        print(test_metrics[1])
        print("混淆矩阵: ")
        print(confusion_matrix(y_true=label_pred[0], y_pred=label_pred[1], normalize="true"))

        # 聚类/表示质量指标
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score, \
            calinski_harabasz_score, davies_bouldin_score
        from sklearn.cluster import KMeans

        y, y_p, X = label_pred[0], label_pred[1], label_pred[2]
        ari = adjusted_rand_score(y, y_p)
        nmi = normalized_mutual_info_score(y, y_p)
        ami = adjusted_mutual_info_score(y, y_p)
        fmi = fowlkes_mallows_score(y, y_p)
        print("[有监督指标] ARI: {:.4f}, NMI: {:.4f}, AMI: {:.4f}, FMI: {:.4f}".format(ari, nmi, ami, fmi))

        print("使用 kmeans ....")
        kmeans = KMeans(n_clusters=n_classes)
        y_pred = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, y_pred)
        ch = calinski_harabasz_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        print("[无监督指标] 轮廓系数: {:.4f}, Calinski-Harabasz 指数: {:.4f}, Davies-Bouldin 指数: {:.4f}".format(
            silhouette, ch, db))

    else:
        print('状态必须是 train/test 之一')
        return


if __name__ == '__main__':
    main()
