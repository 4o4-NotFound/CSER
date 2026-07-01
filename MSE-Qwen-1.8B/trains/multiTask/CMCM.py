import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain
import json
import shutil

logger = logging.getLogger('MSA')

class CMCM():
    def __init__(self, args):

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def _get_real_model(self, model):
        return model.Model if hasattr(model, 'Model') else model

    def _decode_simsv2_text(self, tokenizer, text_ids):
        """
        text_ids: 1D tensor/list
        返回中文字符串和逐token字符串，尽量避免乱码
        """
        ids = text_ids.tolist() if hasattr(text_ids, 'tolist') else list(text_ids)

        # 去掉常见 pad
        ids = [int(x) for x in ids if int(x) >= 0]

        if tokenizer is None:
            return ''.join([str(x) for x in ids]), [str(x) for x in ids]

        try:
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')
        except Exception:
            text = ''.join([str(x) for x in ids])

        token_strs = []
        for tid in ids:
            try:
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    tok = tokenizer.convert_ids_to_tokens(int(tid))
                else:
                    tok = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)

                if isinstance(tok, bytes):
                    tok = tok.decode('utf-8', errors='replace')

                if tok is None:
                    tok = str(tid)

            except Exception:
                tok = str(tid)

            token_strs.append(tok)

        return text, token_strs

    def _to_jsonable(self, obj):
        """
        递归地把 bytes / numpy 类型 / 其他不可序列化对象
        转成 json 可写入的 Python 基本类型
        """
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except Exception:
                return obj.decode('utf-8', errors='replace')

        if isinstance(obj, dict):
            return {self._to_jsonable(k): self._to_jsonable(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._to_jsonable(x) for x in obj]

        if isinstance(obj, tuple):
            return [self._to_jsonable(x) for x in obj]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj

    def _filter_special_tokens(self, token_strs, alpha_before, alpha_after, selected_pos):
        """
        过滤掉 padding / special token，并把 selected_pos 重映射到新位置
        """
        special_tokens = {
            "<|endoftext|>", "<pad>", "[PAD]", "[CLS]", "[SEP]",
            "<s>", "</s>", "<unk>", "[UNK]", ""
        }

        keep_indices = []
        clean_tokens = []

        for i, tok in enumerate(token_strs):
            tok_str = str(tok).strip()
            if tok_str not in special_tokens:
                keep_indices.append(i)
                clean_tokens.append(tok)

        if len(keep_indices) == 0:
            # 极端情况：如果全被过滤，直接原样返回，避免崩掉
            return token_strs, alpha_before, alpha_after, selected_pos

        alpha_before_new = alpha_before[keep_indices]
        alpha_after_new = alpha_after[keep_indices]

        old2new = {old_i: new_i for new_i, old_i in enumerate(keep_indices)}
        selected_pos_new = [old2new[p] for p in selected_pos.tolist() if int(p) in old2new]

        return clean_tokens, alpha_before_new, alpha_after_new, np.asarray(selected_pos_new, dtype=np.int64)

    def _export_final_topk_results(self, model, save_dir='results/simsv2-0.1_final_topk_vis', max_cases=20):
        """
        只在整个最终 TEST 全部结束后调用一次
        从模型缓存中统一导出中文可解释结果
        """
        real_model = self._get_real_model(model)

        if not hasattr(real_model, 'final_topk_cases'):
            logger.info("No final_topk_cases found, skip exporting topk results.")
            return

        cases = real_model.final_topk_cases
        if len(cases) == 0:
            logger.info("final_topk_cases is empty, skip exporting topk results.")
            return

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 尽量找到 tokenizer
        tokenizer = None
        if hasattr(real_model, 'LLM') and hasattr(real_model.LLM, 'tokenizer'):
            tokenizer = real_model.LLM.tokenizer
        elif hasattr(real_model, 'tokenizer'):
            tokenizer = real_model.tokenizer

        # 全局统计
        selected_freq = None
        entropy_before = []
        entropy_after = []

        def entropy(p):
            p = np.asarray(p, dtype=np.float64)
            p = np.clip(p, 1e-12, 1.0)
            return float(-(p * np.log(p)).sum())

        export_cases = cases if max_cases is None else cases[:max_cases]
        freq_cases = cases

        # ===== 先用全部最终 TEST case 统计 selection_frequency，只执行一次 =====
        for item in freq_cases:
            text_ids = item['text_ids']
            alpha_before = np.asarray(item['alpha_before'], dtype=np.float64)
            alpha_after = np.asarray(item['alpha_after'], dtype=np.float64)
            selected_pos = np.asarray(item['selected_pos'], dtype=np.int64)

            raw_text, token_strs = self._decode_simsv2_text(tokenizer, text_ids)

            token_strs, alpha_before, alpha_after, selected_pos = self._filter_special_tokens(
                token_strs, alpha_before, alpha_after, selected_pos
            )

            if selected_freq is None:
                selected_freq = np.zeros_like(alpha_before, dtype=np.float64)

            mask = np.zeros_like(alpha_before, dtype=np.float64)
            valid_selected_pos = selected_pos[(selected_pos >= 0) & (selected_pos < len(mask))]
            mask[valid_selected_pos] = 1.0

            if len(mask) > len(selected_freq):
                selected_freq = np.pad(selected_freq, (0, len(mask) - len(selected_freq)), mode='constant')
            elif len(mask) < len(selected_freq):
                mask = np.pad(mask, (0, len(selected_freq) - len(mask)), mode='constant')

            selected_freq += mask

        # ===== 再只导出前 max_cases 个 case 文件 =====
        for idx, item in enumerate(export_cases):
            text_ids = item['text_ids']
            alpha_before = np.asarray(item['alpha_before'], dtype=np.float64)
            alpha_after = np.asarray(item['alpha_after'], dtype=np.float64)
            selected_pos = np.asarray(item['selected_pos'], dtype=np.int64)

            raw_text, token_strs = self._decode_simsv2_text(tokenizer, text_ids)

            token_strs, alpha_before, alpha_after, selected_pos = self._filter_special_tokens(
                token_strs, alpha_before, alpha_after, selected_pos
            )

            entropy_before.append(entropy(alpha_before))
            entropy_after.append(entropy(alpha_after))

            selected_tokens = []
            for pos in selected_pos.tolist():
                if 0 <= pos < len(token_strs):
                    selected_tokens.append({
                        "position": int(pos),
                        "token": token_strs[pos],
                        "alpha_before": float(alpha_before[pos]),
                        "alpha_after": float(alpha_after[pos]),
                    })

            # 保存中文可解释 json
            case_obj = {
                "case_id": idx,
                "raw_text": raw_text,
                "tokens": token_strs,
                "selected_tokens": selected_tokens,
                "topk_positions": selected_pos.tolist(),
                "alpha_before": alpha_before.tolist(),
                "alpha_after": alpha_after.tolist(),
            }
            case_obj = self._to_jsonable(case_obj)

            with open(os.path.join(save_dir, f'case_{idx:03d}.json'), 'w', encoding='utf-8') as f:
                json.dump(case_obj, f, ensure_ascii=False, indent=2)

            # 保存中文可读 txt
            with open(os.path.join(save_dir, f'case_{idx:03d}.txt'), 'w', encoding='utf-8-sig') as f:
                f.write(f"case_id: {idx}\n")
                f.write(f"raw_text: {raw_text}\n\n")
                f.write("selected_tokens:\n")
                for x in selected_tokens:
                    f.write(f"  pos={x['position']}, token={x['token']}, "
                            f"alpha_before={x['alpha_before']:.6f}, alpha_after={x['alpha_after']:.6f}\n")

            # 保存前后对比图
            x = np.arange(len(alpha_before))
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            axes[0].bar(x, alpha_before)
            axes[0].scatter(selected_pos, alpha_before[selected_pos], marker='o')
            axes[0].set_title(f'Case {idx:03d} - Before TopK')
            axes[0].set_ylabel('Weight')

            axes[1].bar(x, alpha_after)
            axes[1].scatter(selected_pos, alpha_after[selected_pos], marker='o')
            axes[1].set_title(f'Case {idx:03d} - After TopK')
            axes[1].set_xlabel('Token Position')
            axes[1].set_ylabel('Weight')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'case_{idx:03d}_compare.png'), dpi=200)
            plt.close(fig)

        # 全局图
        selected_freq = selected_freq / len(freq_cases)

        plt.figure(figsize=(14, 6))
        plt.bar(np.arange(len(selected_freq)), selected_freq)
        plt.title('TopK Selection Frequency over Final Test Cases')
        plt.xlabel('Token Position')
        plt.ylabel('Selection Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'selection_frequency.png'), dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.bar(['Before TopK', 'After TopK'], [np.mean(entropy_before), np.mean(entropy_after)])
        plt.ylabel('Average Entropy')
        plt.title('Entropy Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'entropy_compare.png'), dpi=200)
        plt.close()

        with open(os.path.join(save_dir, 'summary.txt'), 'w', encoding='utf-8-sig') as f:
            f.write(f'export_cases: {len(export_cases)}\n')
            f.write(f'mean_entropy_before: {np.mean(entropy_before):.8f}\n')
            f.write(f'mean_entropy_after: {np.mean(entropy_after):.8f}\n')

    def do_train(self, model, dataloader):

        scaler = GradScaler()
        optimizer = optim.AdamW(model.Model.parameters(), lr= self.args.learning_rate, eps=1e-4)
        total_steps = len(dataloader['train'])*self.args.warm_up_epochs   #大致的一个训练step数
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
        #                               threshold=0.0001, eps=1e-08)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        # with tqdm(dataloader['train']) as td:
        #     for batch_data in td:
        #         if self.args.train_mode == 'regression':
        #             labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
        #         else:
        #             labels_m = batch_data['labels']['M']
        #         indexes = batch_data['index'].view(-1)
        #         # self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        losses = []

        CPC_Losses = []
        # valid_F1 = []
        lr = []
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0     #评价阈值的初始化
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            CPC_Loss_sum = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()      #在训练1个batch之后停止梯度清0，当新的epoch来临时才清0
                    left_epochs -= 1                #这么做相当于把batch_size扩大为（N-1）*batch_size，其中N为一个epoch中的batch数

                    # optimizer.zero_grad()
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if self.args.train_mode == 'regression':
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        prefix_label = batch_data['labels_prefix']
                        cur_id = batch_data['id']
                        ids.extend(cur_id)
                    else:
                        labels_m = batch_data['labels']['M']

                    indexes = batch_data['index'].view(-1)


                    if not self.args.need_data_aligned:
                        text_lengths = batch_data['text_lengths'].to(self.args.device)
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    # forward
                    # with autocast():
                    #     output= model(labels_m, (text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                    #     loss = output['Loss']
                    with autocast():
                        output= model(labels_m, (text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                        loss = output['Loss']



                    # backward
                    scaler.scale(loss).backward()
                    train_loss += loss.item()
                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                    # update parameters
                    if not left_epochs:
                        # update
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    scaler.step(optimizer)
                    scaler.update()
            # scheduler.step()   #每个epoch衰减一次学习率
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f" % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            losses.append(train_loss)

            # validation

            if epochs >= 1:         #前3epochs不做eval
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                # valid_losses.append(val_results['Loss'])
                # valid_F1.append(cur_valid)
                # save best model
                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    # save model
                    # torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    self.save_model(model, epochs, self.args.model_save_path)
                    model.to(self.args.device)

                # # save labels
                # if self.args.save_labels:
                #     tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                #     tmp_save['ids'] = ids
                #     saved_labels[epochs] = tmp_save
                # early stop
                if epochs - best_epoch >= self.args.early_stop:     #如果比best_epoch再过了early_stop轮之后还没有出现新的best_epoch，就停止训练
                    if self.args.save_labels:
                        with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                            plk.dump(saved_labels, df, protocol=4)
                    # self.loss_plt(losses,CPC_Losses)
                    # self.lr_plt(lr)
                    return


    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()

        real_model = self._get_real_model(model)

        # 只在最终 TEST 前清空缓存；VAL 不保存任何可视化
        if hasattr(real_model, 'enable_topk_save'):
            real_model.enable_topk_save = (mode == "TEST")

        if mode == "TEST":
            # 最终 TEST 开始前清空旧缓存
            real_model.final_topk_cases = []

        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}

        # eval_loss = 0.0
        # criterion = nn.L1Loss()
        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast():
                            outputs = model.generate((text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        predict_label = torch.Tensor(outputs).to(self.args.device)

                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        # loss = self.l1_loss(predict_label, labels_m)
                        # eval_loss += loss.item()
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())
            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            # print(pred)
            # eval_loss = eval_loss / len(dataloader)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>" )
            eval_results = self.metrics(pred, true)
            logger.info('M: >> ' + dict_to_str(eval_results))
            # eval_results['Loss'] = eval_loss
        else:
            # train_mode == 'classification'
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast():
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths),
                                                     (vision, vision_lengths))

                        # predict_label = torch.Tensor(outputs).to(self.args.device)
                        predict_label = outputs
                        labels_m = batch_data['labels']['M']
                        # y_pred['M'].append(predict_label.cpu().numpy())
                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)
            # pred, true = torch.cat(y_pred['M']), torch.cat(y_pred['M'])
            pred, true = list(chain(*y_pred['M'])), list(chain(*y_true['M']))
            # print(pred)
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        # 只在整个最终 TEST 全部结束后统一导出
        if mode == "TEST":
            self._export_final_topk_results(model, save_dir='results/simsv2-0.1_final_topk_vis', max_cases=20)

        return eval_results
    
    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss



    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def save_model(self, model, epoch, save_path):
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.cpu().state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_path))
        torch.save(state_dict, save_path)


    # def loss_plt(self,loss,CPC_Losses):
    #     matplotlib.rcParams['font.family'] = 'serif'  # 设置字体族
    #     matplotlib.rcParams['font.serif'] = ['Arial']  # 选择字体
    #     logging.getLogger('matplotlib').setLevel(logging.ERROR)
    #     # train_x = range(len(loss))
    #     # train_y = loss
    #     # kl_x = range(len(KL_losses))
    #     # kl_y = KL_losses
    #     #
    #     # save_path = os.path.join(self.args.res_save_dir, f'{self.args.datasetName}-{self.args.train_mode}.jpg')
    #     # fig, axs = plt.subplots(2, 1)
    #     #
    #     # # Plot Train Loss
    #     # axs[0].plot(train_x, train_y, label='Train')
    #     # axs[0].set_ylabel('Loss')
    #     # axs[0].set_ylim([0, max(train_y) * 1.2])
    #     # axs[0].set_yticks(np.arange(0, max(train_y) + 0.1, (max(train_y) - min(train_y)) / 5))
    #     # axs[0].legend(loc='upper right')
    #     #
    #     # # Plot KL Loss with Log Scale
    #     # axs[1].plot(kl_x, kl_y, label='KL_losses')
    #     # axs[1].set_yscale('log')  # Set log scale for KL Loss
    #     # axs[1].set_ylabel('KL-Loss')
    #     # axs[1].set_ylim([min(kl_y) - 0.05, max(kl_y) + 0.05])
    #     # axs[1].set_yticks(np.arange(min(kl_y), max(kl_y) + 0.01, (max(kl_y) - min(kl_y)) / 5))
    #     # axs[1].legend(loc='upper right')
    #     #
    #     # plt.xlabel('epoch')
    #     # plt.subplots_adjust(hspace=0.5)
    #     # plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    #     # plt.close()
    #     # plt.show()
    #     kl_x = range(len(CPC_Losses))
    #     kl_y = CPC_Losses
    #
    #     save_path = os.path.join(self.args.res_save_dir, f'{self.args.datasetName}-{self.args.train_mode}_CPC_Losses.jpg')
    #     fig, ax = plt.subplots(figsize=(8, 6))  # 调整图的大小
    #
    #     # Plot KL Loss with Log Scale
    #     ax.plot(kl_x, kl_y, label='KL_losses')
    #     ax.set_yscale('log')  # 设置 KL Loss 的纵坐标为对数坐标
    #     ax.set_ylabel('KL-Loss')
    #     ax.set_ylim([min(kl_y) - 0.05, max(kl_y) + 0.05])
    #
    #     # 使用自动设置刻度
    #     ax.yaxis.set_major_locator(plt.AutoLocator())
    #
    #     ax.legend(loc='upper right')
    # 
    #     plt.xlabel('epoch')
    #     plt.tight_layout()  # 自动调整布局
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    #     plt.close()
    #     plt.show()

    # def lr_plt(self,lr):
    #     plt.plot(np.arange(len(lr)), lr)
    #     plt.xlabel('Step')
    #     plt.ylabel('Learning Rate')
    #     plt.title('Warm-up Learning Rate Schedule')
    #     plt.show()
