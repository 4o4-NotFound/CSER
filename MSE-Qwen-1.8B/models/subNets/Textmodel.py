import os
import sys
import collections
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope import AutoTokenizer, AutoModel, AutoModelForCausalLM


__all__ = ['Language_model']

class Language_model (nn.Module):
    def __init__(self, args, use_PLM = True):
        """
        language: en / cn
        """
        super(Language_model, self).__init__()

        if use_PLM:
            pretrained_model = args.pretrain_LM              #pretrained model select
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model,
                padding_side='left',
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).half()
            # self.pad_token_id = self.tokenizer.convert_tokens_to_ids('<|extra_0|>')
            # self.tokenizer.pad_token_id = self.pad_token_id
            # self.tokenizer.pad_token_id = 0
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
            self.tokenizer.pad_token_id = self.eos_token_id

            self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
            self.tokenizer.bos_token_id = self.bos_token_id

            self.device = args.device
            self.language = args.language
            self.max_new_tokens = args.max_new_tokens
            self.datasetName = args.datasetName
            self.train_mode = args.train_mode
            self.task_specific_prompt = args.task_specific_prompt
            # freeze parameter
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('please use PLM')

    def text_embedding(self,text_ids):
        embeddings = self.model.base_model.get_input_embeddings()
        return embeddings(text_ids)


    def forward(self, fusion_embedding, labels):
        """
        Args:
            fusion_embedding: the "concatenate" result of  multimodal low rank fusion  and text embedding
            label: ground_truth
        """

        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)  #添加多模态输入的special prompt
        opt_tokens, atts_bos, atts_fusion, labels, labels_atts = self.input_processing(fusion_embedding, labels, mode = 'train')          #创建fusion+prompt+answer_mask的input和label

        attention_mask = torch.cat([atts_bos, atts_fusion, labels_atts], dim=1)


        with torch.cuda.amp.autocast():
            output = self.model(inputs_embeds = opt_tokens, return_dict=True, labels = labels)  # Models outputs are now tuples

        return output

    def generate(self, fusion_embedding):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_new_tokens (int): The maximum length of the new tokens to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (int): The k for top-k sampling.
            penalty_alpha (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        """


        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)  # 添加多模态输入的special prompt
        opt_tokens, atts_bos, atts_fusion, _, _= self.input_processing(fusion_embedding, mode = 'generate')  # 创建fusion+prompt的input
        attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
        context_length = opt_tokens.size(1)
        all_responses =[]

        outputs = self.model.generate(inputs_embeds = opt_tokens,
                                      num_beams=1,
                                      do_sample = False,
                                      bos_token_id = self.tokenizer.bos_token_id,
                                      max_new_tokens  = self.max_new_tokens)
        responses = self.tokenizer.batch_decode(outputs[:,1:], add_special_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # print(responses)
        for response in responses:
        # 处理生成结果，将一些不必要的字符转换为0
            if self.train_mode == 'regression':
                try:
                    value = float(
                        response.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(
                            ':', ''))
                    # value = float(re.sub(r'[^0-9.-]', '0', re.sub(r'(?<!^)-', '0', x.replace('–', '-').replace('一', '-').replace('：', ''))))
                except ValueError:
                    value = 0.0
            else:
                try:
                    value = float(response)
                except ValueError:
                    value = 0.0
            all_responses.append(value)
        return all_responses


    def input_processing(self, fusion_embedding,  labels = None, mode = None):
        """
        Args:
            fusion_embedding: the "concatenate" result of  multimodal low rank fusion  and text embedding
            fusion_empty: Create an empty matrix of the same size as fusion's batch, seq, so that it can be filled in during input
            prompt: tokenizer prompt for different language cases
        """

        batch_size = fusion_embedding.shape[0]

        task_prompt = self.get_task_prompt()
        task_prompt_embedding =  self.text_embedding(task_prompt.expand(batch_size, -1))


        opt_tokens = torch.cat([fusion_embedding, task_prompt_embedding], dim=1)    #构建fusion+prompt的tokens
        atts_fusion = torch.ones(opt_tokens.size()[:-1], dtype=torch.long).to(self.device)    #构建opt_tokens 的attention mask

        bos = torch.ones([batch_size, 1], dtype=atts_fusion.dtype, device=self.device) * self.tokenizer.bos_token_id
        bos_embeds = self.text_embedding(bos)
        atts_bos = atts_fusion[:, :1]

        opt_tokens =  torch.cat([bos_embeds, opt_tokens], dim=1)

        opt_tokens, labels, labels_atts = self.input_labels_construct(opt_tokens, labels, mode)

        return opt_tokens, atts_bos, atts_fusion, labels, labels_atts

    def input_labels_construct(self, opt_tokens, labels = None, mode = None):
        """
        Args:
            opt_tokens: the "concatenate" size of  multimodal fusion, text embedding and prompt
            label: ground_truth
            labels_id: tokenizer labels
        """
        batch_size = opt_tokens.shape[0]

        if mode == "train":
            if self.train_mode == "regression":
                # label_template = [f"{label.item():.{1}f}" for label in labels]
                label_template = [f"+{label.item():.{1}f}" if label >= 0 else f"{label.item():.{1}f}" for label in
                                  labels]
                # label_template = [
                #     f"+{label.item():.1f}" if label > 0 else f"{+label.item():.1f}" if label == 0 else f"{label.item():.1f}"
                #     for label in labels]
            else:
                label_template = [f"{label.item()}" for label in labels]

            labels = self.tokenizer(label_template,  padding=True, return_tensors="pt", add_special_tokens=False).to(self.device)
            labels_id = labels["input_ids"]
            labels_atts = labels["attention_mask"]

            # a = [' ','0.20','-0.2','5','2','0','1','3','4','5','6','7','8','9']
            # c = [31106]
            # b = self.tokenizer(a, padding=True, return_tensors="pt", add_special_tokens=False)
            # d = self.tokenizer.decode(c)
            labels_embedding = self.text_embedding(labels_id)
            labels_matrix = torch.empty(opt_tokens.size(0), opt_tokens.size(1)).fill_(-100).long().to(self.device)  # bz * seq_len 只构建和token_ids一个维度的矩阵
            opt_tokens = torch.cat([opt_tokens, labels_embedding], dim=1)  # 将输入与labels拼接
            labels = torch.cat([labels_matrix, labels_id], dim=1)


        else:
            labels_atts = None

        return opt_tokens, labels, labels_atts

    def get_task_prompt(self):
        # get the task_specific_prompt
        prompt_text = self.task_specific_prompt
        prompt_ids = self.tokenizer(prompt_text, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

        return prompt_ids

    def multimodal_prompt_wrap(self,fusion_embeddings):
        """
        Args:
            Wrap the input with a special token
        """
        if self.language == "en":
            prompt = '<Multimodal><MultimodalHere></Multimodal>'
            special_token = '<MultimodalHere>'
        else:
            prompt = '<多模态><MultimodalHere></多模态>'
            special_token = '<MultimodalHere>'

        batch_size = fusion_embeddings.shape[0]
        p_before, p_after = prompt.split(special_token)
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=True).to(self.device)
        p_after_tokens = self.tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.text_embedding(p_before_tokens.input_ids.expand(batch_size, -1))
        p_after_embeds = self.text_embedding(p_after_tokens.input_ids.expand(batch_size, -1))
        wrapped_fusion_embeddings = torch.cat([p_before_embeds, fusion_embeddings, p_after_embeds], dim=1)


        return wrapped_fusion_embeddings