import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)


class T5Generator:
    # def __init__(self, model_checkpoint):
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #     self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    #     self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
    #     self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')
    def __init__(self, model_checkpoint):
        # 1. 加载分词器（建议显式关闭 fast 分词器，更稳）
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
        
        # 2. 加载模型
        print(f"--- 正在从 {model_checkpoint} 载入权重 ---")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        # 2. 我们手动检查权重是否加载成功（这行不会报错）
        if self.model.encoder.embed_tokens.weight.requires_grad:
        # 打印一个参数的平均值，如果不是 0，说明加载成功了
            w_mean = self.model.encoder.embed_tokens.weight.mean().item()
            print(f"--- ✅ 权重加载成功！Embedding 均值: {w_mean:.6f} ---")
        
        # 🚀 【核心调试：检查权重是否为随机值】
        # 我们打印模型第一层（shared 层）前 5 个参数的值
        # 如果是预训练权重，它们会有特定的数值；如果是随机初始化，它们通常会非常接近 0 且分布均匀
        weights_sample = self.model.shared.weight[0][:5].detach().cpu().float().numpy()
        print(f"--- 🚀 权重样本 (应不为全0或极小值): {weights_sample}")
        
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 修正原代码可能的错误
        self.model.to(self.device)
    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("eval")),
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_pred = 0
        total_gt = 0
        tp = 0
        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_asp = gt_val.split(':')[0]

                    try:
                        gt_op = gt_val.split(':')[1]
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(':')[2]
                    except:
                        continue

                    for pred_val in pred_list:
                        pr_asp = pred_val.split(':')[0]

                        try:
                            pr_op = pred_val.split(':')[1]
                        except:
                            continue

                        try:
                            pr_sent = gt_val.split(':')[2]
                        except:
                            continue

                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp+=1

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r), None


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation", tokenized_datasets.get("eval")),
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)