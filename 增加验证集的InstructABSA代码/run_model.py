import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from InstructABSA.config import Config
from instructions import InstructionsHandler
import ast
try:
    use_mps = True if torch.has_mps else False
except:
    use_mps = False

# Set Global Values
config = Config()
instruct_handler = InstructionsHandler()
if config.inst_type == 1:
    instruct_handler.load_instruction_set1()
else:
    instruct_handler.load_instruction_set2()

print('Task: ', config.task)

if config.mode == 'train':
    if config.id_tr_data_path is None:
        raise Exception('Please provide training data path for mode=training.')
    
if config.mode == 'eval':
    if config.id_te_data_path is None and config.ood_te_data_path is None:
        raise Exception('Please provide testing data path for mode=eval.')

if config.experiment_name is not None and config.mode == 'train':
    print('Experiment Name: ', config.experiment_name)
    model_checkpoint = config.model_checkpoint
    model_out_path = config.output_dir
    model_out_path = os.path.join(model_out_path, config.task, f"{model_checkpoint.replace('/', '')}-{config.experiment_name}")
else:
    model_checkpoint = config.model_checkpoint
    model_out_path = config.model_checkpoint

print('Mode set to: ', 'training' if config.mode == 'train' else ('inference' if config.mode == 'eval' \
                                                                  else 'Individual sample inference'))

# Load the data
id_tr_data_path = config.id_tr_data_path
ood_tr_data_path = config.ood_tr_data_path
id_te_data_path = config.id_te_data_path
ood_te_data_path = config.ood_te_data_path

if config.mode != 'cli':
    id_tr_df,  id_te_df = None, None
    ood_tr_df,  ood_te_df = None, None
    eval_df = None
    if id_tr_data_path is not None:
        id_tr_df = pd.read_csv(id_tr_data_path)
        if config.validation_split_size > 0 and config.mode == 'train':
            id_tr_df, eval_df = train_test_split(
                id_tr_df,
                test_size=config.validation_split_size,
                random_state=42
            )
            print(f"Splitting off {config.validation_split_size*100}% of training data for validation.")
            print(f"New training set size: {len(id_tr_df)}")
            print(f"Validation set size: {len(eval_df)}")

    if id_te_data_path is not None:
        id_te_df = pd.read_csv(id_te_data_path)
    if ood_tr_data_path is not None:
        ood_tr_df = pd.read_csv(ood_tr_data_path)
    if ood_te_data_path is not None:
        ood_te_df = pd.read_csv(ood_te_data_path)
    print('Loaded data...')
else:
    print('Running inference on input: ', config.test_input)

# Training arguments
training_args = {
                'output_dir': model_out_path,
                'eval_strategy': config.eval_strategy,
                'learning_rate': config.learning_rate,
                'per_device_train_batch_size': config.per_device_train_batch_size if config.per_device_train_batch_size is not None else None,
                'per_device_eval_batch_size': config.per_device_eval_batch_size,
                'num_train_epochs': config.num_train_epochs if config.num_train_epochs is not None else None,
                'weight_decay': config.weight_decay,
                'warmup_ratio': config.warmup_ratio,
                'save_strategy': config.save_strategy,
                'load_best_model_at_end': config.load_best_model_at_end,
                'push_to_hub': config.push_to_hub,
                'eval_accumulation_steps': config.eval_accumulation_steps,
                'predict_with_generate': config.predict_with_generate,
                
            }

if training_args['load_best_model_at_end']:
    training_args['save_strategy'] = training_args['eval_strategy']

# Create T5 model object
print(config.set_instruction_key)
if config.set_instruction_key == 1:
    indomain = 'bos_instruct1'
    outdomain = 'bos_instruct2'
else:
    indomain = 'bos_instruct2'
    outdomain = 'bos_instruct1'

if config.task == 'ate':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.ate[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.ate[outdomain]
    eos_instruction = instruct_handler.ate['eos_instruct']
if config.task == 'atsc':
    t5_exp = T5Classifier(model_checkpoint)
    bos_instruction_id = instruct_handler.atsc[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.atsc[outdomain]
    delim_instruction = instruct_handler.atsc['delim_instruct']
    eos_instruction = instruct_handler.atsc['eos_instruct']
if config.task == 'joint':
    t5_exp = T5Generator(model_checkpoint)
    bos_instruction_id = instruct_handler.joint[indomain]
    if ood_tr_data_path is not None or ood_te_data_path is not None:
        bos_instruction_ood = instruct_handler.joint[outdomain]
    eos_instruction = instruct_handler.joint['eos_instruct']

if config.mode != 'cli':
    # 1. 初始化 Loader
    loader = DatasetLoader(
        train_df_id=id_tr_df,
        test_df_id=id_te_df,
        train_df_ood=ood_tr_df,
        test_df_ood=ood_te_df,
        sample_size=config.sample_size,
        eval_df_id=eval_df
    )

    # 2. 定义清洗函数
    import ast 
    def clean_data(target):
        try:
            if not isinstance(target, str): return []
            # 关键：直接返回解析后的列表对象，不要进行 join 变成字符串
            val = ast.literal_eval(target)
            return val if isinstance(val, list) else []
        except:
            return []

    # 3. 在格式转换前，清洗所有 Dataframe
    for df in [loader.train_df_id, loader.eval_df_id, loader.test_df_id, loader.train_df_ood, loader.test_df_ood]:
        if df is not None and 'aspectTerms' in df.columns:
            df['aspectTerms'] = df['aspectTerms'].apply(clean_data)
        
    print("✅ 数据清洗完成，已将 JSON 格式转换为纯文本。")

    # 4. 任务判断和格式转换逻辑 (注意：以下所有 if/elif 必须保持同样的左边距)
    if config.task == 'ate':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.eval_df_id is not None:
            loader.eval_df_id = loader.create_data_in_ate_format(loader.eval_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_ate_format(loader.train_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_ate_format(loader.test_df_ood, 'term', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)

        # --- 🚀 核心修复：把生成的 JSON 字符串标签强行洗成纯单词 ---
        def final_format(text):
            if not isinstance(text, str): return str(text)
            # 这个函数把 "[{'term': 'Games'}]" 强行变成 "Games"
            for char in ["[", "]", "{", "}", "'", "term:", "polarity:"]:
                text = text.replace(char, "")
            return text.strip().strip(",").strip()

        for df_attr in ['train_df_id', 'eval_df_id', 'test_df_id', 'train_df_ood', 'test_df_ood']:
            df_obj = getattr(loader, df_attr)
            if df_obj is not None and 'labels' in df_obj.columns:
                df_obj['labels'] = df_obj['labels'].apply(final_format)
                # 顺便把 aspectTerms 也洗了，方便 DEBUG 打印看结果
                df_obj['aspectTerms'] = df_obj['labels']
        
        print("✨ 最终训练标签已同步为纯文本格式，与 Prompt 完美匹配！")

    elif config.task == 'atsc':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.eval_df_id is not None:
            loader.eval_df_id = loader.create_data_in_atsc_format(loader.eval_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_id, delim_instruction, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_atsc_format(loader.train_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)
        if loader.test_df_ood is not None:
            loader.test_df_ood = loader.create_data_in_atsc_format(loader.test_df_ood, 'aspectTerms', 'term', 'raw_text', 'aspect', bos_instruction_ood, delim_instruction, eos_instruction)

    elif config.task == 'joint':
        if loader.train_df_id is not None:
            loader.train_df_id = loader.create_data_in_joint_task_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.eval_df_id is not None:
            loader.eval_df_id = loader.create_data_in_joint_task_format(loader.eval_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.test_df_id is not None:
            loader.test_df_id = loader.create_data_in_joint_task_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_id, eos_instruction)
        if loader.train_df_ood is not None:
            loader.train_df_ood = loader.create_data_in_joint_task_format(loader.train_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
        if loader.test_df_ood is not None:
                loader.test_df_ood = loader.create_data_in_joint_task_format(loader.test_df_ood, 'term', 'polarity', 'raw_text', 'aspectTerms', bos_instruction_ood, eos_instruction)
    print("\n" + "="*50)
    print("【📊 深度调试：自动检测列名与内容】")
    if loader.train_df_id is not None:
        print(f"当前 DataFrame 的所有列名: {loader.train_df_id.columns.tolist()}")
        
        # 自动寻找可能的标签列名（通常是 'target' 或作者原始的 'aspectTerms'）
        potential_target_cols = ['target', 'aspectTerms', 'label', 'output']
        actual_target_col = next((c for c in potential_target_cols if c in loader.train_df_id.columns), None)
        
        print("-" * 20)
        print("1. 第一条训练数据 (Input):")
        print(loader.train_df_id['text'].iloc[0]) # 'text' 看来是存在的
        
        if actual_target_col:
            print(f"\n2. 发现标签列 [{actual_target_col}] (Target):")
            print(loader.train_df_id[actual_target_col].iloc[0])
        else:
            print("\n❌ 错误：找不到任何已知的标签列！请检查 create_data_in_ate_format 函数。")
    print("="*50 + "\n")

    # Tokenize dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs) 

    if config.mode == 'train':
        # Train model
        training_args['eval_strategy'] = 'epoch' if id_tokenized_ds.get("eval") else 'no'
        model_trainer = t5_exp.train(id_tokenized_ds, **training_args)
        print('Model saved at: ', model_out_path)
    elif config.mode == 'eval':
        # Get prediction labels
        print('Model loaded from: ', model_checkpoint)
        if id_tokenized_ds.get("train") is not None:
            id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train', 
                                                  batch_size=config.per_device_eval_batch_size, 
                                                  max_length = config.max_token_length)
            id_tr_df = pd.DataFrame(id_ds['train'])[['text', 'labels']]
            id_tr_df['labels'] = id_tr_df['labels'].apply(lambda x: x.strip())
            id_tr_df['pred_labels'] = id_tr_pred_labels
            id_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_train.csv'), index=False)
            print('*****Train Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_tr_df['labels'], id_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)


        if id_tokenized_ds.get("test") is not None:
            id_te_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'test', 
                                                  batch_size=config.per_device_eval_batch_size, 
                                                  max_length = config.max_token_length)
            id_te_df = pd.DataFrame(id_ds['test'])[['text', 'labels']]
            id_te_df['labels'] = id_te_df['labels'].apply(lambda x: x.strip())
            id_te_df['pred_labels'] = id_te_pred_labels
            id_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_id_test.csv'), index=False)
            print('*****Test Metrics*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(id_te_df['labels'], id_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', recall)
            print('F1-Score: ', f1)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)

        if ood_tokenized_ds.get("train") is not None:
            ood_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'train', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_tr_df = pd.DataFrame(ood_ds['train'])[['text', 'labels']]
            ood_tr_df['labels'] = ood_tr_df['labels'].apply(lambda x: x.strip())
            ood_tr_df['pred_labels'] = ood_tr_pred_labels
            ood_tr_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_train.csv'), index=False)
            print('*****Train Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(ood_tr_df['labels'], ood_tr_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
            
        if ood_tokenized_ds.get("test") is not None:
            ood_te_pred_labels = t5_exp.get_labels(tokenized_dataset = ood_tokenized_ds, sample_set = 'test', 
                                                   batch_size=config.per_device_eval_batch_size, 
                                                   max_length = config.max_token_length)
            ood_te_df = pd.DataFrame(ood_ds['test'])[['text', 'labels']]
            ood_te_df['labels'] = ood_te_df['labels'].apply(lambda x: x.strip())
            ood_te_df['pred_labels'] = ood_te_pred_labels
            ood_te_df.to_csv(os.path.join(config.output_path, f'{config.experiment_name}_ood_test.csv'), index=False)
            print('*****Test Metrics - OOD*****')
            precision, recall, f1, accuracy = t5_exp.get_metrics(ood_te_df['labels'], ood_te_pred_labels)
            print('Precision: ', precision)
            print('Recall: ', precision)
            print('F1-Score: ', precision)
            if config.task == 'atsc':
                print('Accuracy: ', accuracy)
else:
    print('Model loaded from: ', model_checkpoint)
    if config.task == 'atsc':
        config.test_input, aspect_term = config.test_input.split('|')[0], config.test_input.split('|')[1]
        model_input = bos_instruction_id + config.test_input + f'. The aspect term is: {aspect_term}' + eos_instruction
    else:
        model_input = bos_instruction_id + config.test_input + eos_instruction
    input_ids = t5_exp.tokenizer(model_input, return_tensors="pt").input_ids
    outputs = t5_exp.model.generate(input_ids, max_length = config.max_token_length)
    print('Model output: ', t5_exp.tokenizer.decode(outputs[0], skip_special_tokens=True))
