import os, argparse

import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    ('mps' if torch.backends.mps.is_available() else
    'cpu')
)

def set_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        print('Setting seed for CUDA')
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_function(examples, tokenizer, max_length):

    c_texts = tokenizer.apply_chat_template(examples['chosen'], tokenize = False)
    r_texts = tokenizer.apply_chat_template(examples['rejected'], tokenize = False)

    c_encodings = tokenizer(c_texts, truncation = True, padding = 'max_length', max_length = max_length)
    r_encodings = tokenizer(r_texts, truncation = True, padding = 'max_length', max_length = max_length)

    return {
        'input_ids_chosen': c_encodings['input_ids'], 'attention_mask_chosen': c_encodings['attention_mask'],
        'input_ids_rejected': r_encodings['input_ids'], 'attention_mask_rejected': r_encodings['attention_mask'],
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, required = True)
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--outdir', default = 'models/reward', type = str)
    parser.add_argument('--wandb', default = True, type = bool)
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    os.environ['WANDB_DISABLED'] = 'true' if not args.wandb or args.debug else 'false'

    model_name = args.model_name #'emiliocantuc/SmolLM2-135M-SFT-0'
    tokenizer_name = 'HuggingFaceTB/SmolLM2-135M-Instruct' # Need the instruct version for chat template
    dataset_name = 'HuggingFaceH4/ultrafeedback_binarized'

    model_seed = int(model_name.split('-')[-1])
    out_model_name = model_name + '-Reward'
    hub_model_id = out_model_name.split('/')[1]
    print(f'Getting {model_name}, saving to {out_model_name}, and uploading to {hub_model_id}')

    # uses flash att: pip install flash-attn --no-build-isolation
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1, torch_dtype = torch.bfloat16, attn_implementation = 'flash_attention_2', device_map = 'auto')
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = True)

    tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue
    model.config.pad_token_id = model.config.eos_token_id
    
    MAX_LENGTH = 2048

    print('Preprocessing the dataset')
    if args.debug:
        dataset = DatasetDict({k: dataset[k].select(range(10)) for k in dataset})
        print(dataset)

    column_names = dataset.column_names['train_prefs']
    dataset = dataset.map(
        preprocess_function,
        fn_kwargs = {'tokenizer': tokenizer, 'max_length': MAX_LENGTH}, #?
        num_proc = min(16, os.cpu_count()),
        batched = True,
        remove_columns = column_names,
        desc = 'Preprocessing dataset',
    )
    dataset.set_format(type = 'torch')

    print(f'Running with seed {args.seed}')
    set_seed(args.seed)

    if args.wandb and not args.debug:
        print('Initing wandb')
        import wandb
        wandb.login()
        wandb.init(
            project = 'salsa-reward-training',
            name = f'reward_{model_seed}_seed_{args.seed}',
            config = {'sft_seed': model_seed, 'seed': args.seed},
            save_code = True
        )

    config = RewardConfig(

        num_train_epochs = 2.0, #?

        learning_rate = 1.0e-03,  
        weight_decay = 0.001,  
        lr_scheduler_type = 'linear',  
        warmup_ratio = 0.0, 

        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,

        output_dir = f'{args.outdir}/{model_seed}_{args.seed}',
        seed = args.seed,
        data_seed = args.seed,

        bf16 = True,
        do_eval = True,
        eval_strategy = 'epoch',

        save_strategy = 'epoch',

        log_level = 'info',
        logging_strategy = 'steps',
        overwrite_output_dir = True,
        report_to = 'wandb' if args.wandb and not args.debug else None,

        hub_strategy = 'every_save',
        hub_model_id = hub_model_id,
        push_to_hub = not args.debug,

        # max_steps = 1 if args.debug else -1
    )

    trainer = RewardTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset['train_prefs'],
        eval_dataset = dataset['test_prefs'],
        args = config,
    )

    trainer.train()
