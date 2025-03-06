import os, argparse

import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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

def apply_chat_template(example, tokenizer):
    example['text'] = tokenizer.apply_chat_template(
        example['messages'],
        tokenize = False,
        add_generation_prompt = False,
    )
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, required = True)
    parser.add_argument('--outdir', default = 'models/sft', type = str)
    parser.add_argument('--wandb', default = True, type = bool)
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    os.environ['WANDB_DISABLED'] = 'true' if not args.wandb or args.debug else 'false'


    model_name = 'HuggingFaceTB/SmolLM2-135M'
    tokenizer_name = 'HuggingFaceTB/SmolLM2-135M-Instruct' # Need the instruct version for chat template

    # uses flash att: pip install flash-attn --no-build-isolation
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16, attn_implementation = 'flash_attention_2', device_map='auto')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue


    print('Preprocessing the dataset')
    sft_data = load_dataset('HuggingFaceTB/smol-smoltalk')
    column_names = sft_data.column_names['train']

    if args.debug:
        sft_data = DatasetDict({
            'train': sft_data['train'].select(range(10)),
            'test': sft_data['test'].select(range(10))
        })
        print(sft_data)

    sft_data = sft_data.map(
        apply_chat_template,
        fn_kwargs = {'tokenizer': tokenizer},
        num_proc = min(16, os.cpu_count()),
        batched = True, #?
        remove_columns = column_names,
        desc = 'Applying chat template',
    )

    # Do it here instead of in SFTTrainer so that HF caches it
    sft_data = sft_data.map(
        lambda examples: tokenizer(
            examples['text'],
            padding = True,
            truncation = True,
            max_length = 2048,
            return_tensors = 'pt'
        ),
        num_proc = min(16, os.cpu_count()),
        batched = True,
        desc = 'Tokenizing the dataset'
    )

    sft_train, sft_test = sft_data['train'], sft_data['test']
    
    print(f'Train len: {len(sft_train)}, test len: {len(sft_test)}')

    print(f'Running with seed {args.seed}')
    set_seed(args.seed)

    if args.wandb and not args.debug:
        import wandb
        wandb.login()
        wandb.init(
            project = 'salsa-sft-training',
            name = f'sft_seed_{args.seed}',
            config = {'seed': args.seed},
            save_code = True
        )

    sft_config = SFTConfig(
        dataset_text_field = 'text',
        num_train_epochs = 1.0, # changed from 2.0 to save compute

        learning_rate = 1.0e-03,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'cosine',

        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        max_seq_length = 2048,

        output_dir = f'models/sft/{args.seed}',
        seed = args.seed,
        data_seed = args.seed,

        bf16 = True,
        packing = True,
        do_eval = True,
        eval_strategy = 'epoch',

        save_strategy = 'epoch',

        log_level = 'info',
        logging_strategy = 'steps',
        overwrite_output_dir = True,
        report_to = 'wandb' if args.wandb and not args.debug else None,

        hub_strategy = 'every_save',
        hub_model_id = f'SmolLM2-135M-SFT-{args.seed}',
        push_to_hub = not args.debug,

        # max_steps = 1 if args.debug else -1
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = sft_train,
        eval_dataset = sft_test,
        args = sft_config,
    )

    trainer.save_model(args.outdir)
    trainer.train()
