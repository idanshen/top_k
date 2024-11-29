import shutil
from dataclasses import asdict, dataclass, field, fields
from tqdm import tqdm
import numpy as np
import random

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
)
from peft import LoraConfig
from torch.utils.data import DataLoader
from trl import ModelConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from top_k_logit_processor import TopKLogitsProcessor


from utils import is_correct_answer

@dataclass
class ScriptArguments:
    output_dir: str = field(default=None, metadata={"help": "log dir."})
    dataset_name: str = field(default="openai/gsm8k", metadata={"help": "dataset name."})
    dataset_train_split: str = field(default="train", metadata={"help": "dataset train split."})
    per_device_decode_batch_size: int = field(default=4, metadata={"help": "per device decode batch size."})
    response_length: int = field(default=256, metadata={"help": "response length."})
    stop_token: str = field(default="eos", metadata={"help": "stop token."})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    script_args, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(script_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype="float16",
    )
    model = model.cuda()
    model_tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    if model_tokenizer.chat_template is None:
        model_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Reward Model & Tokenizer
    ################
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="Ray2333/GRM-gemma2-2B-rewardmodel-ft",
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16, 
    )
    reward_model = reward_model.cuda()
    reward_tokenizer = AutoTokenizer.from_pretrained("Ray2333/GRM-gemma2-2B-rewardmodel-ft", 
                                                     padding_side="left", 
                                                     trust_remote_code=model_config.trust_remote_code
                                                     )
    if reward_tokenizer.chat_template is None:
        reward_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Dataset
    ################
    dataset = load_dataset("lmsys/lmsys-chat-1m", split='train', token="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")#, 'main',split=script_args.dataset_train_split)
    #dataset_text_field = "question"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            #chat = [[{"role": "user", "content": row}] for row in element[dataset_text_field]]
            chat = [[item[0]] for item in element['conversation']]
            inputs = model_tokenizer.apply_chat_template(chat, tokenize=False)
            outputs = model_tokenizer(
               inputs,
               padding=False,
            )
            return {"input_ids": outputs["input_ids"], "chat": chat, "prompt": inputs, }

        return dataset.map(
            tokenize,
            batched=True,
            #remove_columns=dataset.column_names,
        )

    def data_collator(batch):
        """collate a tokenzied dataset"""
        keys = batch[0].keys()
        outputs = model_tokenizer.pad({"input_ids": [e["input_ids"] for e in batch]}, padding="longest", return_tensors="pt", padding_side="left")
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        other_fields = {key: [e[key] for e in batch] for key in keys if key != "input_ids"}
        return {"input_ids": input_ids, "attention_mask": attention_mask, **other_fields}

    dataset = prepare_dataset(dataset, model_tokenizer)
    number_of_samples = 200
    # set random seed and shuffle
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(number_of_samples))
    dataset_loader = DataLoader(dataset, batch_size=script_args.per_device_decode_batch_size, shuffle=False, collate_fn=data_collator)
    
    ################
    # Generation
    ################

    def lmsys_eval(model, dataset_loader, script_args, model_tokenizer, reward_model, reward_tokenizer):
        decoded_responses = []
        rewards = []
        logits_processors = [TopKLogitsProcessor(temp_1=0.7)]
        for batch in tqdm(dataset_loader):
            outputs = model.generate(
                    input_ids=batch["input_ids"].cuda(), 
                    attention_mask=batch["attention_mask"].cuda(), 
                    num_return_sequences=1,
                    max_new_tokens=script_args.response_length,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    do_sample=False,
                    #logits_processor=logits_processors,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(batch["input_ids"], outputs)
            ]
            decoded_text = model_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_text = [item[5:] for item in decoded_text]
            decoded_responses.extend(decoded_text)
            chat_with_response = [batch["chat"][i] + [{"role": "assistant", "content": decoded_text[i]}] for i in range(len(decoded_text))]
            reward_inputs = reward_tokenizer.apply_chat_template(chat_with_response, tokenize=False)
            reward_inputs = reward_tokenizer(reward_inputs, padding=True, return_tensors="pt", padding_side="left")
            reward_outputs = reward_model(
                input_ids=reward_inputs["input_ids"].cuda(),
                attention_mask=reward_inputs["attention_mask"].cuda(),
            )
            rewards.extend(reward_outputs[0].cpu().detach().numpy().flatten().tolist())

        return np.mean(rewards)

    print(lmsys_eval(model, dataset_loader, script_args, model_tokenizer))
