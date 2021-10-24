# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from torch import nn
from transformers import (XLMRobertaTokenizer, XLMRobertaModel, BertConfig,
                          get_linear_schedule_with_warmup, BertPreTrainedModel,
                          RobertaConfig,
                          get_cosine_schedule_with_warmup,
                          LayoutLMv2Model, LayoutLMv2Config,
                          AutoTokenizer, AutoConfig)
import re
from transformers import (ViTConfig, LayoutLMv2FeatureExtractor, 
                          LayoutLMv2TokenizerFast, LayoutLMv2Processor)
from models import LayoutLMvForSequenceClassification
from models import ImageFeatureExtractor, CustomDataset
import random
import os
from tqdm import tqdm
import argparse
import json
from torch.utils.data import (Dataset, DataLoader, 
                              RandomSampler, SequentialSampler)
import torch
import transformers
from sklearn import metrics
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)
manualSeed = 2021

#from transformers import ViTForImageClassification

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Defining some key variables that will be used later on in the training
MAX_LEN = 256
LEARNING_RATE = 2e-05

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default="data/train.txt",
                    help='Train file (train.txt)')
parser.add_argument('--test_file', type=str, default="data/test.txt",
                    help='Test file (test.csv)')
parser.add_argument(
    '--out',
    type=str,
    default='experiments',
    help='The path of the directory where the *.csv data will be saved')
parser.add_argument('--val_steps', type=int, default=1000,
                    help='val_steps')
parser.add_argument('--batch_size', type=int, default=8,
                    help='val_steps')
parser.add_argument(
    "--adam_beta1",
    default=0.9,
    type=float,
    help="BETA1 for Adam optimizer.")
parser.add_argument(
    "--adam_beta2",
    default=0.999,
    type=float,
    help="BETA2 for Adam optimizer.")
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    help="Set this flag if you are using an uncased model.")
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight decay if we apply some.")
parser.add_argument("--local_rank", type=int, default=-
                    1, help="For distributed training: local_rank")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--num_train_epochs",
    default=3.0,
    type=float,
    help="Total number of training epochs to perform.")
parser.add_argument(
    "--no_cuda",
    action="store_true",
    help="Avoid using CUDA when available")
parser.add_argument(
    "--do_train",
    action="store_true",
    help="Whether to run training.")
parser.add_argument(
    "--do_eval",
    action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do_predict",
    action="store_true",
    help="Whether to run predictions on the test set.")
parser.add_argument(
    "--learning_rate",
    default=5e-5,
    type=float,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--cache_dir",
    default="temp",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list ",
)
parser.add_argument(
    "--adam_epsilon",
    default=1e-8,
    type=float,
    help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--warmup_steps",
    default=0,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=50,
    help="Save checkpoint every X updates steps.")

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file
output_dir = args.out
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.batch_size


# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1

args.device = device

print('device', device)

train_df = pd.read_csv(train_file, sep='\t')
test_df = pd.read_csv(test_file, sep='\t')

print(train_df.head())


def text_cleaner(text):
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')

    text = re.sub(r"@\w*", " ", str(text)).strip()  # removing username
    text = re.sub(r'https?://[A-Za-z0-9./]+', " ",
                  str(text)).strip()  # removing links
    text = re.sub(r'[^a-zA-Z]', " ", str(text)).strip()  # removing sp_char
    tw = []

    for text in text.split():
        if text not in stopwords:
            if not tw.startwith('@') and tw != 'RT':
                tw.append(text)
#    print( " ".join(tw))
    tw = re.sub(r"\s+", '-', ' '.join(tw))
    return tw


print("TRAIN Dataset: {}".format(train_df.shape))
print("TEST Dataset: {}".format(test_df.shape))


class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def loss_fct(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets.view(-1, 1))


def evaluation(model, epoch):
    model.eval()
    fin_targets_priorities = []
    fin_outputs_priorities = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader),
                            total=len(testing_loader)):
            ids = data['input_ids'].to(device, dtype=torch.long)
            ids_text = data['ids_text'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            mask_text = data['mask_text'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            token_type_ids_text = data['token_type_ids_text'].to(
                device, dtype=torch.long)
            bbox = data['bbox'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            pixel_values = data['image'].to(device, dtype=torch.float)

            logits = model(input_ids=ids,
                           input_ids_text=ids_text,
                           bbox=bbox,
                           image=pixel_values,
                           attention_mask=mask,
                           token_type_ids=token_type_ids,
                           attention_mask_text=mask_text,
                           token_type_ids_text=token_type_ids_text)

            fin_outputs_priorities.extend(
                torch.argmax(
                    nn.Softmax(
                        dim=1)(logits), -1).cpu().detach().numpy().tolist())
            print(nn.Softmax(
                        dim=1)(logits))
            fin_targets_priorities.extend(
                targets.cpu().detach().numpy().tolist())
            print(targets)
            print('-'*30)

    accuracy = metrics.accuracy_score(
        fin_targets_priorities,
        fin_outputs_priorities)
    f1_score_micro = metrics.f1_score(
        fin_targets_priorities, fin_outputs_priorities, average='micro')
    f1_score_macro = metrics.f1_score(
        fin_targets_priorities, fin_outputs_priorities, average='macro')

    print(f"Accuracy Score  = {accuracy}")
    print(f"F1 Score (Micro)= {f1_score_micro}")
    print(f"F1 Score (Macro)  = {f1_score_macro}")
    print(
        metrics.classification_report(
            fin_targets_priorities,
            fin_outputs_priorities,
            digits=4))
#
#    with open(os.path.join(output_dir, 'test.results.' + str(epoch) + '.json'), 'w', encoding='utf-8') as f:
#        json.dump(data, f, ensure_ascii=False, indent=2)

    return fin_targets_priorities, fin_outputs_priorities

from sadice import SelfAdjDiceLoss

criterion = SelfAdjDiceLoss()

def train(model, optimizer, scheduler, epoch):
    model.train()

    for step, data in tqdm(enumerate(training_loader),
                           total=len(training_loader)):

        #        dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
        ids = data['input_ids'].to(device, dtype=torch.long)
        ids_text = data['ids_text'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        mask_text = data['mask_text'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        token_type_ids_text = data['token_type_ids_text'].to(
            device, dtype=torch.long)
        bbox = data['bbox'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        pixel_values = data['image'].to(device, dtype=torch.float)

        logits = model(input_ids=ids,
                       input_ids_text=ids_text,
                       bbox=bbox,
                       image=pixel_values,
                       attention_mask=mask,
                       token_type_ids=token_type_ids,
                       attention_mask_text=mask_text,
                       token_type_ids_text=token_type_ids_text)

        optimizer.zero_grad()
        # CrossEntropyLoss

#        pclass_onehot = torch.zeros(targets.shape[0], 2)
#        targets = pclass_onehot.scatter_(1, targets.to(torch.int64).unsqueeze(1), 1.0)
#        print(targets)
#        loss = torch.nn.CrossEntropyLoss()(logits, targets)
#        loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
        # torch.nn.functional torch.nn.CrossEntropyLoss()
#        loss = torch.nn.BCEWithLogitsLoss()(logits.view(-1), targets)

#        import pdb;pdb.set_trace()
        loss = criterion(logits, targets.to(torch.int64))
        
#        loss = torch.nn.MSELoss()(logits.view(-1), targets)
#        loss = loss_fn(torch.argmax(nn.Softmax(dim=1)(logits), -1).float(), targets)
#        print(loss)
#        loss = sum(losses)

        if step % args.val_steps == 0 and step > 0:
            evaluation(model, "train_step{}_epoch{}".format(step, epoch))
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
#
#        if step % args.save_steps == 0:
#            model_to_save = (
#                model.module if hasattr(model, "module") else model
#            )  # Take care of distributed/parallel training
# model_to_save.save_pretrained(output_dir)
#
#            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pth'))
#
#            tokenizer.save_pretrained(output_dir)
#            torch.save(
#                args, os.path.join(
#                    output_dir, "training_args.bin"))
#            torch.save(
#                model.state_dict(), os.path.join(
#                    output_dir, "model.pt"))
#            torch.save(
#                optimizer.state_dict(), os.path.join(
#                    output_dir, "optimizer.pt"))
#            torch.save(
#                scheduler.state_dict(), os.path.join(
#                    output_dir, "scheduler.pt"))
#            print(
#                "Saving optimizer and scheduler states to %s", output_dir)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


if args.do_train:

    #    tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased', do_lower_case=True, truncation=True)

    feature_extractor = ImageFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224')

    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer_image = LayoutLMv2TokenizerFast.from_pretrained(
        "microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer_image)

    tokenizer_text = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=True, truncation=True)
    training_set = CustomDataset(train_df, tokenizer_text, processor, MAX_LEN)
    testing_set = CustomDataset(test_df, tokenizer_text, processor, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)


#    config = RobertaConfig.from_pretrained(args.model_name_or_path,
#        cache_dir=args.cache_dir if args.cache_dir else None,
#    )

#    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
#    config_image = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    config_image = LayoutLMv2Config.from_pretrained(
        'microsoft/layoutlmv2-base-uncased')
    config_text = AutoConfig.from_pretrained(args.model_name_or_path)
#    model = TextImageClassification(config_image, config, args)
    model = LayoutLMvForSequenceClassification(config_image, config_text, args)
#    model = BERTClass.from_pretrained(
#        args.model_name_or_path,
#        from_tf=bool(".ckpt" in args.model_name_or_path),
#        config=config,
#        cache_dir=args.cache_dir if args.cache_dir else None,
#    )

    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": args.weight_decay, }, {
            "params": [
                p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0}, ]

    t_total = len(
        training_loader) // args.gradient_accumulation_steps * args.num_train_epochs
#
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(
            args.adam_beta1,
            args.adam_beta2))

#    optimizer = create_optimizer(model)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

#    scheduler = get_cosine_schedule_with_warmup(
#            optimizer,
#            num_training_steps=t_total,
#            num_warmup_steps=args.warmup_steps)


    for epoch in range(int(args.num_train_epochs)):
        train(model, optimizer, scheduler, epoch)
        _, _ = evaluation(model, epoch)

#if args.do_predict:
#
#    config = RobertaConfig.from_pretrained(
#        args.model_name_or_path,
#        cache_dir=args.cache_dir if args.cache_dir else None,
#    )
#    tokenizer = AutoTokenizer.from_pretrained(
#        output_dir,
#        do_lower_case=args.do_lower_case)
#
#    testing_set = CustomDataset(test_df, tokenizer, MAX_LEN)
#
#    test_params = {'batch_size': VALID_BATCH_SIZE,
#                   'shuffle': False,
#                   'num_workers': 10
#                   }
#
#    testing_loader = DataLoader(testing_set, **test_params)
#    model = ViTForImageClassification(config)
#    model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pth')))
#    model.to(args.device)
#    if args.n_gpu > 1:
#        model = torch.nn.DataParallel(model)
#    evaluation(model, 'final', None, None)

# for epoch in range(EPOCHS):
#    import pdb;pdb.set_trace()
