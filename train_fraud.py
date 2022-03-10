# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from torch import nn
from transformers import (get_linear_schedule_with_warmup, 
                          get_cosine_schedule_with_warmup,
                          AutoTokenizer, AutoConfig)
from models_fraud import LayoutLMv2Processor#, LayoutLMv2Tokenizer
#from tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast
#        from transformers import LayoutLMv2TokenizerFast, LayoutXLMTokenizerFast

#from tokenization_layoutxlm import LayoutXLMTokenizer
#from tokenization_layoutxlm_fast import LayoutXLMTokenizerFast
from transformers import LayoutLMv2FeatureExtractor        

from models_fraud import LayoutLMvForSequenceClassification
from models_fraud import CustomDataset
import random
import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import torch
from sklearn import metrics
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)
manualSeed = 3435

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

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default="data/train.txt",
                    help='Train file (train.txt)')
parser.add_argument('--test_file', type=str, default="data/test.txt",
                    help='Test file (test.csv)')
parser.add_argument('--valid_file', type=str, default="data/valid.txt",
                    help='Test file (test.csv)')
parser.add_argument('--max_length', type=int, default=256,
                    help='Maximum text length')
parser.add_argument('--out_dir', type=str,  default='runs',
    help='The path of the directory where the data/models will be saved')
parser.add_argument('--val_steps', type=int, default=10000,
                    help='val_steps')
parser.add_argument('--batch_size', type=int, default=8,
                    help='val_steps')
parser.add_argument("--adam_beta1", default=0.9, type=float,
    help="BETA1 for Adam optimizer.")
parser.add_argument("--adam_beta2", default=0.999, type=float,
    help="BETA2 for Adam optimizer.")
parser.add_argument("--do_lower_case", action="store_true",
    help="Set this flag if you are using an uncased model.")
parser.add_argument("--weight_decay", default=0.0, type=float,
    help="Weight decay if we apply some.")
parser.add_argument("--local_rank", type=int, default=-1, 
                    help="For distributed training: local_rank")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--num_train_epochs", default=4.0, type=float,
    help="Total number of training epochs to perform.")
parser.add_argument("--no_cuda", action="store_true",
    help="Avoid using CUDA when available")
parser.add_argument("--do_train", action="store_true",
    help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action="store_true",
    help="Whether to run predictions on the test set.")
parser.add_argument("--learning_rate", default=2e-05, type=float,
    help="The initial learning rate for Adam.")
parser.add_argument("--cache_dir", default="temp", type=str,
    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    help="Path to pre-trained model or shortcut name selected in the list ")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument("--save_steps", type=int, default=50,
    help="Save checkpoint every X updates steps.")

args = parser.parse_args()
train_file = args.train_file
test_file = args.test_file
valid_file = args.valid_file
output_dir = args.out_dir
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.batch_size
MAX_LEN = args.max_length
LEARNING_RATE = args.learning_rate

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
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

print('Device', device)


def evaluation(model, epoch, test_df, loader):
    
    model.eval()
    
    fin_targets = []
    fin_outputs = []
    fin_probabilities = []
    
    print('EPOCH:', epoch)
    
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader),
                            total=len(loader)):
            
            ids_image = data['ids_image'].to(device, dtype=torch.long)
            ids_text = data['ids_text'].to(device, dtype=torch.long)
            
            mask_image = data['mask_image'].to(device, dtype=torch.long)
            mask_text = data['mask_text'].to(device, dtype=torch.long)
            
            token_type_ids_text = data['token_type_ids_text'].to(device, dtype=torch.long)
            
            bbox = data['bbox'].to(device, dtype=torch.long)
            
            targets = data['targets'].to(device, dtype=torch.float)
            
            pixel_values = data['image'].to(device, dtype=torch.float)
    
            logits = model(ids_image=ids_image,
                           ids_text=ids_text,
                           bbox=bbox,
                           image=pixel_values,
                           mask_image=mask_image,
                           mask_text=mask_text,
                           token_type_ids_text=token_type_ids_text)
            
            fin_outputs += logits.reshape(-1).cpu().detach().numpy().tolist()

            fin_probabilities.extend(torch.max(nn.Sigmoid()(logits), dim=1).values.cpu().detach().numpy().tolist())
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())

    from scipy.stats import pearsonr
    pearsonr_corr = pearsonr(fin_outputs, fin_targets)
    
    print("Pearson-r:", pearsonr_corr[0])
    print("p-value:", pearsonr_corr[1])

    roc_auc_score_micro = metrics.roc_auc_score(fin_targets, fin_probabilities,
                                          average='micro')
    roc_auc_score_macro = metrics.roc_auc_score(fin_targets, fin_probabilities, 
                                          average='macro')

    print(f"ROC AUC (Micro) = {roc_auc_score_micro}")
    print(f"ROC AUC (Macro) = {roc_auc_score_macro}")
    
    fin_outputs = np.array(fin_outputs)
    print(fin_targets[:10])
    print(fin_outputs[:10])
    
    try:
        fin_outputs = np.where(fin_outputs < np.mean(fin_outputs), 0, fin_outputs)
        fin_outputs = np.where(fin_outputs >= np.mean(fin_outputs), 1, fin_outputs)
        print(fin_outputs[:10])
        print(metrics.classification_report(fin_targets, fin_outputs, digits=4))#, target_names=['Normal', 'Fraud']))
    except:
        pass
    with open(os.path.join(output_dir, 'test.results.' + str(epoch) + '.txt'), 'w', encoding='utf-8') as f:
        for y_true, y_predicted, y_proba, image_path, text_path in zip(fin_targets, 
                                       fin_outputs,
                                       fin_probabilities,
                                       test_df.image, test_df.text):
            f.write(image_path + '\t' + text_path + '\t' + 
                    str(int(y_true)) + '\t' + str(y_predicted) + '\t' + str(y_proba) + '\n')
        
    return fin_targets, fin_outputs

def loss_fn(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets.view(-1, 1))

def train(model, optimizer, scheduler, epoch, tokenizer_text, processor, test_df, valid_df, testing_loader, valid_loader):

    model.train()

    for step, data in tqdm(enumerate(training_loader),
                           total=len(training_loader)):

        ids_image = data['ids_image'].to(device, dtype=torch.long)
        ids_text = data['ids_text'].to(device, dtype=torch.long)
        
        mask_image = data['mask_image'].to(device, dtype=torch.long)
        mask_text = data['mask_text'].to(device, dtype=torch.long)
        
        token_type_ids_text = data['token_type_ids_text'].to(device, dtype=torch.long)
        
        bbox = data['bbox'].to(device, dtype=torch.long)
        
        targets = data['targets'].to(device, dtype=torch.float)
        
        pixel_values = data['image'].to(device, dtype=torch.float)

        logits = model(ids_image=ids_image,
                       ids_text=ids_text,
                       bbox=bbox,
                       image=pixel_values,
                       mask_image=mask_image,
                       mask_text=mask_text,
                       token_type_ids_text=token_type_ids_text)

        optimizer.zero_grad()
        
        loss = loss_fn(logits, targets)

        if step % args.val_steps == 0 and step > 0:
            evaluation(model, "test_step{}_epoch{}".format(step, epoch), test_df, testing_loader)
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        if step % args.save_steps == 0:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pth'))

            tokenizer_text.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            torch.save(
                args, os.path.join(
                    output_dir, "training_args.bin"))
            torch.save(
                model.state_dict(), os.path.join(
                    output_dir, "model.pt"))
            torch.save(
                optimizer.state_dict(), os.path.join(
                    output_dir, "optimizer.pt"))
            torch.save(
                scheduler.state_dict(), os.path.join(
                    output_dir, "scheduler.pt"))
            print(
                "Saving optimizer and scheduler states to %s", output_dir)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

if __name__ == '__main__':
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    valid_df = pd.read_csv(valid_file, sep='\t')
    
    train_df['id'] = train_df['text'].apply(lambda x: int(x[x.index('txt/')+4:-4]))
    test_df['id'] = test_df['text'].apply(lambda x: int(x[x.index('txt/')+4:-4]))
    valid_df['id'] = valid_df['text'].apply(lambda x: int(x[x.index('txt/')+4:-4]))
    
    print(train_df.head())
    
    entities_df = pd.read_csv('data/entities.csv', sep='\t')
    columns = ['Nom Ticket', 'date', 'heure', 'entreprise', 'adresse', 'Ville',
       'code postal', 'telephone', 'siren',
       'produits : quantité, prix unitaire, somme prix', 'nombre articles',
       'total',
       'paiement intermediaire : montant, moyen paiement, monnaie rendue',
       'montant paye']
    
    columns = ['id', 'date', 'heure', 'entreprise', 'adresse', 'Ville',
       'code postal', 'telephone', 'siren',
       'produits : quantité, prix unitaire, somme prix', 'nombre articles',
       'total', 'paiement intermediaire : montant, moyen paiement, monnaie rendue',
       'montant paye']
    
    entities_df.columns = columns
    
    entities_df['produits : quantité, prix unitaire, somme prix'] = \
        entities_df['produits : quantité, prix unitaire, somme prix'].apply(lambda x: ' '.join([y.lower().replace('_', ' ') + ' '.join([str(price) for price in list(z)]) for y, z in zip(eval(x).keys(), eval(x).values())]))
    
    entities_df['nombre articles'].fillna((entities_df['nombre articles'].mean()), inplace = True)
    
    entities_df['paiement intermediaire : montant, moyen paiement, monnaie rendue'] = \
        entities_df['paiement intermediaire : montant, moyen paiement, monnaie rendue'].apply(lambda x: ' '.join([ ' '.join([str(price) for price in list(z)]) for y, z in zip(eval(x).keys(), eval(x).values())]))
    
    entities_df['montant paye'].fillna((entities_df['montant paye'].mean()), inplace = True)
    
    for column in ['date', 'heure', 'entreprise', 'adresse', 'Ville',
       'code postal', 'telephone', 'siren',
       'produits : quantité, prix unitaire, somme prix', 'nombre articles',
       'total', 'paiement intermediaire : montant, moyen paiement, monnaie rendue',
       'montant paye']:
        entities_df[column] = entities_df[column].apply(lambda x: str(x).replace('_', ' '))
 
    print(entities_df.head())
        
    print(train_df.head())
    
    train_df = train_df.merge(entities_df, left_on='id', right_on='id')
    test_df = test_df.merge(entities_df, left_on='id', right_on='id')
    valid_df = valid_df.merge(entities_df, left_on='id', right_on='id')
    
    print("TRAIN Dataset: {}".format(train_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    print("VALID Dataset: {}".format(valid_df.shape))
    
    print(train_df.head())

    if args.do_train:
        from transformers import LayoutLMv2Tokenizer

        MODEL_IMAGE = 'microsoft/layoutlmv2-base-uncased'

        feature_extractor = LayoutLMv2FeatureExtractor()
        tokenizer_image = LayoutLMv2Tokenizer.from_pretrained(MODEL_IMAGE)
        processor = LayoutLMv2Processor(feature_extractor, tokenizer_image)
        
        tokenizer_text = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False, truncation=True)
        
        training_set = CustomDataset(train_df, tokenizer_text, processor, MAX_LEN)
        testing_set = CustomDataset(test_df, tokenizer_text, processor, MAX_LEN)
        valid_set = CustomDataset(valid_df, tokenizer_text, processor, MAX_LEN)


        # baseline ------------------------------------------
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        text_transformer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=150000)
        
        def get_text(path):
            try:
                with open(path.replace('data/receipts_corpus/txt/', 'data/triples_split/all/').replace('data/receipts_corpus/forgedtxt/', 'data/triples_split/all/'), 'r') as f:
                    text = str(f.read())
            except:
                print(path)
                with open(path, 'r') as f:
                    text = str(f.read())
                    
            text = text.replace('_', ' ').strip()
            return text.strip()
        
        data_train = [get_text(path) for path in training_set.text]
        data_test = [get_text(path) for path in testing_set.text]

        X_train_text = text_transformer.fit_transform(data_train)
        X_test_text = text_transformer.transform(data_test)
        
        logit = LogisticRegression(random_state=17, n_jobs=4)
        logit.fit(X_train_text, training_set.targets)
        
        test_preds = logit.predict(X_test_text)
        
        print(metrics.classification_report(testing_set.targets, test_preds, digits=4))
        
        from sklearn.svm import LinearSVR
        logit = LinearSVR()
        logit.fit(X_train_text, training_set.targets)
        
        test_preds = logit.predict(X_test_text)

        test_preds = np.where(test_preds < np.mean(test_preds), 0, test_preds)
        test_preds = np.where(test_preds >= np.mean(test_preds), 1, test_preds)
        test_preds = np.where(test_preds < np.mean(test_preds), 0, test_preds)
        
        print(metrics.classification_report(testing_set.targets, test_preds, digits=4))
        # baseline ------------------------------------------


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
        valid_loader = DataLoader(valid_set, **test_params)

        config_image = AutoConfig.from_pretrained(MODEL_IMAGE) # config for image
        config_text = AutoConfig.from_pretrained(args.model_name_or_path) # config for language model - camembert

        model = LayoutLMvForSequenceClassification(config_image, config_text, args, MODEL_IMAGE)
    
        model.to(device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        t_total = len(training_loader) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer = torch.optim.AdamW(model.parameters(),
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=(
                args.adam_beta1,
                args.adam_beta2))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total)

        for epoch in range(int(args.num_train_epochs)):
            train(model, optimizer, scheduler, epoch, 
                  tokenizer_text, processor, test_df, valid_df, testing_loader, valid_loader)
            _, _ = evaluation(model, epoch, test_df, testing_loader)
    
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
