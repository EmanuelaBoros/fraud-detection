from transformers import (AutoTokenizer, XLMRobertaModel, BertConfig,
                          get_linear_schedule_with_warmup, BertPreTrainedModel,
                          XLMRobertaConfig,
                          get_cosine_schedule_with_warmup,
                          )
from transformers import ViTPreTrainedModel, ViTModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPooling,
                                           SequenceClassifierOutput)
from typing import List, Optional, Union

from typing import List, Optional, Union

from transformers.file_utils import TensorType
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
#from transformers.models.layoutlmv2.feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor
#from transformers.models.layoutlmv2.tokenization_layoutlmv2 import LayoutLMv2Tokenizer
#from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

from typing import List, Optional, Union

from transformers.file_utils import TensorType
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
#from transformers.models.layoutlmv2.feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor
#from transformers.models.layoutlmv2.tokenization_layoutlmv2 import LayoutLMv2Tokenizer
#from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

import numpy as np
from PIL import Image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.file_utils import TensorType
#from transformers.image_utils import (IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, 
#                                      ImageFeatureExtractionMixin, is_torch_tensor)
from transformers.utils import logging
import torch
from torch.utils.data import Dataset
#from transformers import (LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, 
#                          AutoModel)
from transformers import AutoModel
import spacy
from spacy.lang.fr.examples import sentences 

nlp = spacy.load("en_core_web_sm")
nlp_french = spacy.load("fr_core_news_md")

import re
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import copy

from transformers import pipeline#, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
summarizer = pipeline("summarization", model="t5-large", tokenizer="t5-large")
def get_summary(text):
    summary = summarizer(str(text), min_length=5, max_length=128)[0]['summary_text']
    print(summary)
    return summary


import yake
import math
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=300)#, features=None)

def extract_keywords(text, to_lower=True):
    if to_lower:
        text = text.lower()
    keywords = dict(kw_extractor.extract_keywords(text))
    #processed_keywords = " ".join(keywords.keys())
    processed_keywords = {}
    for (keyword, score) in keywords.items():
        #score = 2-score
        score = -math.log(score)
        if score <= 0.0:
            continue
        processed_keywords[keyword] = score
    return ' '.join(processed_keywords.keys())

def tweet_cleaner(tweet):
#    import pdb;pdb.set_trace()
    tweet = re.sub(r"@\w*", " ", str(tweet)).strip() #removing username
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ", str(tweet)).strip() #removing links
    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet)).strip() #removing sp_char
#    tw = []
#    
#    for text in tweet.split():
#        if text not in stopwords:
#            if not tw.startwith('@') and tw != 'RT':
#                tw.append(text)
#    print( " ".join(tw))
    tweet = re.sub(r"\s+", ' ', tweet)
    return tweet

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords)


from enchant.checker import SpellChecker

spell_checker = SpellChecker("en_US")

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and len(word) > 2) # remove stopwors from text
    text = re.sub(r'\s+', ' ', text)
    
#    text += ' ' + extract_keywords(text)
    
#    correctwords = [w for w in  text.split() if spell_checker.check(w)]
    
    
#    entities = []
#    doc = nlp(text)
#    for ent in doc.ents:
#        if ent.text not in entities:
##            entities.append(' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
#            entities.append(ent.text)# + ' ' + ent.label_)
#            text = text.replace(ent.text, ' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
#            text = text.replace(ent.text, ' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> ')
#  
#    
#    text = '[TEXT] ' + text + ' [ENTITIES] ' + ' '.join(entities)
#    text = ' '.join(correctwords)
    
#    return ' '.join(entities)
    return text


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, image_feature_extractor, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.text
        columns = ['id', 'date', 'heure', 'entreprise', 'adresse', 'Ville',
       'code postal', 'telephone', 'siren',
       'produits : quantité, prix unitaire, somme prix', 'nombre articles',
       'total', 'paiement intermediaire : montant, moyen paiement, monnaie rendue',
       'montant paye']
        
        self.entities = ''
        for column in columns[1:]:
            self.entities += ' [' + column + '] ' + self.data[column]
            self.entities += ' ' + self.data[column]
            
#        import pdb;pdb.set_trace()
        
        self.image = self.data.image
        self.targets = self.data.label
        self.string_targets = self.data.string_label

        self.max_len = max_len
        self.image_feature_extractor = image_feature_extractor # apply_ocr is set to True by default

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        label = self.targets[index]
        
#        with open(self.text[index], 'r') as f:
#        text = str(self.text[index])
        # with open(self.text[index], 'r') as f:
        #     text = str(f.read())
        try:
            with open(self.text[index].replace('data/receipts_corpus/txt/', 'data/triples_split/all/').replace('data/receipts_corpus/forgedtxt/', 'data/triples_split/all/'), 'r') as f:
                text = str(f.read())
        except:
            print(self.text[index], 'Missing')
            with open(self.text[index], 'r') as f:
                text = str(f.read())
#            text = 'None'
            
        text = text.strip()
#        text = clean_text(text)
        
        # text = text + ' '  + self.entities[index].replace('_', ' ').strip()
        # text = text.replace('_', ' ').strip()
        # print(text)
##        image_path = Image.open('data/forged/3.jpg')
##        print(index, '--'*10)
        image_path = Image.open(self.image[index])
###        import pdb;pdb.set_trace()
#        if len(text) < 10:
#            print(index, self.targets[index], ':', text)
#        doc = nlp(text)
###        doc_french = nlp_french(text)
##        
###        print(text)
###        for token in doc:
###            print(token.text, token.pos_, token.dep_)
###        text = ''
      
#        text_preprocessed = get_summary(text)
#        text_preprocessed = ''
#        for token in doc: 
#            text_preprocessed += ' ' + token.lemma_ + " " + token.dep_ + " " + token.pos_ + " " + token.tag_ + ' '
#        text_preprocessed = extract_keywords(text)
#        for ent in doc.ents:
#            text_preprocessed += ' ' + ent.text + ' ' + ent.label_.lower() 
#            
#        text_preprocessed = re.sub(r"\s+", ' ', text_preprocessed)
#        text_entities = ''
#        entities = []
#        for ent in doc.ents:
#            if ent.text not in entities:
#                entities.append(ent.text)
#                text_entities += ' <' + ent.label_.lower().replace('_', ' ') + '> ' + ent.text + ' '  + ' </' + ent.label_.lower().replace('_', ' ') + '> '

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            padding='max_length', 
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        # input_image = self.image_feature_extractor(images=image_path, return_tensors="pt", 
        #                                            max_length=self.max_len,
        #  padding='max_length', truncation=True)
        # input_image = self.image_feature_extractor(
        #     images=image_path,
        #     return_tensors="pt",
        #     max_length=self.max_len,
        #     padding='max_length',
        #     truncation=True)
        
        return {
            'ids_text': torch.tensor(
                ids,
                dtype=torch.long),
            'mask_text': torch.tensor(
                mask,
                dtype=torch.long),
            'token_type_ids_text': torch.tensor(
                token_type_ids,
                dtype=torch.long),
            'input_ids': torch.tensor(
                ids,
                dtype=torch.long),
            'token_type_ids': torch.tensor(
                ids,
                dtype=torch.long),
            'attention_mask': torch.tensor(
                ids,
                dtype=torch.long),
            'bbox': torch.tensor(
                ids,
                dtype=torch.long),
            'targets': torch.tensor(
                label,
                dtype=torch.long),
            'image': torch.tensor(
                ids,
                dtype=torch.long),}
        
#         return {
#             'ids_text': torch.tensor(
#                 ids,
#                 dtype=torch.long),
#             'mask_text': torch.tensor(
#                 mask,
#                 dtype=torch.long),
#             'token_type_ids_text': torch.tensor(
#                 token_type_ids,
#                 dtype=torch.long),
#             'input_ids': input_image.input_ids.squeeze(),
# #            'token_type_ids': input_image.token_type_ids.squeeze(),
#             'token_type_ids': torch.tensor(
#                 token_type_ids,
#                 dtype=torch.long),
#             'attention_mask': input_image.attention_mask.squeeze(),
#             'bbox': input_image.bbox.squeeze(),
#             'targets': torch.tensor(
#                 label,
#                 dtype=torch.float),
#             'image': input_image.image.squeeze()}


class BERTClass(torch.nn.Module):

    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, args):
        super().__init__()
        config = XLMRobertaConfig.from_pretrained(args.model_name_or_path)

        self.bert = XLMRobertaModel.from_pretrained(args.model_name_or_path,
                                                    config=config)
        self.dropout = nn.Dropout(0.3)

        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, ids, mask, token_type_ids, pixel_values):

        element = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)

        bert_outputs = element[1]

        output = self.dropout(bert_outputs)

        logits = self.linear(output)
        return logits


# class ViTForImageClassification(ViTPreTrainedModel):
class TextImageClassification(torch.nn.Module):
    def __init__(self, config_image, config_text, args):
        super().__init__()

        self.num_labels = config_image.num_labels

        self.config = config_image
#        config_image = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.layoutlmv2 = LayoutLMv2Model(config_image)

        config_text = LayoutLMv2Config.from_pretrained(args.model_name_or_path)
        self.bert = LayoutLMv2Model.from_pretrained(args.model_name_or_path,
                                                    config=config_text)
        # Classifier head
        self.classifier = nn.Linear(
            config_image.hidden_size,
            config_image.num_labels) if config_image.num_labels > 0 else nn.Identity()

        self.dropout = nn.Dropout(0.3)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size * 2, 1)

        self.init_weights(self.vit)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        ids, mask, token_type_ids,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            >>> model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
#        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        element = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)

        bert_outputs = element[1]

        bert_outputs = self.dropout(bert_outputs)

        outputs = self.vit(pixel_values,
                           head_mask=head_mask,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           interpolate_pos_encoding=interpolate_pos_encoding,
                           return_dict=return_dict,
                           )

        sequence_output = outputs[0]

        all_features = torch.cat(
            [sequence_output[:, 0, :], bert_outputs], dim=1)

        logits = self.linear(all_features)

        return logits


#from transformers import LayoutLMv2LayerNorm
LayoutLMv2LayerNorm = torch.nn.LayerNorm

from transformers import LayoutLMv2Model, LayoutLMv2Config

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
    
n_heads = 12
head_dims = 128
d_model = n_heads * head_dims
num_layers = 2
feedforward_dim = int(2 * d_model)
trans_dropout = 0.45
attn_type = 'transformer'
after_norm = 1
fc_dropout = 0.4
scale = attn_type == 'transformer'
dropout_attn = None
pos_embed = 'fix'

from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer


# class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
class LayoutLMvForSequenceClassification(torch.nn.Module):
    def __init__(self, config_image, config_text, args, MODEL_IMAGE):
        super().__init__()

        self.config = config_image

        # self.num_labels = 2
        self.num_labels = 1
#        self.num_labels = 10

        self.bert = AutoModel.from_pretrained(args.model_name_or_path,
                                              config=config_text)

        self.layoutlmv2 = LayoutLMv2Model.from_pretrained(MODEL_IMAGE) 

#        self.dropout = nn.Dropout(config_text.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Linear(128 * 24, self.num_labels)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        self.head = AttentionHead(2 * config_text.hidden_size)

        self.transformer = TransformerEncoder(num_layers, d_model, n_heads,
                                              feedforward_dim, trans_dropout,
                                              after_norm=after_norm,
                                              attn_type=attn_type,
                                              scale=scale,
                                              dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
#        self.len_bert = 768
        self.in_fc = torch.nn.Linear(self.bert.config.hidden_size * (512+1), d_model)
        
        self.self_attn = MultiHeadAttn(
                d_model, n_heads, trans_dropout, scale=True)

        # self.init_weights(self.layoutlmv2)

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    # def init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, nn.Linear):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(
    #             mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(
    #             mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, LayoutLMv2LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids=None,
        input_ids_text=None,
        bbox=None,
        image=None,
        attention_mask=None,
        attention_mask_text=None,
        token_type_ids=None,
        token_type_ids_text=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import LayoutLMv2Processor, LayoutLMv2ForSequenceClassification
            >>> from PIL import Image
            >>> import torch

            >>> processor = LayoutLMv2Processor.from_pretrained('microsoft/layoutlmv2-base-uncased')
            >>> model = LayoutLMv2ForSequenceClassification.from_pretrained('microsoft/layoutlmv2-base-uncased')

            >>> image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")

            >>> encoding = processor(image, return_tensors="pt")
            >>> sequence_label = torch.tensor([1])

            >>> outputs = model(**encoding, labels=sequence_label)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        element = self.bert(
            input_ids_text,
            attention_mask=attention_mask_text,
            token_type_ids=token_type_ids_text
            )

        _, bert_outputs = element[0], element[1]
#        import pdb;pdb.set_trace()1
#        bert_outputs = torch.cat([bert_outputs.unsqueeze(1), doc], 1).view(bert_outputs.shape[0], -1)

        bert_outputs = self.dropout(bert_outputs)

#         visual_shape = list(input_shape)
#         visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
#         visual_shape = torch.Size(visual_shape)
#         final_shape = list(input_shape)
#         final_shape[1] += visual_shape[1]
#         final_shape = torch.Size(final_shape)

#         visual_bbox = self.layoutlmv2._calc_visual_bbox(
#             self.config.image_feature_pool_shape, bbox, device, final_shape
#         )

#         visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
#             input_shape[0], 1
#         )

#         initial_image_embeddings = self.layoutlmv2._calc_img_embeddings(
#             image=image,
#             bbox=visual_bbox,
#             position_ids=visual_position_ids,
#         )

#         outputs = self.layoutlmv2(
#             input_ids=input_ids,
#             bbox=bbox,
#             image=image,
#             attention_mask=attention_mask,
# #            token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         if input_ids is not None:
#             input_shape = input_ids.size()
#         else:
#             input_shape = inputs_embeds.size()[:-1]

#         seq_length = input_shape[1]
        
        
#         sequence_output, final_image_embeddings = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        
# #        print(outputs[0][:, seq_length:].shape)
# #        mask_doc = attention_mask.ne(0)
        
# #        transformer_output_text = self.transformer(sequence_output, mask_doc)

#         cls_final_output = sequence_output[:, 0, :]
# #        
# #        transformer_output_text = self.transformer(sequence_output, mask_doc)

#         # average-pool the visual embeddings
#         pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
#         pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
#         # concatenate with cls_final_output
# #        sequence_output = torch.cat(
# #            [cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings], dim=1
# #        )
#         sequence_output = torch.cat([cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings, bert_outputs], dim=1)

        sequence_output = bert_outputs
#        import pdb;pdb.set_trace()
        logits = self.classifier(sequence_output)
        
        return logits

from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, LayoutLMv2Tokenizer

class LayoutLMv2Processor:
    r"""
    Constructs a LayoutLMv2 processor which combines a LayoutLMv2 feature extractor and a LayoutLMv2 tokenizer into a
    single processor.

    :class:`~transformers.LayoutLMv2Processor` offers all the functionalities you need to prepare data for the model.

    It first uses :class:`~transformers.LayoutLMv2FeatureExtractor` to resize document images to a fixed size, and
    optionally applies OCR to get words and normalized bounding boxes. These are then provided to
    :class:`~transformers.LayoutLMv2Tokenizer` or :class:`~transformers.LayoutLMv2TokenizerFast`, which turns the words
    and bounding boxes into token-level :obj:`input_ids`, :obj:`attention_mask`, :obj:`token_type_ids`, :obj:`bbox`.
    Optionally, one can provide integer :obj:`word_labels`, which are turned into token-level :obj:`labels` for token
    classification tasks (such as FUNSD, CORD).

    Args:
        feature_extractor (:obj:`LayoutLMv2FeatureExtractor`):
            An instance of :class:`~transformers.LayoutLMv2FeatureExtractor`. The feature extractor is a required
            input.
        tokenizer (:obj:`LayoutLMv2Tokenizer` or :obj:`LayoutLMv2TokenizerFast`):
            An instance of :class:`~transformers.LayoutLMv2Tokenizer` or
            :class:`~transformers.LayoutLMv2TokenizerFast`. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
#        if not isinstance(feature_extractor, LayoutLMv2FeatureExtractor):
#            raise ValueError(
#                f"`feature_extractor` has to be of type {LayoutLMv2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
#            )
#        if not isinstance(tokenizer, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)):
#            raise ValueError(
#                f"`tokenizer` has to be of type {LayoutLMv2Tokenizer.__class__} or {LayoutLMv2TokenizerFast.__class__}, but is {type(tokenizer)}"
#            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def save_pretrained(self, save_directory):
        """
        Save a LayoutLMv2 feature_extractor object and LayoutLMv2 tokenizer object to the directory ``save_directory``,
        so that it can be re-loaded using the :func:`~transformers.LayoutLMv2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, use_fast=True, **kwargs):
        r"""
        Instantiate a :class:`~transformers.LayoutLMv2Processor` from a pretrained LayoutLMv2 processor.

        .. note::

            This class method is simply calling LayoutLMv2FeatureExtractor's
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` and
            LayoutLMv2TokenizerFast's
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.SequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.

            use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to instantiate a fast tokenizer.

            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.SequenceFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if use_fast:
            tokenizer = LayoutLMv2TokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            tokenizer = LayoutLMv2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        This method first forwards the :obj:`images` argument to
        :meth:`~transformers.LayoutLMv2FeatureExtractor.__call__`. In case :class:`~LayoutLMv2FeatureExtractor` was
        initialized with :obj:`apply_ocr` set to ``True``, it passes the obtained words and bounding boxes along with
        the additional arguments to :meth:`~transformers.LayoutLMv2Tokenizer.__call__` and returns the output, together
        with resized :obj:`images`. In case :class:`~LayoutLMv2FeatureExtractor` was initialized with :obj:`apply_ocr`
        set to ``False``, it passes the words (:obj:`text`/:obj:`text_pair`) and :obj:`boxes` specified by the user
        along with the additional arguments to :meth:`~transformers.LayoutLMv2Tokenizer.__call__` and returns the
        output, together with resized :obj:`images`.

        Please refer to the docstring of the above two methods for more information.
        """
        # verify input
        if self.feature_extractor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes "
                "if you initialized the feature extractor with apply_ocr set to True."
            )

        if self.feature_extractor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels "
                "if you initialized the feature extractor with apply_ocr set to True."
            )

        # first, apply the feature extractor
        features = self.feature_extractor(images=images, return_tensors=return_tensors)

        # second, apply the tokenizer
        if text is not None and self.feature_extractor.apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the feature extractor always adds a batch dimension)
            text_pair = features["words"]
        
#        text = [' '.join(features["words"][0])][0]
#        import pdb;pdb.set_trace()
        encoded_inputs = self.tokenizer(
            text=text if text is not None else features["words"],
            text_pair=text_pair if text_pair is not None else None,
            boxes=boxes if boxes is not None else features["boxes"],
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
#        print(text)
#        encoded_inputs = self.tokenizer.encode_plus(
#            text,
#            None,
#            add_special_tokens=True,
#            max_length=max_length,
#            pad_to_max_length=True,
#            return_token_type_ids=True,
#            padding='max_length', truncation=True
#        )
        # add pixel values
#        import pdb;pdb.set_trace()
        encoded_inputs["image"] = features.pop("pixel_values")

        return encoded_inputs
