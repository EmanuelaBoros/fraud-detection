from transformers import (AutoTokenizer, XLMRobertaModel, BertConfig,
                          get_linear_schedule_with_warmup, BertPreTrainedModel,
                          XLMRobertaConfig,
                          get_cosine_schedule_with_warmup,
                          )
from transformers import ViTPreTrainedModel, ViTModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput)
from typing import List, Optional, Union
from transformers.file_utils import TensorType
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.file_utils import TensorType
from transformers.utils import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, LayoutLMv2Model


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, image_feature_extractor, max_len):

        self.data = dataframe
        self.text = self.data.text

        columns = [
            'id',
            'date',
            'heure',
            'entreprise',
            'adresse',
            'Ville',
            'code postal',
            'telephone',
            'siren',
            'produits : quantité, prix unitaire, somme prix',
            'nombre articles',
            'total',
            'paiement intermediaire : montant, moyen paiement, monnaie rendue',
            'montant paye']

        self.entities = ''
        for column in columns[1:]:
            self.entities += ' [' + column + '] ' + self.data[column]

        self.image = self.data.image
        self.targets = self.data.label
        self.string_targets = self.data.string_label

        self.max_len = max_len

        #########
        # apply_ocr is set to True by default
        self.image_feature_extractor = image_feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.targets[index]
        try:
            with open(self.text[index].replace('data/receipts_corpus/txt/', 'data/triples_split/all/').replace(
                    'data/receipts_corpus/forgedtxt/', 'data/triples_split/all/'), 'r') as f:
                text = str(f.read())
        except BaseException:
            print(self.text[index], 'Missing')
            with open(self.text[index], 'r') as f:
                text = str(f.read())

        text = text.strip()

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

        image = Image.open(self.image[index])
        input_image = self.image_feature_extractor(
            images=image,
            return_tensors="pt",
            max_length=self.max_len,
            padding='max_length',
            truncation=True)
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
            'ids_image': input_image.input_ids.squeeze(),
            'mask_image': input_image.attention_mask.squeeze(),
            'bbox': input_image.bbox.squeeze(),
            'targets': torch.tensor(
                label,
                dtype=torch.float),
            'image': input_image.image.squeeze()}


class LayoutLMvForSequenceClassification(torch.nn.Module):
    def __init__(self, config_image, config_text, args, MODEL_IMAGE):
        super().__init__()

        self.config = config_image

        self.num_labels = 1

        self.text_model = AutoModel.from_pretrained(args.model_name_or_path,
                                                    config=config_text)

        self.image_model = LayoutLMv2Model.from_pretrained(MODEL_IMAGE)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            self.text_model.config.hidden_size * 4,
            self.num_labels)

    def get_input_embeddings(self):
        return self.image_model.embeddings.word_embeddings

    def forward(
        self,
        ids_image=None,
        ids_text=None,
        bbox=None,
        image=None,
        mask_image=None,
        mask_text=None,
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

        if ids_image is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif ids_image is not None:
            input_shape = ids_image.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = ids_image.device if ids_image is not None else inputs_embeds.device

        element = self.text_model(
            ids_text,
            attention_mask=mask_text,
            token_type_ids=token_type_ids_text
        )

        _, text_outputs = element[0], element[1]

        text_outputs = self.dropout(text_outputs)

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * \
            self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self.image_model._calc_visual_bbox(
            self.config.image_feature_pool_shape, bbox, device, final_shape
        )

        visual_position_ids = torch.arange(
            0,
            visual_shape[1],
            dtype=torch.long,
            device=device).repeat(
            input_shape[0],
            1)

        initial_image_embeddings = self.image_model._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.image_model(
            input_ids=ids_image,
            bbox=bbox,
            image=image,
            attention_mask=mask_image,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if ids_image is not None:
            input_shape = ids_image.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        sequence_output, final_image_embeddings = outputs[0][:,
                                                             :seq_length], outputs[0][:, seq_length:]

        cls_final_output = sequence_output[:, 0, :]

        # average-pool the visual embeddings
        pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
        # concatenate with cls_final_output

        sequence_output = torch.cat([cls_final_output,
                                     pooled_initial_image_embeddings,
                                     pooled_final_image_embeddings,
                                     text_outputs],
                                    dim=1)

        logits = self.classifier(sequence_output)

        return logits
