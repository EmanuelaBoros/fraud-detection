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
        
        columns = ['id', 'date', 'heure', 'entreprise', 'adresse', 'Ville',
       'code postal', 'telephone', 'siren',
       'produits : quantité, prix unitaire, somme prix', 'nombre articles',
       'total', 'paiement intermediaire : montant, moyen paiement, monnaie rendue',
       'montant paye']
        
        self.entities = ''
        for column in columns[1:]:
            self.entities += ' [' + column + '] ' + self.data[column]
        
        self.image = self.data.image
        self.targets = self.data.label
        self.string_targets = self.data.string_label

        self.max_len = max_len
        
        #########
        self.image_feature_extractor = image_feature_extractor # apply_ocr is set to True by default
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.targets[index]
        try:
            with open(self.text[index].replace('data/receipts_corpus/txt/', 'data/triples_split/all/').replace('data/receipts_corpus/forgedtxt/', 'data/triples_split/all/'), 'r') as f:
                text = str(f.read())
        except:
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
        input_image = self.image_feature_extractor(images=image, return_tensors="pt", 
                                                    max_length=self.max_len,
                                                    padding='max_length', truncation=True)
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

        self.classifier = nn.Linear(self.text_model.config.hidden_size * 4, self.num_labels)
        
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
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self.image_model._calc_visual_bbox(
            self.config.image_feature_pool_shape, bbox, device, final_shape
        )

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )

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
        
        sequence_output, final_image_embeddings = outputs[0][:, :seq_length], outputs[0][:, seq_length:]
        
        cls_final_output = sequence_output[:, 0, :]

        # average-pool the visual embeddings
        pooled_initial_image_embeddings = initial_image_embeddings.mean(dim=1)
        pooled_final_image_embeddings = final_image_embeddings.mean(dim=1)
        # concatenate with cls_final_output

        sequence_output = torch.cat([cls_final_output, pooled_initial_image_embeddings, pooled_final_image_embeddings, text_outputs], dim=1)

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
        encoded_inputs["image"] = features.pop("pixel_values")

        return encoded_inputs
