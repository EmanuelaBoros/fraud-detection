from transformers import (AutoTokenizer, XLMRobertaModel, BertConfig,
                          get_linear_schedule_with_warmup, BertPreTrainedModel,
                          XLMRobertaConfig,
                          get_cosine_schedule_with_warmup,
                          LayoutLMv2Model, LayoutLMv2Config)
from transformers import ViTPreTrainedModel, ViTModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput)
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.file_utils import TensorType
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    is_torch_tensor)
from transformers.utils import logging
import torch
from torch.utils.data import Dataset
from transformers import (LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast,
                          LayoutLMv2Processor, AutoModel)
import spacy
from spacy.lang.fr.examples import sentences

nlp = spacy.load("en_core_web_sm")


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, image_feature_extractor, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.text
        self.image = self.data.image
        self.targets = self.data.label

        self.max_len = max_len
        # apply_ocr is set to True by default
        self.image_feature_extractor = image_feature_extractor

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        with open(self.text[index], 'r') as f:
            text = str(f.read())

#        image_path = Image.open('data/forged/3.jpg')
        image_path = Image.open(self.image[index])
#        import pdb;pdb.set_trace()
#        doc = nlp(text)
#        for token in doc:
#            print(token.text, token.pos_, token.dep_)
#        text = ''
#        for ent in doc.ents:
#            print(ent.text, '-', ent.label_)
#            text += ent.text + ' [' + ent.label_ + '] '

#        print(text)
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            padding='max_length', truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

#        input_image = self.image_feature_extractor(images=image_path, return_tensors="pt", max_length=28,
        # padding='max_length', truncation=True)
        input_image = self.image_feature_extractor(
            images=image_path,
            return_tensors="pt",
            max_length=32,
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
            'input_ids': input_image.input_ids.squeeze(),
            'token_type_ids': input_image.token_type_ids.squeeze(),
            'attention_mask': input_image.attention_mask.squeeze(),
            'bbox': input_image.bbox.squeeze(),
            'targets': torch.tensor(
                self.targets[index],
                dtype=torch.float),
            'image': input_image.image.squeeze()}


class ImageFeatureExtractor(
        FeatureExtractionMixin,
        ImageFeatureExtractionMixin,
        Dataset):
    r"""
    Constructs a ViT feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int` or :obj:`Tuple(int)`, `optional`, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if :obj:`do_resize`
            is set to :obj:`True`.
        resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BOX`,
            :obj:`PIL.Image.BILINEAR`, :obj:`PIL.Image.HAMMING`, :obj:`PIL.Image.BICUBIC` or :obj:`PIL.Image.LANCZOS`.
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (:obj:`List[int]`, defaults to :obj:`[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`List[int]`, defaults to :obj:`[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        .. warning::

           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.

        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to :obj:`'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return NumPy :obj:`np.ndarray` objects.
                * :obj:`'jax'`: Return JAX :obj:`jnp.ndarray` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)
                      ) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(
                    images[0], (Image.Image, np.ndarray)) or is_torch_tensor(
                    images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples).")

        is_batched = bool(
            isinstance(
                images, (list, tuple)) and (
                isinstance(
                    images[0], (Image.Image, np.ndarray)) or is_torch_tensor(
                    images[0])))

        if not is_batched:
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            images = [
                self.resize(
                    image=image,
                    size=self.size,
                    resample=self.resample) for image in images]
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs


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

# class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):


class LayoutLMvForSequenceClassification(torch.nn.Module):
    def __init__(self, config_image, config_text, args):
        super().__init__()

        self.config = config_image

        self.num_labels = 2

        self.bert = AutoModel.from_pretrained(args.model_name_or_path,
                                              config=config_text)

        self.layoutlmv2 = LayoutLMv2Model.from_pretrained(
            'microsoft/layoutlmv2-base-uncased')

        self.dropout = nn.Dropout(config_text.hidden_dropout_prob)
        self.classifier = nn.Linear(
            4 * config_text.hidden_size, self.num_labels)

        self.init_weights(self.layoutlmv2)

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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
        elif isinstance(module, LayoutLMv2LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
            token_type_ids=token_type_ids_text)

        bert_outputs = element[1]

        bert_outputs = self.dropout(bert_outputs)

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * \
            self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self.layoutlmv2._calc_visual_bbox(
            self.config.image_feature_pool_shape, bbox, device, final_shape
        )

        visual_position_ids = torch.arange(
            0,
            visual_shape[1],
            dtype=torch.long,
            device=device).repeat(
            input_shape[0],
            1)

        initial_image_embeddings = self.layoutlmv2._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
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
                                     bert_outputs],
                                    dim=1)
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        return logits
