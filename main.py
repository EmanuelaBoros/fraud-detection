# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer, VisualBertModel

model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("What is the man eating?", return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path



#visual_embeds = get_visual_embeddings(image_path)
#
#visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
#visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
#inputs.update({
#    "visual_embeds": visual_embeds,
#    "visual_token_type_ids": visual_token_type_ids,
#    "visual_attention_mask": visual_attention_mask
#})
#outputs = model(**inputs)
#last_hidden_state = outputs.last_hidden_state


from models import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from transformers import ViTConfig

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

for image_path in ['data/1954_correct.jpg', 'data/1954_forged.jpg']:
    image = Image.open(image_path)

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.hidden_states)
    outputs.attentions
    import pdb;pdb.set_trace()
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])



#text_path = ''
#with open(text_path)