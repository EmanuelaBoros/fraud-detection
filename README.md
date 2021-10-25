# fraud detection

### 1st model: [LayoutLMV2](https://huggingface.co/transformers/model_doc/layoutlmv2.html)
The LayoutLMV2 model was proposed in [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/pdf/2012.14740.pdf) by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMV2 improves LayoutLM to obtain state-of-the-art results across several document image understanding benchmarks:

  - information extraction from _scanned_ documents: the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset (a collection of 199 annotated forms comprising more than 30,000 words), the [CORD](https://github.com/clovaai/cord) dataset (a collection of 800 _receipts_ for training, 100 for validation and 100 for testing), the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset (a collection of 626 _receipts_ for training and 347 receipts for testing) and the [Kleister-NDA](https://github.com/applicaai/kleister-nda) dataset (a collection of _non-disclosure agreements_ from the EDGAR database, including 254 documents for training, 83 documents for validation, and 203 documents for testing).
  - document image classification: the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset (a collection of 400,000 digitized images belonging to one of 16 classes).
  - document visual question answering: the [DocVQA](https://arxiv.org/abs/2007.00398) dataset (a collection of 50,000 questions defined on 12,000+ document images).

[Tips](https://huggingface.co/transformers/model_doc/layoutlmv2.html):

  - The main difference between LayoutLMv1 and LayoutLMv2 is that the latter incorporates visual embeddings during pre-training (while LayoutLMv1 only adds visual embeddings during fine-tuning).
  - LayoutLMv2 adds both a relative 1D attention bias as well as a spatial 2D attention bias to the attention scores in the self-attention layers. Details can be found on page 5 of the paper.
  - Demo notebooks on how to use the LayoutLMv2 model on RVL-CDIP, FUNSD, DocVQA, CORD can be found here.
  - LayoutLMv2 uses Facebook AI’s Detectron2 package for its visual backbone. See this link for installation instructions.
  - In addition to input_ids, forward() expects 2 additional inputs, namely image and bbox. The image input corresponds to the original document image in which the text tokens occur. The model expects each document image to be of size 224x224. This means that if you have a batch of document images, image should be a tensor of shape (batch_size, 3, 224, 224). This can be either a torch.Tensor or a Detectron2.structures.ImageList. You don’t need to normalize the channels, as this is done by the model. Important to note is that the visual backbone expects BGR channels instead of RGB, as all models in Detectron2 are pre-trained using the BGR format. The bbox input are the bounding boxes (i.e. 2D-positions) of the input text tokens. This is identical to LayoutLMModel. These can be obtained using an external OCR engine such as Google’s Tesseract (there’s a Python wrapper available). Each bounding box should be in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000 scale. 

### 2nd model: [Camembert](https://huggingface.co/camembert-base)

CamemBERT is a state-of-the-art language model for French based on the RoBERTa model. It is now available on HuggingFace in 6 different versions with varying number of parameters, amount of pretraining data and pretraining data source domains.
For further information or requests, please go to [Camembert Website](https://camembert-model.fr/).


### Model: LayoutLMV2+Camembert

#### [Dice loss](https://aclanthology.org/2020.acl-main.45.pdf) (for imbalanced datasets)

Dice loss attaches similar importance to false positives and false negatives, and is more immune to the data-imbalance issue.


### Install

```
pip -r install requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
