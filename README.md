
# HOW TO RUN

**SHOPEE DATASET** (**SIZE:** 1.79GB)
https://www.kaggle.com/c/shopee-product-matching/data

- All the notebooks are fully executed and saved with results.

**List of pretrained and self-trained models used**

***Note:*** 
- Most models are greater than 100 MB, so cannot be stored in GITHUB.
- Self-trained model EfficientNetB6 is published in kaggle for public use **by our team**.
- Below models has to be downloaded for us by the notebooks.

https://www.kaggle.com/pytorch/resnet152

https://www.kaggle.com/kwisatzhaderach/glove2word2vec


**Trained for 34hours** 

Notebook:
[EfficientNet-B[3,5,6] Image + TRAINING.ipynb

https://github.com/sudha-vijayakumar/CMPE256-SHOPEE/blob/main/EfficientNet-B%5B3%2C5%2C6%5D%20Image%20%2B%20TRAINING.ipynb

published B6 model

https://www.kaggle.com/sudhavijayakumar/arcface-512x512-eff-b6-pt

Pre-trained models

https://www.kaggle.com/parthdhameliya77/shopee-pytorch-models

https://www.kaggle.com/kozodoi/timm-pytorch-image-models

- Recreate the below folder structure to run the notebooks successfully,

    Project Folder structure

    | - input

        | - shopee-product-matching data

        | - pretrained & self-trained models 

    | - output

        | -  submission_ensembled.csv

    | - Notebooks

# ML PIPELINES

![High-level](https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/report/DIAGRAMS/High-level.jpg?raw=True)

**FIND SIMILARITY BY PRODUCT IMAGE**
![Image](https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/report/DIAGRAMS/ImageProcessing.jpg?raw=True)

**FIND SIMILARITY BY PRODUCT TITLE**
![Text](https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/report/DIAGRAMS/TextProcessing.jpg?raw=True)

# PROJECT STORY & RESULTS 

**[1] Understanding the Problem - EDA + Phase + NLP**
      
***Notebook*** 
- https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B1%5D%20Understanding-the-problem-eda-phase-nlp.ipynb

***Goal***
- Data Exploration of the image dataset and the meta-data.

***Inferences***

There are multiple ways to find similar products, 

- Using title & Finding similar product titles.

- Using product images & Finding similar product images.

- Using image hash & Finding similar product images.

- Using image OCR Text matching & Finding similar product images.

***Step-2***
 - Will be trying baseline models to find similar images, similar titles.

**[2] RESNET152 + TFDIF**
      
***Notebook*** 
- https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B2%5D%20Resnet152%20%2B%20Tfdif.ipynb
      
***Goal***
- To use [RESNET152] pretrained model as the backbone to train SHOPEE image dataset for product matching by [IMAGE].

- To use [TF-DIF Vectorizer] sklearn model as the backone to train SHOPEE metadata for product matching by [Title].

- Combine predictions from Step[1] & Step[2]

***Inferences***
- Combined predictions using RESNET152along with TF-DIF Vectorizer provided better F1 score over stand-alone predictions.

***LEARNING***
- Using pretrained models to train the SHOPEE Dataset and make predictions(product match).

***Step-3***
- To replace [RESNET152] model with the EfficientNet [3,5,6] models and train them on the SHOPEE Dataset, to see if the current [F1] Score can be improved.

**[3] TRAIN & INFER using EfficientNetB[3,5,6]/ eca-nfnet-l0 Image+TFIDF**

***Notebook***
- https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B3.1%5D%20EffNetB%5B3%2C5%2C6%5D%20%2B%20nfNet10%20(Training).ipynb
- https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B3.2%5D%20EffNetB%5B3%2C5%2C6%5D%20%2B%20nfNet10%20%2B%20Tfdif(Inference).ipynb
                    
***Goal***
- To use EfficientNetB[3,5,6]/ eca-nfnet-l0 pretrained models as the backbone to train SHOPEE image dataset for product matching by [IMAGE].

- To use [TF-DIF Vectorizer] sklearn model as the backone to train SHOPEE metadata for product matching by [Title].

- Combine predictions from Step[1] (EfficientNet [3,5,6]/ eca-nfnet-l0 models individually)& Step[2].

***Inferences***
- Combined predictions using EfficientNetB6 along with TF-DIF Vectorizer provided better F1 score over EfficientNetB[3,5] / RESNET152 + TF-DIF predictions.

***LEARNING***
- How to train EfficientNetB[3,5,6]/ eca-nfnet-l0 on the SHOPEE dataset and how to stay calm :) P.S B6 training took 34 hours.

***Step-4***
- Ensemble [EfficientNetB6] + [eca-nfnet-10] to see if the [F1]-Score improves.

**[4] Ensemble (EffNetB6 + nfnet-l0) [Inference]**

***Notebook*** 
- https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B4%5D%20Ensemble%20Effnetb6%20%2B%20nfnet-10%20%2B%20Tfdif.ipynb

***Goal***
- To ensemble EfficientNetB[6] + eca-nfnet-10 pretrained models to make predictions on the SHOPEE image dataset for product matching by [IMAGE].

- To use [TF-DIF Vectorizer] sklearn model as the backone to train SHOPEE metadata for product matching by [Title].

- Combine predictions from Step[1] & Step[2].

***Inferences***
- Ensembled predictions using [EfficientNetB6] + [eca-nfnet-l0] along with TF-DIF Vectorizer provided better F1 score over EfficientNetB[3,5] / RESNET152 eca-nfnet-l0 [+] TF-DIF predictions.

***LEARNING***

How to ensemble different pretrained models to make predictions on the SHOPEE dataset.

**Step-5** (ONGOING)

- To try different ensembling with different neighbour threshhold to see if the score improves.

- To try [Fastai + Annoy] (Match by image) along with BART tokenizer(match by title) to find similar products.

- To try domain specific proven model [Large Scale Multimodal Classification Using an Ensemble of Transformer Models and Co-Attention](https://paperswithcode.com/paper/large-scale-multimodal-classification-using)

# Results
  https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/src/%5B5%5D%20Metric%20Evaluation.ipynb

## F1 SCORE
![Results](https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/report/DIAGRAMS/F1_SCORE_REPORT.png?raw=True)

## TRAINING TIME
![TIME](https://github.com/sudha-vijayakumar/CMPE256_SHOPEE_TERMPROJECT/blob/main/report/DIAGRAMS/Training_Time.png?raw=True)



