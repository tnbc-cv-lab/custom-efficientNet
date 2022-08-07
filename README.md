# custom-efficientNet

## Introduction
We have carried out transfer learning on an EfficientNetb0 network. The network was originally trained on the imagenet dataset and then on our custom datasets. We have three different datasets that were used for training which will be explained below:

### Choice_1
This dataset consists of two classes, whiteSpace and cellSpace. Different variants of the model were trained here with varying parameters and the statistics obtained for the different iterations are given below:

| Epochs | Finetune | Flatten Layer | Dense Layer | Accuracy | Validation Accuracy | Loss | Validation Loss | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 20 | False | False | False | 0.9338 | 0.9531 | 0.1952 | 0.1495 |
| 20 | True | False | False | 0.9798 | 0.9757 | 0.0653 | 0.0790 |
| 20 | False | True | False | 0.8888 | 0.9288 | 0.2938 | 0.212 |
| 20 | True | True | False | 0.9466 | 0.9531 | 0.1808 | 0.1518 |
| 20 | False | False | True | 0.9705 | 0.9670 | 0.0958 | 0.087 |
| 20 | True | False | True | 0.9761 | 0.9705 | 0.0722 | 0.0699 |
| 20 | False | True | True | 0.9724 | 0.9653 | 0.0793 | 0.0950 |
| 20 | True | True | True | 0.9789 | 0.9701 | 0.0622 | 0.0725 |

### Choice_2
This dataset consists of 5ive classes; Stroma, stromaTils, tumor, tumorTils, and whiteSpace derived from Dr. Madhura's labelled dataset. Different variants of the model were trained here with varying parameters and the statistics obtained for the different iterations are given below:

| Epochs | Finetune | Flatten Layer | Dense Layer | Accuracy | Validation Accuracy | Loss | Validation Loss | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 20 | False | False | False | 0.2917 | 0.2000 | 2.2637 | 2.0888 |
| 20 | True | False | False | 0.5083 | 0.4667 | 1.6657 | 1.6665 |
| 20 | False | True | False | 0.3167 | 0.1833 | 2.1592 | 2.0444 |
| 20 | True | True | False | 0.3833 | 0.3833 | 1.9456 | 1.6529 |
| 20 | False | False | True | 0.5583 | 0.5500 | 1.2167 | 1.1043 |
| 20 | True | False | True | 0.6333 | 0.6500 | 0.8508 | 0.9104 |
| 20 | False | True | True | 0.5667 | 0.5333 | 1.0768 | 1.0506 |
| 20 | True | True | True | 0.6833 | 0.6167 | 0.7448 | 0.8537 |

### Choice_3
This dataset consists of three classes; c0, c2, and c3 from an unsupervised clustering done on our dataset earlier. Different variants of the model were trained here with varying parameters and the statistics obtained for the different iterations are given below:

| Epochs | Finetune | Flatten Layer | Dense Layer | Accuracy | Validation Accuracy | Loss | Validation Loss | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 20 | False | False | False | 0.5656 | 0.5913 | 1.0222 | 0.9131 |
| 20 | True | False | False | 0.7964 | 0.8558 | 0.6559 | 0.5587 |
| 20 | False | True | False | 0.6372 | 0.6154 | 0.9631 | 0.9335 |
| 20 | True | True | False | 0.6581 | 0.7115 | 0.8747 | 0.7085 | 
| 20 | False | False | True | 0.8281 | 0.8221 | 0.5174 | 0.4923 |
| 20 | True | False | True | 0.8488 | 0.8365 | 0.4617 | 0.3965 |
| 20 | False | True | True | 0.7977 | 0.8077 | 0.5309 | 0.4822 |
| 20 | True | True | True | 0.8302 | 0.851 | 0.4649 | 0.4432 |

