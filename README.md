# ArabicT5: Efficient Adaptation of T5 on Arabic Language

# Model Description

This model adapts T5 on the Arabic Language by pre-training T5 on : 
- Arabic Wikipedia.
- Marefa encyclopedia.
- Hindawi Books.
- a collection of Arabic News.

Total Corpora size is 17GB. We restrict our corpora to News and Encyclopedias to enhance the performance of the model on informative tasks such as Factoid Question Answering and Generative task that uses classic Arabic ( الفصحى ). This also gives our models an advantage if you don't want the generative text to contain inappropriate language. This model uses an efficient implementation of T5 which reduces the fine-tuning and memory used [Link](https://arxiv.org/abs/2109.10686) .

## Link to our models

[ArabicT5-Base](https://huggingface.co/sultan/ArabicT5-Base)


[ArabicT5-Large](https://huggingface.co/sultan/ArabicT5-Large)


[ArabicT5-xLarge](https://huggingface.co/sultan/ArabicT5-xLarge)


## Link to our Paper

[Generative Approach for Gender-Rewriting Task with ArabicT5](https://aclanthology.org/2022.wanlp-1.55/)


## Pre-training Settings and Results on TyDi QA Development Dataset ( Model in this card is highlighted in bold )

|     Model        | Hidden Layer | Atten. head | Atten. Layers | Vocab | Hardware  |Training Steps | Batch  |  Train x Batch Factor |Corpora                 |
|------------------|--------------|-------------|---------------|-------|-----------|---------------|--------|-----------------------|------------------------|
| AraT5-Base       |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |248GB 29B tokens (MSA + Tweets)    |
| AraT5-Base-MSA   |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |70GB (MSA)              |
| AraT5-Base-Tweets|     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |178GB (Tweets)          |
| AraBART-Base     |     768      |      12     |      12       |  50K | 128 V100 GPUs (60h)    |25 epochs|  -     | -                     |73GB (MSA)          |
| mT5-Base         |     768      |      12     |      12       |  250K |TPUv3-32   |        1M     |  1024  | 8.0x                  |6.3T tokens (mC4)|
| ArabicT5-Base   |     512      |      8     |      20      |  32K  |TPUv3-32   |       256K    |  256   | 0.5x                 |17GB (MSA)          |
| ArabicT5-Large    |     768      |      12     |      16       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |
| ArabicT5-xLarge  |     768      |      12     |      36       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |

##  Results on TyDi QA, HARD, Sentiment Analysis, Sarcasm Detection ( Best Score is highlighted in bold )

|    Model            | <center>TyDi QA| <center>HARD| <center>ArSarcasm-v2-Sentiment| <center>ArSarcasm-v2-Sarcasm| XL-SUM |
|----------------------|---------------|---------------------|-------------------------------------|----------------------------------|----------------------------------
| AraT5-Base           |  <center>70.36/84.21  |<center>96.49|<center>69.7/72.63|<center>60.44|<center>30.31|
| AraT5-Base-MSA       |  <center>70.90/84.00  |<center>**96.52**|<center>70.03/72.73|<center>60.69|<center>27.36|
| AraT5-Base-Tweets    |  <center>65.14/79.00  |<center>96.26|<center>70.67/73.52|<center>61.11|<center>25.08|
| mT5-Base             |  <center>72.20/84.13  |<center>96.24|<center>67.33/68.78|<center>52.18|<center>25.68|
| AraBART-Base         |  <center>48.75/71.15  |<center>96.11|<center>66.23/68.18|<center>56.30|<center>31.20|
| ArabicT5-Base        |  <center>70.79/84.76  |<center>96.36|<center>68.93/71.20|<center>58.93|<center>29.19|
| ArabicT5-Large       |  <center>73.29/86.08  |<center>96.40|<center>70.4/73.01|<center>59.79|<center>30.30|
| ArabicT5-xLarge      |  <center>**75.46/87.12**  |<center>96.50| <center>**72.23/75.17**|<center>**61.66**|<center>**31.70**|

Evaluation Metrics: TyDi QA (EM/F1), HARD (Accuracy), Sentiment Analysis (Accuracy / F1-PN positive-negative), Sarcasm Detection (F1-sarcastic), XL-SUM (Rouge-L with Stemmer).

You can download the full details of our grid search for all models in all tasks above from this link: https://github.com/salrowili/ArabicT5/raw/main/ArabicT5_Grid_Search.zip

For the XL-Sum task, we choose our best run for each model using the eval set. We use the official evaluation script from XL-Sum, which uses the stemmer function, which may show better results than papers that don't use the stemmer function. The official XL-Sum paper uses a stemmer function.

In our XL-Sum results, although we show that AraT5-Base exceeded our ArabicT5-Large, in most runs, our ArabicT5-Large shows better results, as you can see from our grid search file.



## FineTuning our ArabicT5 model on generative and abstractive tasks with FLAX ###

[![Open In Colab][COLAB]](https://colab.research.google.com/github/salrowili/ArabicT5/blob/main/FineTuning_ArabicT5_with_FLAX_and_TPU.ipynb)

# Acknowledgment

We want to acknowledge the support from the Tensorflow Research Cloud (TRC) team to grant us access to TPUv3 units.



# Citation
```
@inproceedings{alrowili-shanker-2022-generative,
    title = "Generative Approach for Gender-Rewriting Task with {A}rabic{T}5",
    author = "Alrowili, Sultan  and
      Shanker, Vijay",
    booktitle = "Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wanlp-1.55",
    pages = "491--495",
    abstract = "Addressing the correct gender in generative tasks (e.g., Machine Translation) has been an overlooked issue in the Arabic NLP. However, the recent introduction of the Arabic Parallel Gender Corpus (APGC) dataset has established new baselines for the Arabic Gender Rewriting task. To address the Gender Rewriting task, we first pre-train our new Seq2Seq ArabicT5 model on a 17GB of Arabic Corpora. Then, we continue pre-training our ArabicT5 model on the APGC dataset using a newly proposed method. Our evaluation shows that our ArabicT5 model, when trained on the APGC dataset, achieved competitive results against existing state-of-the-art methods. In addition, our ArabicT5 model shows better results on the APGC dataset compared to other Arabic and multilingual T5 models.",
}
```


[COLAB]: https://colab.research.google.com/assets/colab-badge.svg
[HF]: https://huggingface.co/front/assets/huggingface_logo-noborder.svg
