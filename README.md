# ArabicT5: Efficient Adaptation of T5 on Arabic Language


# Model Description

This model adapts T5 on the Arabic Language by pre-training T5 on : 
- Arabic Wikipedia.
- Marefa encyclopedia.
- Hindawi Books.
- a collection of Arabic News.
- OSCAR Dataset (32GB)

This model uses an efficient implementation of T5 which reduces the fine-tuning and memory used [Link](https://arxiv.org/abs/2109.10686) and uses T5x for pre-training [Link](https://github.com/google-research/t5x)


## Pre-training Settings and Results on TyDi QA Development Dataset ( Model in this card is highlighted in bold )

|     Model        | Hidden Layer | Atten. head | Atten. Layers | Vocab | Hardware  |Training Steps | Batch  |  Train x Batch Factor |Corpora                 |
|------------------|--------------|-------------|---------------|-------|-----------|---------------|--------|-----------------------|------------------------|
| AraT5-base       |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |248GB 29B tokens (MSA + Tweets)    |
| AraT5-msa-base   |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |70GB (MSA)              |
| AraT5-tweets-base|     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |178GB (Tweets)          |
| AraBART-base     |     768      |      12     |      12       |  50K | 128 V100 GPUs (60h)    |25 epochs|  -     | -                     |73GB (MSA)          |
| mT5-base         |     768      |      12     |      12       |  250K |TPUv3-32   |        1M     |  1024  | 8.0x                  |6.3T tokens (mC4)|
| ArabicT5-17GB-small   |     512      |      8     |      20      |  32K  |TPUv3-32   |       256K    |  256   | 0.5x                 |17GB (MSA)          |
| ArabicT5-49GB-small   |     512      |      8     |      16      |  32K  |TPUv3-64   |       500K    |  256   | 1.0x                 |49GB (MSA + OSCAR)          |
| ArabicT5-17GB-base    |     768      |      12     |      16       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |
| ArabicT5-49GB-base    |     768      |      12     |      16       |  32K  |TPUv3-64  |       500K    |  256   | 1.0x                  |49GB (MSA + OSCAR)          |
| ArabicT5-17GB-large  |     768      |      12     |      36       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |


##  Results on TyDi QA, HARD, Sentiment Analysis, Sarcasm Detection ( Best Score is highlighted in bold )

|    Model            | <center>TyDi QA| <center>HARD| <center>ArSarcasm-v2-Sentiment| <center>ArSarcasm-v2-Sarcasm| XL-SUM |
|----------------------|---------------|---------------------|-------------------------------------|----------------------------------|----------------------------------
| AraT5-base           |  <center>70.4/84.2  |<center>**96.5**|<center>69.7/72.6|<center>60.4|<center>30.3|
| AraT5-msa-base       |  <center>70.9/84.0  |<center>**96.5**|<center>70.0/72.7|<center>60.7|<center>27.4|
| AraT5-tweets-base    |  <center>65.1/79.0  |<center>96.3|<center>70.7/73.5|<center>61.1|<center>25.1|
| mT5-base             |  <center>72.2/84.1  |<center>96.2|<center>67.3/68.8|<center>52.2|<center>25.7|
| AraBART-base         |  <center>48.8/71.2  |<center>96.1|<center>66.2/68.2|<center>56.3|<center>31.2|
| ArabicT5-17GB-small        |  <center>70.8/84.8  |<center>96.4|<center>68.9/71.2|<center>58.9|<center>29.2|
| ArabicT5-49GB-small        |  <center>72.4/85.1  |<center>96.4|<center>70.2/73.4|<center>61.0|<center>30.2|
| ArabicT5-17GB-base       |  <center>73.3/86.1  |<center>96.4|<center>70.4/73.0|<center>59.8|<center>30.3|
| ArabicT5-49GB-base       |  <center>72.1/85.1  |<center>**96.5**|<center>71.3/74.1|<center>60.4|<center>30.9|
| ArabicT5-17GB-large      |  <center>**75.5/87.1**  |<center>**96.5**| <center>**72.2/75.2**|<center>**61.7**|<center>**31.7**|

Evaluation Metrics: TyDi QA (EM/F1), HARD (Accuracy), Sentiment Analysis (Accuracy / F1-PN positive-negative), Sarcasm Detection (F1-sarcastic), XL-SUM (Rouge-L with Stemmer).

You can download the full details of our grid search for all models in all tasks above from this link: https://github.com/salrowili/ArabicT5/raw/main/ArabicT5_Grid_Search.zip

For the XL-Sum task, we choose our best run for each model using the eval set. We use the official evaluation script from XL-Sum, which uses the stemmer function, which may show better results than papers that don't use the stemmer function. The official XL-Sum paper uses a stemmer function.

# FineTuning our efficient ArabicT5-49GB-Small model with Torch on 3070 laptop GPU ###

[![Open In Colab][COLAB]](https://colab.research.google.com/github/salrowili/ArabicT5/blob/main/ArabicT5_49GB_Small_on_3070_Laptop_GPU.ipynb)

If you are running your code on a laptop GPU (e.g., a gaming laptop) or limited GPU memory, we recommended using our ArabicT5-49GB-Small model, which was the only model from the list that we were able to run on 3070 Laptop card with a batch size of 8. We manage to achieve an F1 score of 85.391 (slightly better than our FLAX code ) on the TyDi QA task. 


# FineTuning our ArabicT5 model on generative and abstractive tasks with FLAX ###

[![Open In Colab][COLAB]](https://colab.research.google.com/github/salrowili/ArabicT5/blob/main/FineTuning_ArabicT5_with_FLAX_and_TPU.ipynb)

[COLAB]: https://colab.research.google.com/assets/colab-badge.svg


# FineTuning ArabicT5 on TPUv3-8 with free Kaggle ###


https://www.kaggle.com/code/sultanalrowili/arabict5-on-tydi-with-free-tpuv3-8-with-kaggle



# Continual Pre-Training of ArabicT5 with T5x
if you want to continue pre-training ArabicT5 on your own data, we have uploaded the raw t5x checkpoint to this link https://huggingface.co/sultan/ArabicT5-49GB-base/blob/main/arabict5_49GB_base_t5x.tar.gz
We will soon share a tutorial on how you can do that for free with Kaggle TPU



# Acknowledgment

We want to acknowledge the support we have from The TPU Research Cloud (TRC) team to grant us access to TPUv3 units.


# Paper

[Generative Approach for Gender-Rewriting Task with ArabicT5](https://aclanthology.org/2022.wanlp-1.55/)

# Citation

```bibtex
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
