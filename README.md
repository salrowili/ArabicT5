# ArabicT5: Efficient Adaptation of T5 on Arabic Language


# Model Description

This model adapt T5 on Arabic Language by pre-training T5 on ArabicWikipedia, Marefa, Hindawi Books and collection of Arabic News. Total Corpora size is 17GB. We restrict our corpora to News and Encyclopedias to enhance the performance of the model on informative tasks such as Factoid Question Answering and Generative task that uses classic Arabic ( الفصحى ). This model uses an efficient implementation of T5 which reduces the fine-tuning and memory used [Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers
](https://arxiv.org/abs/2109.10686) . 

## Link to our Paper

[Generative Approach for Gender-Rewriting Task with ArabicT5](https://aclanthology.org/2022.wanlp-1.55/)

## Pre-training Settings and Results on TyDi QA Development Dataset.

|     Model        | Hidden Layer | Atten. head | Atten. Layers | Vocab | Hardware  |Training Steps | Batch  |  Train x Batch Factor |Corpora                 | TyDi QA EM/F1| Link |
|------------------|--------------|-------------|---------------|-------|-----------|---------------|--------|-----------------------|------------------------|--------------|---|
| AraT5-Base       |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |248GB 29B tokens (MSA + Tweets)    |  69.16/82.82 | [![link][HF]](https://huggingface.co/UBC-NLP/AraT5-base) |  
| AraT5-Base-MSA   |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |70GB (MSA)              |  68.51/82.66 | [![link][HF]](https://huggingface.co/UBC-NLP/AraT5-msa-base) |
| AraT5-Base-Tweets|     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |178GB (Tweets)          |  64.39/78.22 | [![link][HF]](https://huggingface.co/UBC-NLP/AraT5-tweet-base) |
| mT5-Base         |     768      |      12     |      12       |  250K |TPUv3-32   |        1M     |  1024  | 8.0x                  |6.3T tokens (mC4)|  72.53/85.04 | [![link][HF]](https://huggingface.co/google/mt5-base) |
| ArabicT5-Base    |     512      |      8     |      20      |  32K  |TPUv3-32   |       256K    |  256   | 0.5x                 |17GB (MSA)          |  72.75/85.49 | [![link][HF]](https://huggingface.co/sultan/ArabicT5-Base)|
| ArabicT5-Large   |     768      |      12     |      16       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |  74.27/86.37      | [![link][HF]](https://huggingface.co/sultan/ArabicT5-Large) |
| ArabicT5-xLarge  |     768      |      12     |      36       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |  74.38/86.60       | [![link][HF]](https://huggingface.co/sultan/ArabicT5-xLarge) | 


## FineTuning our ArabicT5 model on generative and abstractive tasks with FLAX ###

[![Open In Colab][COLAB]](https://colab.research.google.com/github/salrowili/ArabicT5/blob/main/FineTuning_ArabicT5_with_FLAX_and_TPU.ipynb)

# Acknowledgment

We would like to acknowledge the support we have from Tensorflow Research Cloud (TFRC) team to grant us access to TPUv3 units.



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
