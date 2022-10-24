# ArabicT5: Efficient Adaptation of T5 on Arabic Language


# Model Description

This model adapt T5 on Arabic Language by pre-training T5 on ArabicWikipedia, Marefa, Hindawi Books and collection of Arabic News. Total Corpora size is 17GB. We restrict our corpora to News and Encyclopedias to enhance the performance of the model on informative tasks such as Factoid Question Answering and Generative task that uses classic Arabic ( الفصحى ). This model uses an efficient implementation of T5 which reduces the fine-tuning and memory used [Link](https://arxiv.org/abs/2109.10686) .

## Pre-training Settings and Results on TyDi QA Development Dataset ( Model in this card is highlighted in bold )

|     Model        | Hidden Layer | Atten. head | Atten. Layers | Vocab | Hardware  |Training Steps | Batch  |  Train x Batch Factor |Corpora                 | TyDi QA EM/F1|
|------------------|--------------|-------------|---------------|-------|-----------|---------------|--------|-----------------------|------------------------|--------------|
| AraT5-Base       |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |248GB 29B tokens (MSA + Tweets)    |  69.16/82.82 |  
| AraT5-Base-MSA   |     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |70GB (MSA)              |  68.51/82.66 | 
| AraT5-Base-Tweets|     768      |      12     |      12       |  110K |TPUv3-8    |        1M     |  128   | 1.0x                  |178GB (Tweets)          |  64.39/78.22 |
| mT5-Base         |     768      |      12     |      12       |  250K |TPUv3-32   |        1M     |  1024  | 8.0x                  |6.3T tokens (mC4)|  71.55/83.78 | 
| ArabicT5-Base    |     512     |      8      |      20      |  32K  |TPUv3-32   |       256K    |  256   | 0.5x                |17GB (MSA)         |  72.75/85.49 | 
| ArabicT5-Large   |     768      |      12     |      16       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |  --/--       | 
| ArabicT5-xLarge  |     768      |      12     |      36       |  32K  |TPUv3-128  |       500K    |  512   | 2.0x                  |17GB (MSA)          |  --/--       |  



# Acknowledgment

We would like to acknowledge the support we have from Tensorflow Research Cloud (TFRC) team to grant us access to TPUv3 units.

# Citation
Paper will be shared soon
