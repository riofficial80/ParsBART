# 🌐 **Persian BART**  

This repository hosts a **6-layer BART (Bidirectional and Auto-Regressive Transformers)** model, pretrained exclusively on **Persian** text data (**Naab** dataset [1]). The model is designed to handle **natural language processing (NLP)** tasks with a **context length of 256 tokens**.  

---

## 🧠 **Key Features:**  
- 🏗️ **Architecture:** BART with **6 encoder** and **6 decoder** layers.  
- 📚 **Pretraining:** Using the **Persian Naab dataset** for enhanced language understanding.  
- 📏 **Context Length:** Supports input sequences of up to **256 tokens**.  
- 🚀 **Applications:**  
  - 📝 **Summarization**  
  - 🌐 **Translation**  
  - ✍️ **Text Generation**  
  - 📊 **Classification**  

---

## 🔡 **Train BART Tokenizer:**  

To train BART Tokenizer you can use **train_tokenizer.ipynb** scrip that is used to **train a custom tokenizer** on the **PN-summary dataset** (see this link to understand characteristics of this dataset: https://huggingface.co/datasets/HooshvareLab/pn_summary), employing the **Byte-Pair Encoding (BPE)** algorithm through the **Hugging Face `tokenizers` library**. The mentioned script is located at the following path:


The tokenizer is configured with the following parameters:

- **vocab_size=52000**: Sets the vocabulary size to **52,000 tokens**, balancing model expressiveness with memory efficiency.
- **min_frequency=10**: Establishes the minimum token frequency required for inclusion in the vocabulary, ensuring that rare tokens are excluded to improve model generalization.
- **special_tokens**: Defines a list of special tokens reserved for model-specific tasks:
  - `<s>`: Start-of-sequence token.
  - `<pad>`: Padding token.
  - `</s>`: End-of-sequence token.
  - `<unk>`: Unknown token.
  - `<mask>`: Masking token.



The trained tokenizer is **saved** in a **format compatible** with the **`transformers` library**, enabling **smooth integration** with the **BART model** for **Persian text processing**.  



---

## 🔄 **Pretraining Process:**  

### ⚙️ **Data Preprocessing:**  
For the pre-training of our language model, a massive amount of data was required. While large datasets were collected, they needed to be thoroughly cleaned to ensure data quality. We implemented a heuristic function to create an automatic cleaning pipeline for our pre-training datasets.

First, for each document in our dataset, we separated sentences and removed those that met any of the following criteria [2]:

- 📝 Sentences with fewer than five words.

- ❌ Sentences that do not end with valid Persian end-of-sentence marks.

- 🚫 Sentences containing specific keywords from Persian webpages and JavaScript code.

After sentence filtering, we excluded documents with fewer than three sentences remaining.

Next, we utilized the langdetect package to filter out non-Persian documents, keeping only those with a Persian language probability of 0.99 or higher.

Finally, we removed duplicate paragraphs from documents to maintain content uniqueness.

This preprocessing procedure was exclusively applied to the pre-training datasets.

#### Limitations:
- Large Dataset Size: The Naab dataset is extremely large, which posed challenges in processing with limited computational resources.
- Resource Constraints: We had access only to Google Colab Free, which: Limits session duration.
- Is vulnerable to interruptions, such as internet connection drops, which could terminate the session unexpectedly.

#### Handling Colab Session Limitations:
To address these challenges, we adopted the following strategies:

- Streaming Data: Leveraged the streaming feature of Hugging Face Datasets to clean each row individually without needing to load the entire dataset into memory.
- 
- Session Recovery: Stored the index of the last cleaned row in a file, allowing us to resume preprocessing from the same point in case of session disconnection.

#### Preprocessing Optimizations:
To save time during pretraining, we applied various perturbation functions [3] during preprocessing, including:

- Token Infilling: Randomly replace tokens with a mask.
- Token Deletion: Remove random tokens from the input.
- Token Masking: Mask specific tokens for model learning.
- Document Rotation: Rotate sections of documents to improve model generalization.
- Sentence Permutation: Randomly shuffle sentences within documents.

#### Dataset Statistics:

|-------------------------------------------------------|-----------------|
| Number of Cleaned Documents                           | 11,500,000      |
| Number of Rows Cleaned from NAAB                      | 197,667,045     |
| Size of Dataset                                       | 19 GB           |
| Total Size of NAAB Used for Cleaning                  | 134 GB          |
| Documents per `document_rotation` Function            | 2,305,006       |
| Documents per `sentence_permutation` Function         | 2,291,018       |
| Documents per `token_infilling` Function              | 2,305,881       |
| Documents per `token_masking` Function                | 2,296,858       |
| Documents per `token_deletion` Function               | 2,301,237       |


#### Code and Dataset Availability:
All preprocessing code can be found in the **data_preparation.ipynb** script, located at:


To access cleaned dataset, please contact using the following email:
ri.official80@gmail.com


### 🚦 **Pretraining Execution:**
In this section, we will explain how we trained BART model.
- Utilize **Hugging Face Transformers** and **PyTorch** for model training.  
- Configure **batch size**, **learning rate**, and **epoch settings**.  


#### 🧬 **Model Architecture:**  
- **Encoder-Decoder** structure with **6 layers** in each.  
- Optimized for **Persian language tasks**.  


#### 📑 **Pretraining Arguments:**  
- `--max_length 256`  
- `--num_beams 4`  
- `--early_stopping True`  

---

## 🎯 **Finetuning:**  

### 🧠 **Natural Language Understanding (NLU) Tasks:**  

#### 😊 **Sentiment Analysis:**  
- Classify Persian text as **positive**, **negative**, or **neutral**.  

#### 🏷️ **Text Classification:**  
- Assign **labels** to Persian documents.  

### 📝 **Natural Language Generation (NLG) Tasks:**  

#### 📰 **News Summarization:**  
- Generate concise summaries of **Persian news articles**.  

#### 🗝️ **Key Generation:**  
- Create **keywords** for Persian texts.  

---

## 📚 **References:**  
[1] **Sabouri, S., Rahmati, E., Gooran, S. and Sameti, H.**, 2024. *naab: A ready-to-use plug-and-play corpus for Farsi.* *Journal of Artificial Intelligence, Applications and Innovations, 1(2), pp.1-8.*
[📄 Full Paper](https://jaiai.iranaiai.ir/article_211486_3e490bce92a8af967a56870c8d200e90.pdf)  

[2] **Salemi, A., Kebriaei, E., Minaei, G.N. and Shakery, A.**, 2021. *Arman: Pre-training with semantically selecting and reordering of sentences for persian abstractive summarization*. *arXiv preprint arXiv:2109.04098.*
[📄 Full Paper](https://arxiv.org/pdf/2109.04098) 

[3] **Lewis, M.**, 2019. *Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.* *arXiv preprint arXiv:1910.13461.*
[📄 Full Paper](https://arxiv.org/pdf/1910.13461) 

---

