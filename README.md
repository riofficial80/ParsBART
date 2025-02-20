# 🌐 **Persian BART: 6-Layer Model for Persian Language**  

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

## 🔄 **Pretraining Process:**  

### ⚙️ **Data Preprocessing:**  
For the pre-training of our language model, a massive amount of data was required. While large datasets were collected, they needed to be thoroughly cleaned to ensure data quality. We implemented a heuristic function to create an automatic cleaning pipeline for our pre-training datasets.

First, for each document in every dataset, we separated sentences and removed those that met any of the following criteria:

📝 Sentences with fewer than five words.

❌ Sentences that do not end with valid Persian end-of-sentence marks.

🚫 Sentences containing specific keywords from Persian webpages and JavaScript code.

After sentence filtering, we excluded documents with fewer than three sentences remaining.

Next, we utilized the langdetect package to filter out non-Persian documents, keeping only those with a Persian language probability of 0.99 or higher.

Finally, we removed duplicate paragraphs from documents to maintain content uniqueness.

This preprocessing procedure was exclusively applied to the pre-training datasets.



### 🚦 **Pretraining Execution:**
- Utilize **Hugging Face Transformers** and **PyTorch** for model training.  
- Configure **batch size**, **learning rate**, and **epoch settings**.  
#### **Training BART Tokenizer:**
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
[1] **Sabouri, S., Rahmati, E., Gooran, S. and Sameti, H.**, 2024. *naab: A ready-to-use plug-and-play corpus for Farsi.*  
*Journal of Artificial Intelligence, Applications and Innovations, 1(2), pp.1-8.*  
[📄 Full Paper](https://jaiai.iranaiai.ir/article_211486_3e490bce92a8af967a56870c8d200e90.pdf)  

---

