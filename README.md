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
- Clean and tokenize Persian text data.  
- Ensure compatibility with BART's architecture.  

### 🚦 **Pretraining Execution:**
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
[1] **Sabouri, S., Rahmati, E., Gooran, S. and Sameti, H.**, 2024. *naab: A ready-to-use plug-and-play corpus for Farsi.*  
*Journal of Artificial Intelligence, Applications and Innovations, 1(2), pp.1-8.*  
[📄 Full Paper](https://jaiai.iranaiai.ir/article_211486_3e490bce92a8af967a56870c8d200e90.pdf)  

---

