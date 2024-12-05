# **AlphaAI - Intelligent Chatbot**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-orange)
![NLP](https://img.shields.io/badge/NLP-Tfidf%20%26%20Logistic%20Regression-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## **Overview**
Welcome to **AlphaAI**, a conversational chatbot designed using **Natural Language Processing (NLP)** and **Machine Learning** techniques. AlphaAI understands user inputs, predicts their intent, and provides dynamic responses through an elegant **Streamlit** web interface. It also logs conversations for future reference, offering a seamless and interactive experience.

---

## **Features**

### 🧠 **Core Functionalities**
- **Intent Recognition:** Uses **TF-IDF Vectorizer** and **Logistic Regression** for intent prediction.
- **Dynamic Responses:** Responds with predefined phrases based on the detected intent.
- **Conversation Logging:** Logs all interactions (user queries, chatbot responses, timestamps) to a `chat_log.csv` file for transparency and analysis.

### 🌐 **Streamlit-Based Interface**
- **User-Friendly Chatbox:** A clean and interactive interface for seamless conversations.
- **Sidebar Menu:**  
  - **Home:** Chat with AlphaAI.  
  - **Conversation History:** View a detailed log of past interactions.  
  - **About:** Learn more about the project, its architecture, and future scope.

### 📊 **Conversation History**
View a chronological log of all chats, including:
- User queries.
- Chatbot responses.
- Timestamps for every interaction.

---

## **Project Highlights**

### 🛠️ **Technologies Used**
1. **Natural Language Processing (NLP):**  
   - **TF-IDF Vectorizer** for feature extraction.
   - **Logistic Regression** for intent classification.
2. **Web Interface:**  
   - Built with **Streamlit**, a Python library for interactive web applications.
3. **File Logging:**  
   - Uses `chat_log.csv` to record all conversations.

---

## **Installation and Setup**

### **Prerequisites**
- Python 3.9+
- Install the required packages listed in `requirements.txt`.

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/Saket22-CS/Chatbot.git
   cd Chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK Data
   ```python
   import nltk
   nltk.download('punkt')
   ```
4. Run the chatbot:
   ```bash
   streamlit run chatbot.py
   ```
5. Open the provided URL in your browser to start chatting with **AlphaAI**.
   ```bash
   https://chatbot-vje3p3qjvowj5iuhbaaedh.streamlit.app/#alphaai
   ```
---

## **Project Structure**
```
📂 Chatbot/
├── chatbot.py            # Main application file.
├── intents.json          # Contains labeled intents and patterns for training.
├── chat_log.csv          # Logs user queries and chatbot responses.
├── requirements.txt      # Required Python packages.
└── README.md             # Documentation (this file).
```

---

## **Usage**
1. Launch the chatbot and type your queries in the text box.
2. Check the **Conversation History** menu for previous interactions.
3. Explore the **About** section to understand the technical architecture.

---

## **Acknowledgments**
- **NLTK** for natural language processing.
- **Scikit-learn** for machine learning algorithms.
- **Streamlit** for building the web interface.

---

## **Future Enhancements**
1. Expand the dataset with more intents and responses for improved versatility.
2. Integrate advanced **NLP models** (e.g., BERT, GPT) for higher accuracy and natural conversation.
3. Enable **speech-to-text** and **text-to-speech** capabilities for voice interaction.
4. Store conversation logs in a **database** for persistent storage across sessions.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contributions**
Contributions are welcome!  
Feel free to submit a pull request or open an issue if you’d like to add new features or report bugs.

---

## **Contact**
Created by **Saket Chaudhary**  
For inquiries or feedback, contact me on [Linkedin](https://www.linkedin.com/in/saket-chaudhary22/).
