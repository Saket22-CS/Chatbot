import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("AlphaAI ")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("👋 Welcome to your friendly chatbot! Type a message below and hit Enter to start chatting.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("🌟 It was great chatting with you! If you need help again, feel free to come back. Have an amazing day! 😊")
                st.balloons()  # Adds a celebratory touch
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("Welcome to the Chatbot Project, a demonstration of leveraging Natural Language Processing (NLP) and Machine Learning to create an intelligent conversational agent. This chatbot is designed to understand and respond to user inputs based on predefined intents, offering a dynamic and user-friendly experience through a web-based interface built with Streamlit.")

        st.subheader("Project Overview:")

        st.write("""
        This project consists of two main components:
        1. NLP and Machine Learning:
            - We used NLP techniques and the Logistic Regression algorithm to train the chatbot on labeled intents and entities.
            - The model processes user input, predicts the intent, and generates relevant responses.      
        2. Interactive Interface with Streamlit:
            - A web-based interface was developed using Streamlit to allow users to interact with the chatbot seamlessly.
            - The interface includes a text input field for queries and a display area for responses, ensuring an engaging user experience.
        """)

        st.subheader("Dataset Details")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: Categories representing the purpose of user inputs (e.g., greeting, budget, about).
        - Entities: Extracted information or key phrases from the user input (e.g., "Hi", "How do I create a budget?", "What is your purpose?").
        - Text: User input phrases used to train and test the model.
        """)

        st.subheader("Chatbot Interface")

        st.write("""
        The chatbot interface is designed for simplicity and interactivity.
        - User Input Field: Users can type their queries into a text box.
        - Chat Display: Responses generated by the trained model are displayed in a chat-like format.
        - Real-Time Interaction: The chatbot leverages the trained model to provide quick and meaningful replies, ensuring a smooth conversational flow.     
        """)

        st.subheader("Conclusion")

        st.write("""
        This project demonstrates how to build a chatbot that can. 
        - Understand user input using NLP and Machine Learning.
        - Deliver relevant responses based on intents through a sleek Streamlit interface.                   
        """)

        st.subheader("Future Scope")

        st.write("""
        To enhance this project, we plan to:  
        1. Expand the dataset with more intents and diverse user queries.
        2. Integrate advanced NLP models like transformers (e.g., BERT, GPT) for better intent detection.
        3. Incorporate features like speech-to-text and text-to-speech for multimodal interactions.
                 
        By combining innovation and simplicity, this chatbot lays the foundation for developing more sophisticated conversational agents.                  
        """)

if __name__ == '__main__':
    main()
