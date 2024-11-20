import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import os
import tkinter
from tkinter import *
import cv2
from fer import FER  # Import the FER library for facial expression recognition
import speech_recognition as sr  # Import speech recognition library

# Check if nltk packages are downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the model
model = load_model('chatbot_model.h5')

# Load intents from the correct JSON file
json_file_path = 'E:\myProjects\AI-Powered-Emotion-and-Voice-Based-Mental-Health-Companion.-main\sample.json'
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"JSON file not found: {json_file_path}")
intents = json.loads(open("sample.json").read())


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    print("Predicted results:", results)
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        return random.choice(["Sorry, I didn't understand that.", "Can you please rephrase?", "I'm not sure how to respond to that."])
    
    tag = ints[0]['intent']
    print("Selected intent:", tag)  # Log the selected intent
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            print("Response selected:", result)  # Log the selected response
            return result
    
    return random.choice(["Sorry, I didn't understand that.", "Can you please rephrase?", "I'm not sure how to respond to that."])

# Define the chatbot response function
def chatbot_response(msg):
    ints = predict_class(msg, model)  # Get the predicted class
    res = getResponse(ints, intents)   # Get the response based on the predicted class
    return res

def detect_emotion():
    detector = FER()
    video_capture = cv2.VideoCapture(0)  # Start capturing video from webcam

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Analyze the frame for emotions
        emotions = detector.detect_emotions(frame)
        if emotions:
            # Get the sadness score
            sadness_score = emotions[0]['emotions'].get('sad', 0)
            print(f"Detected sadness level: {sadness_score}")

            # Lower the threshold to open the chatbot faster
            if sadness_score > 0.5:  # Lower threshold to 0.5
                print("Detected emotion: Depressed")
                video_capture.release()
                cv2.destroyAllWindows()
                return 'depressed'
            elif 0.3 <= sadness_score <= 0.5:  # Adjust the range for extremely sad
                print("Detected emotion: Extremely Sad")
            elif sadness_score > 0:
                print("Detected emotion: Sad")

        cv2.imshow('Facial Emotion Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return 'none'


# Function to capture voice input
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

        try:
            msg = recognizer.recognize_google(audio)
            print(f"You said: {msg}")
            return msg
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

# Creating GUI with tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg == '':  # Check if the text entry is empty
        msg = listen()  # Capture voice input if text entry is empty

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)  # Use the newly defined function
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Check for 'depressed' emotion before starting the chat
emotion = detect_emotion()
if emotion == 'depressed':
    print("User is depressed. Starting the chatbot...")

    # Create Chat window
    ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
    ChatLog.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                        command=send)

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    ChatLog.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)

    base.mainloop()
else:
    print(f"User is {emotion}. Chatbot will not open.")
