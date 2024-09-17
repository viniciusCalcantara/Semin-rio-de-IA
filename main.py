# -*- coding: utf-8 -*-

import nltk
import random
import string
import warnings
from nltk.corpus import stopwords
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
warnings.filterwarnings('ignore')

f = open('C:\\Users\\ViníciusAlcântara\\sklearn-env\\criptomoeda.txt', 'r', errors='ignore', encoding='utf-8')
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) #converts to list of scentences
word_tokens = nltk.word_tokenize(raw) #converts to list of words

# sentToken = sent_tokens[:4]
# print(sentToken)
# wordToken = word_tokens[:4]
# print(wordToken)

#preprocessing  
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Greetings
GREETING_INPUTS = ("olá", "oi", "saudações", "e aí", "como vai", "olá")
GREETING_RESPONSES = ["olá", "oi", "e aí", "olá, como vai?", "olá", "estou feliz em falar com você"]

def greeting(scentence):
    
    for word in scentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
#Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=stopwords.words('portuguese'), encoding='utf-8')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        chatbot_response = chatbot_response + "Desculpe, eu não entendi o que você disse."
        return chatbot_response
    
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]
        return chatbot_response
    

if __name__ == "__main__":
    flag = True
    print("Olá, meu nome é Aneka. Vou responder suas perguntas sobre criptomoedas. Se quiser sair, digite 'tchau'.")
    while(flag==True):
        user_response = input()
        user_response = user_response.lower()
        if user_response != 'tchau':
            if user_response in ('obrigado', 'obrigada', 'valeu'):
                flag = False
                print("Aneka: De nada!")
            else:
                if greeting(user_response) is not None:
                    print("Aneka: " + greeting(user_response))
                else:
                    print("Aneka:", end='')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print("Aneka: Tchau! Tenha um ótimo dia!")