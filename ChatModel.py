# Importam librariile necesare
# Importă biblioteci și module necesare
# Importă clase din biblioteca transformers pentru a utiliza tokenizerul și modelul Blenderbot pentru chat
# Importă funcții din modulul TextToSpeech pentru a gestiona operațiile legate de conversia textului în vorbire, cum ar fi ascultarea comenzilor utilizatorului, redarea răspunsurilor și schimbarea vocii botului
# Importă funcția classifyMessage din modulul EmotionModel pentru a detecta emoția din intrarea utilizatorului
# Importă funcții din modulul WebOpener pentru a efectua operații legate de web, cum ar fi verificarea emoțiilor botului și inițierea unei conversații despre filme
# Importă modulul webbrowser pentru a facilita funcționalități legate de web, cum ar fi deschiderea paginilor web
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from TextToSpeech import listenForCommand, speak, changeVoice
from EmotionModel import classifyMessage
from WebOpener import botEmotionChecker, startMovieConversation
import webbrowser

# Incarcam tokenizer-ul si modelul
"""Un tokenizer este o componentă cheie în prelucrarea limbajului natural (PLN),
 care se ocupă de împărțirea unui text în unități mai mici, 
 numite tokeni.
Tokenizarea este un pas esențial în PLN, deoarece transformă textul într-o formă pe care modelele de învățare automată pot să o înțeleagă și să o proceseze."""


"""
Prelucrarea Limbajului Natural se referă la abilitatea computerelor de a înțelege, interpreta și genera limbajul uman în mod automat. 
Aceasta implică dezvoltarea de algoritmi și modele care pot analiza și extrage informații din text, vorbire sau alte forme de comunicare în limbaj natural.
"""

chat_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
chat_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

def chatFunction():
    while 1:
        # Ascultam pentru comanda USER-ului
        utterance = listenForCommand()
        
        # Variabila ce ne spune daca bot-ul nostru poate vorbi sau nu
        freespeak = True
        
        # Verifica daca user-ul doreste ca bot-ul sa-si schimbe vocea
        if "change your voice" in utterance:
            changeVoice()
            speak("I have changed my voice! Hope you like it!")
            continue
        
        # Verifica daca user-ul doreste sa incheie conversatia
        if "goodbye" in utterance:
            speak("I will close now!")
            break
        
        # Verifica daca user-ul doreste sa-i fie recomandat un film
        if "recommend me a movie" in utterance:
            # Folosind WebOpener initializaza o discutie 
            initialDialog = ['Do you have any recommendation about a movie?']
            startMovieConversation(initialDialog)
            continue
        
        # Detecteaza emotii in dialogul persoanei
        emotionDetected = classifyMessage(utterance)
        
        # Verifica daca s-a gasit vreo emotie in conversatia cu utilizatorul
        if emotionDetected != "neutral":
            # Analizeaza emotiile folosind un model ce detecteaza emotii
            botEmotionChecker(emotionDetected)
            continue
        else:
            # Daca nicio emotie nu este gasita, genereaza un raspund folosind modelul de chat
            inputs = chat_tokenizer(utterance, return_tensors="pt")
            res = chat_model.generate(**inputs, max_length=50)
            result = chat_tokenizer.decode(res[0])
            
            # Converteste raspunsul in audio What 
            speak(result)

