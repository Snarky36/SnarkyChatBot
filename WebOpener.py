import webbrowser
from TextToSpeech import speak,listenForCommand
import requests
import sys, bs4
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

instruction = f'Instruction: given a dialog about movie recommendation, you need to respond based on human preferences.'
dialog = []

#userSet = pd.read_csv("./urlSets.csv")
tokenizer = AutoTokenizer.from_pretrained("bluenguyen/movie_chatbot_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("bluenguyen/movie_chatbot_v1")

""" Această metodă creează un context actualizat pentru modelul de conversație. 
Converteste lista de dialog într-un singur șir de caractere folosind " EOS " ca separator și adaugă instrucțiunea înainte de dialog. 
Returnează contextul generat."""

def current_context(dialog, instruction):
    dialog = ' EOS '.join(dialog)
    context = f"{instruction} [CONTEXT] {dialog} "
    print(context)
    return context

"""
Această metodă utilizează modelul preantrenat pentru a genera un răspuns pe baza contextului dat. 
Prin intermediul tokenizatorului, se construiește un tensor de intrare. 
Apoi, se utilizează modelul pentru a genera o secvență de ieșire.
Ieșirea este decodificată și returnată ca text.
"""



"""
În contextul învățării automate, tensorii sunt utilizati pentru a stoca și manipula date. 
Aceste date pot reprezenta imagini, sunete, texte sau alte tipuri de informații. 
Tensorii sunt esențiali în operațiile matematice și algoritmii de învățare automată,
deoarece permit efectuarea de calcule eficiente și paralele pe seturi mari de date.
"""

def generate(context):
    input_ids = tokenizer(f"{context}", return_tensors="pt")
    outputs = model.generate(**input_ids, max_length=128)
    output = tokenizer.decode(outputs[0])
    return output

"""
Această metodă extrage numele ultimului film menționat în dialog. 
Parcurge ultimul răspuns din dialog și identifică începutul și sfârșitul numelui filmului în formatul dat. 
Returnează numele filmului extras ca un șir de caractere.
"""

def extractLastMovie(dialog):
    print(dialog)
    lastResponse = dialog[-1]
    index = lastResponse.find("about")
    if index == -1:
        index = lastResponse.find("seen")
    while lastResponse[index] != '\"':
        index += 1
    movie = []
    index +=1
    while lastResponse[index]!= '\"':
        movie.append(lastResponse[index])
        index += 1
    print(''.join(movie))
    return ''.join(movie)

"""
Această metodă inițiază o conversație cu utilizatorul despre recomandări de filme. 
Întreabă utilizatorul despre genul preferat de filme și adaugă întrebarea în dialog-ul context. 
Apoi, într-un ciclu while, ascultă comanda utilizatorului și răspunde .
Dacă utilizatorul menționează deschiderea sau căutarea unui film, se extrage numele filmului din dialog și se realizează o căutare avansată pe web. 
Dacă utilizatorul nu dorește recomandări de filme sau menționează ieșirea, conversația se încheie.
"""

def startMovieConversation(dialog):
    query = "Any particular genre that you\'d like to see mentioned?"
    speak(query)
    dialog.append(query)
    while(1):
        query = listenForCommand()


        if "open" in query or "search" in query :
            speak("Would you like to see it now?")
            response = listenForCommand()
            if "yes" in response:
                movie = extractLastMovie(dialog)
                cautare_avansata(movie + " Online subtitrat")
        if "i don't want" in query and "recommendations" in query or "exit" in query:
            break
        dialog.append(query)
        response = generate(current_context(dialog, instruction))
        speak(response)
        dialog.append(response)


"""
Această metodă analizeaza starea emoțională a utilizatorului și răspunde în consecință. 
Dacă emoția este "happy", botul va transmite că este încântat că utilizatorul este fericit. 
Dacă emoția este "sad", botul întreabă utilizatorul dacă dorește să vadă ceva amuzant pentru a-l imbuna. 
Dacă emoția este "angry", botul sugerează căutarea de videoclipuri despre controlul și înțelegerea emoțiilor.
 Dacă emoția este "bored", botul întreabă utilizatorul dacă dorește o recomandare de film și inițiază o conversație despre filme.
"""

def botEmotionChecker(emotion):
    if emotion == "happy":
        speak("I am glad to see that you are happy!")
        response = listenForCommand()
    if emotion == "sad":
        speak("I see that you are sad! Do you want to see something fun to cheer you up?")
        response = listenForCommand()
        if "yes" in response:
            open("https://www.youtube.com/results?search_query=funny+videos")
            speak("I searched for you some funny videos enjoy and don't be sad!")
    if emotion == "angry":
        speak("Don't be so angry i can recommand you some videos about how to control and understand your emotions")
        response = listenForCommand()
    if emotion == "bored":
        speak("If you feel bored i can help you with that. Can i recommand you a movie?")
        response = listenForCommand()
        if "yes" in response:
            initialDialog = ['Do you have any recommendation about a movie?']
            startMovieConversation(initialDialog)



def open(url):
    webbrowser.open(url)

"""
Această metodă efectuează o căutare avansată pe web utilizând Google. 
Construiește un URL de căutare pe baza mesajului dat și folosește biblioteca requests pentru a obține rezultatele paginii.
Apoi, utilizează biblioteca bs4 pentru a analiza HTML-ul paginii și a extrage linkurile relevante. 
În cele din urmă, deschide primele linkuri într-un browser web cu ajutorul metodei open().
"""

def cautare_avansata(message):
    res = requests.get('https://google.com/search?q=' + ''.join(message))
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    linkElements = soup.select(' a')
    linkOpen = min(9, len(linkElements))
    for i in range(linkOpen):
        if i != 0 and i != 1 and i != 6:
            webbrowser.open('https://www.google.com' + linkElements[i].get('href'))

    speak('That is everything I found for' + message)
