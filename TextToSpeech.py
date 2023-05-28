import speech_recognition as sr
import pyttsx3

# Variable ce ne spune ce voce alegem
vocea = 0

# Functie care schimba vocea
def changeVoice():
    global vocea
    if vocea == 1:
        vocea = 0
    else:
        vocea = 1

# Functie ce converteste text in audio
def speak(audio):
    # Vorbeste audio-ul

    global vocea
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[vocea].id)
    print(audio)
    engine.say(audio)
    engine.runAndWait()

# Functie ce asculta o comanda prin microfon
def listenForCommand():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        # Folosim google speech recognition pentru a asculta audio
        comanda = r.recognize_google(audio).lower()
        print('You said: ' + comanda + '\n')

    except sr.UnknownValueError:
        # Daca comanda nu a fost recunoscuta, cere input din nou
        print('Your last command couldn\'t be heard')
        comanda = listenForCommand()

    return comanda