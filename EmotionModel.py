from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# Importă clasele pipeline, AutoTokenizer și AutoModelForSequenceClassification din biblioteca transformers.
# Acestea sunt utilizate pentru a crea un pipeline de prelucrare a limbajului natural și pentru a accesa modele pre-antrenate pentru clasificarea secvențelor.

# Creează un obiect de tip pipeline pentru clasificare zero-shot
# Zero-shot classification (clasificarea fără etichete) este o tehnică de învățare automată folosită pentru a clasifica input-uri în mai multe tipuri clase, 
# chiar și atunci când aceste input-uri nu au fost intalinte în timpul antrenamentului. 
# În cazurile tradiționale de clasificare, un model este antrenat pe exemple etichetate din fiecare clasă și apoi poate prezice clasa noilor exemple neintalnite.
classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-deberta-base')
# Se specifică utilizarea clasificării zero-shot și se alege modelul pre-antrenat "cross-encoder/nli-deberta-base".

# Listează emoțiile implicite
default_emotions = ["angry", "sad", "happy", "bored", "neutral"]

# Funcția pentru clasificarea unui mesaj cu clasele specificate
def classifyMessage2(message, classifications):
    # Se aplică clasificarea zero-shot asupra mesajului, cu clasele specificate
    response = classifier(message, classifications)
    scores = response['scores']
    labels = response['labels']

    # Căutarea celei mai mari scoruri și a clasei corespunzătoare
    max_score = 0
    max_index = 0
    i = 0
    for score in scores:
        if score > max_score:
            max_score = scores[i]
            max_index = i
        i += 1

    # Afișarea clasei cu cel mai mare scor și returnarea ei
    print(labels[max_index])
    return labels[max_index]

# Funcția pentru clasificarea unui mesaj cu emoțiile implicite
def classifyMessage(message):
    # Se aplică clasificarea zero-shot asupra mesajului, cu emoțiile implicite
    response = classifier(message, default_emotions)
    scores = response['scores']
    labels = response['labels']

    # Căutarea celei mai mari scoruri și a clasei corespunzătoare
    max_score = 0
    max_index = 0
    i = 0
    for score in scores:
        if score > max_score:
            max_score = scores[i]
            max_index = i
        i += 1

    # Verificarea dacă scorul maxim depășește un prag și returnarea clasei corespunzătoare
    if max_score > 0.80:
        return labels[max_index]
    
    # Afișarea clasei cu cel mai mare scor și returnarea valorii implicite "neutral" în caz contrar
    print(labels[max_index])
    return "neutral"
