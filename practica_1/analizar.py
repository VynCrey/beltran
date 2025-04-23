import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import matplotlib.pyplot


corpus = PlaintextCorpusReader(".", 'CorpusLenguajes.txt')


texto_tokenizado=word_tokenize(corpus.raw()) 


def quitarStopwords_eng(text):
    ingles = stopwords.words("english")
    texto_limpio = [w.lower() for w in text if w.lower() not in ingles
        and w not in string.punctuation
        and w not in ["'s", '|', '--', "''", "``",".-", "-","word_tokenize","quitarStopwords_eng","lematizar","corpus"] ]
    return texto_limpio


quitar = quitarStopwords_eng(texto_tokenizado)

#InicializarelLematizador
lemmatizer=WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    texto_lema=[lemmatizer.lemmatize(w,get_wordnet_pos(w))for w in texto]
    return texto_lema

print("")
print("-" * 36)
print("")


# Corpus preparado
print("Corpus preparado (lematizado):")
print(lematizar(quitar))


# TF-IDF
texto_limpio_str = ' '.join(lematizar(quitar))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([texto_limpio_str])

print("")
print("-" * 36)
print("")


# Mostrar Matriz TF-IDF
print("Matriz TF-IDF:")
print(X.toarray())

print("")
print("-" * 36)
print("")


# Vocabulario
print("Vocabulario generado:")
print(vectorizer.get_feature_names_out())

print("")
print("-" * 36)
print("")


frecuencia = FreqDist(lematizar(quitar))

#6 palabras más usadas
palabras_mas_usadas = frecuencia.most_common(6)
print("6 palabras más usadas:", palabras_mas_usadas)

print("")
print("-" * 36)
print("")


#Palabra menos usada
palabra_menos_usada = min(frecuencia, key=frecuencia.get)

#Imprimir la palabra menos utilizada
print("La palabra menos utilizada es:", palabra_menos_usada)

print("")
print("-" * 56)
print("")


# Gráfico de distribución de frecuencia
frecuencia.plot(20, title='Distribución de Frecuencia - Top 20 Palabras')
matplotlib.pyplot.show()

# Palabras más repetidas en una misma oración
print("Palabras más repetidas por oración:")
print("")

oraciones = sent_tokenize(corpus.raw())
for idx, oracion in enumerate(oraciones):
    tokens = quitarStopwords_eng(word_tokenize(oracion))
    tokens_lema = lematizar(tokens)
    dist = FreqDist(tokens_lema)
    if len(dist) > 0:
        palabra, repeticiones = dist.most_common(1)[0]
        if repeticiones > 1:
            print("Palabra más repetida en cada oracion: ", palabra)














