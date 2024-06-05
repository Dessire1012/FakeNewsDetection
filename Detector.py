import joblib
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')
stopwords_es = set(stopwords.words('spanish'))

def leer_archivo(filepath):
    df = pd.read_csv(filepath)
    print(f"Columnas en el archivo CSV: {df.columns.tolist()}")
    return df[df.columns[0]].tolist()

def cargar_modelos(vectorizer_path, model_path):
    bow_vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return bow_vectorizer, model

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text).lower()
    words = text.split()
    words_filtered = [word for word in words if word not in stopwords_es]
    return ' '.join(words_filtered)

def transformar_texto(vectorizer, text):
    return vectorizer.transform([text])

def predecir_noticia(model, transformed_text):
    return model.predict(transformed_text)

def main():
    # Paso 1: Leer el archivo de texto
    noticias = leer_archivo('Noticias.csv')

    # Paso 2: Cargar el vectorizador y el modelo
    bow_vectorizer, model = cargar_modelos('vectorizers/bow_vectorizer.joblib', 'models/Naive_Bayes_1_model.joblib')

    total_noticias = len(noticias)
    fake_count = 0
    true_count = 0

    for i, noticia in enumerate(noticias):
        # Paso 3: Eliminar las stop words de la noticia
        noticia_procesada = preprocess_text(noticia)

        # Paso 4: Transformar la noticia procesada
        noticia_transformada = transformar_texto(bow_vectorizer, noticia_procesada)

        # Paso 5: Predecir la etiqueta de la noticia
        prediccion = predecir_noticia(model, noticia_transformada)

        # Paso 6: Contar predicciones
        if prediccion[0] == 'Fake':
            fake_count += 1
            print(f"La noticia {i+1} es falsa.")
        else:
            true_count += 1
            print(f"La noticia {i+1} es verdadera.")

    # Imprimir porcentajes
    print(f"\nNoticias clasificadas como falsas: {fake_count}")
    print(f"Noticias clasificadas como verdaderas: {true_count}")

if __name__ == "__main__":
    main()
