import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Paso 1: Cargar y preprocesar los datos
def load_and_preprocess_data():
    archivo_excel = 'Data.xlsx'
    df = pd.read_excel(archivo_excel, usecols=range(7))
    stopwords_es = set(stopwords.words('spanish'))

    columnas_a_mantener = [1, 5]
    df = df.iloc[:, columnas_a_mantener]
    df.columns = ['label', 'text']

    df['text'] = df['text'].apply(clean_text, args=(stopwords_es,))
    return df

# Paso 2: Limpiar el texto
# Esta función limpia el texto eliminando caracteres no alfabéticos y stopwords.
# También aplica lematización a las palabras. Lematización es el proceso de convertir una palabra a su forma base.

def clean_text(text, stopwords_es):
    lemmatizer = WordNetLemmatizer()
    text_clean = re.sub(r'[^a-zA-ZáéíóúüÁÉÍÓÚÜñÑ]', ' ', str(text).lower())
    words = [lemmatizer.lemmatize(word) for word in text_clean.split() if word not in stopwords_es]
    return ' '.join(words)

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba y guardarlos en archivos CSV
def split_and_save_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    print("Preprocesamiento completo. Los datos se han guardado en 'train_data.csv' y 'test_data.csv'.")

# Ejecutar las funciones
df = load_and_preprocess_data()
split_and_save_data(df)
