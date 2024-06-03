import pandas as pd
import joblib

# Paso 1: Cargar los datos preprocesados
def load_test_data():
    archivo_test = 'test_data.csv'
    df_test = pd.read_csv(archivo_test)
    return df_test

# Paso 2: Vectorizar los datos de prueba
def vectorize_test_data(X_test):
    tfidf_vectorizer = joblib.load('vectorizers/tfidf_vectorizer.joblib')
    bow_vectorizer = joblib.load('vectorizers/bow_vectorizer.joblib')
    ngram_vectorizer = joblib.load('vectorizers/ngram_vectorizer.joblib')

    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_test_bow = bow_vectorizer.transform(X_test)
    X_test_ngrams = ngram_vectorizer.transform(X_test)

    return X_test_tfidf, X_test_bow, X_test_ngrams

# Paso 3: Evaluar los modelos en los datos de prueba
def evaluate_models_on_test_data(X_test_tfidf, X_test_bow, X_test_ngrams, y_test, training_times_df):
    vectorizer_dict = {
        "n-grams": X_test_ngrams,
        "TF-IDF": X_test_tfidf,
        "Bag of Words": X_test_bow
    }

    results = []

    for _, row in training_times_df.iterrows():
        model_name = row['Modelo']
        vectorizer_name = row['Vectorizer']
        hyperparams = row['Hiperparámetros']
        model_filename = f"models/{model_name}_model.joblib"

        model = joblib.load(model_filename)
        X_test = vectorizer_dict[vectorizer_name]

        accuracy = model.score(X_test, y_test)
        results.append((model_name, vectorizer_name, accuracy, hyperparams))

    return results

# Paso 4: Ejecutar el flujo de trabajo
def evaluate_workflow():
    # Cargar datos de prueba
    df_test = load_test_data()
    X_test = df_test['text']
    y_test = df_test['label']

    # Vectorizar datos de prueba
    X_test_tfidf, X_test_bow, X_test_ngrams = vectorize_test_data(X_test)

    # Cargar el archivo de tiempos de entrenamiento
    training_times_df = pd.read_excel('training_times.xlsx')

    print("\nEvaluando modelos en datos de prueba")
    results = evaluate_models_on_test_data(X_test_tfidf, X_test_bow, X_test_ngrams, y_test, training_times_df)

    # Crear DataFrame con los resultados
    results_df = pd.DataFrame(results, columns=['Modelo', 'Vectorizer', 'Accuracy', 'Hiperparámetros'])

    # Guardar el DataFrame en un archivo Excel
    results_df.to_excel('results.xlsx', index=False)
    print("\nEvaluación completa. Los resultados se han guardado en 'results.xlsx'.")

evaluate_workflow()
