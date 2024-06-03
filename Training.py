import pandas as pd
import time
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Paso 1: Cargar los datos preprocesados
def load_data():
    archivo = 'train_data.csv'
    df = pd.read_csv(archivo)
    return df

# Paso 2: Transformar los datos en diferentes representaciones y combinarlas
def transform_data(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    bow_vectorizer = CountVectorizer(max_features=1000)
    ngram_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 3))

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    X_train_ngrams = ngram_vectorizer.fit_transform(X_train)
    X_test_ngrams = ngram_vectorizer.transform(X_test)

    if not os.path.exists("vectorizers"):
        os.makedirs("vectorizers")

    joblib.dump(tfidf_vectorizer, 'vectorizers/tfidf_vectorizer.joblib')
    joblib.dump(bow_vectorizer, 'vectorizers/bow_vectorizer.joblib')
    joblib.dump(ngram_vectorizer, 'vectorizers/ngram_vectorizer.joblib')

    return X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow, X_train_ngrams, X_test_ngrams

# Paso 3: Entrenar y guardar modelos
def train_and_save_models(X_train, y_train, vectorizer_type):
    models = [
        ("Naive_Bayes_1", MultinomialNB(alpha=1.0)),
        ("Naive_Bayes_2", MultinomialNB(alpha=0.5)),
        ("Naive_Bayes_3", MultinomialNB(alpha=0.1)),
        ("SVM_1", SVC(C=1.0, kernel='linear')),
        ("SVM_2", SVC(C=0.5, kernel='linear')),
        ("SVM_3", SVC(C=1.0, kernel='rbf')),
        ("Logistic_Regression_1", LogisticRegression(max_iter=1000, solver='lbfgs')),
        ("Logistic_Regression_2", LogisticRegression(max_iter=1000, solver='liblinear')),
        ("Logistic_Regression_3", LogisticRegression(max_iter=2000, solver='lbfgs'))
    ]

    if not os.path.exists("models"):
        os.makedirs("models")

    times = []

    for name, model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        joblib.dump(model, f'models/{name}_model.joblib')

        hyperparams = model.get_params()
        relevant_hyperparams = {k: hyperparams[k] for k in hyperparams if k in ['alpha', 'C', 'kernel', 'max_iter', 'solver']}
        times.append((name, vectorizer_type, training_time, relevant_hyperparams))

    return times

# Paso 4: Ejecutar el flujo de trabajo completo
def run_workflow():
    df = load_data()
    X_train = df['text']
    y_train = df['label']
    X_train_tfidf, X_test_tfidf, X_train_bow, X_test_bow, X_train_ngrams, X_test_ngrams = transform_data(X_train, X_train)

    ngrams_times = train_and_save_models(X_train_ngrams, y_train, 'n-grams')
    tfidf_times = train_and_save_models(X_train_tfidf, y_train, 'TF-IDF')
    bow_times = train_and_save_models(X_train_bow, y_train, 'Bag of Words')

    # Crear DataFrame con los tiempos
    times_df = pd.DataFrame(ngrams_times + tfidf_times + bow_times,
                            columns=['Modelo', 'Vectorizer', 'Tiempo de Entrenamiento (segundos)', 'Hiperparámetros'])

    times_df.to_excel('training_times.xlsx', index=False)
    print("\nTiempos de entrenamiento guardados en 'training_times.xlsx'.")

run_workflow()
