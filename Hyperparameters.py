import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Función para cargar los datos
def load_data():
    archivo = 'train_data.csv'
    df = pd.read_csv(archivo)
    return df['text'], df['label']

# Función para transformar los datos
def transform_data(X_train):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    bow_vectorizer = CountVectorizer(max_features=1000)
    ngram_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 3))

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_train_ngrams = ngram_vectorizer.fit_transform(X_train)

    return X_train_tfidf, X_train_bow, X_train_ngrams

# Función para buscar los mejores hiperparámetros
def find_best_hyperparameters(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params, round(best_score, 2)

# Paso 4: Ejecutar el flujo de trabajo completo
def run_workflow():
    X_train, y_train = load_data()
    X_train_tfidf, X_train_bow, X_train_ngrams = transform_data(X_train)

    # Define los modelos
    models = {
        "Naive_Bayes": MultinomialNB(),
        "SVM": SVC(),
        "Logistic_Regression": LogisticRegression()
    }

    # Define los espacios de búsqueda de hiperparámetros
    param_grids = {
        "Naive_Bayes": {'alpha': [0.01, 0.1, 1, 5, 10]},
        "SVM": {'C': [0.1, 1, 5, 10], 'kernel': ['linear', 'rbf']},
        "Logistic_Regression": {'max_iter': [1000, 2000, 5000], 'solver': ['lbfgs', 'liblinear']}
    }

    for model_name, model in models.items():
        if model_name == "Naive_Bayes":
            best_params, best_score = find_best_hyperparameters(X_train_tfidf, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with TF-IDF: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_bow, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with Bag of Words: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_ngrams, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with N-grams: {best_params} with accuracy: {best_score}\n")
        elif model_name == "SVM":
            best_params, best_score = find_best_hyperparameters(X_train_tfidf, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with TF-IDF: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_bow, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with Bag of Words: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_ngrams, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with N-grams: {best_params} with accuracy: {best_score}\n")
        else:
            best_params, best_score = find_best_hyperparameters(X_train_tfidf, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with TF-IDF: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_bow, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with Bag of Words: {best_params} with accuracy: {best_score}")
            best_params, best_score = find_best_hyperparameters(X_train_ngrams, y_train, model, param_grids[model_name])
            print(f"Best parameters for {model_name} with N-grams: {best_params} with accuracy: {best_score}")

run_workflow()
