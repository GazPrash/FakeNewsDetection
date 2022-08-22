import os, joblib


def classify_news(articles:list, model_signature):
    data = []
    for article in articles:
        data.append(article.strip())

    base_path_model = "src/model/saved_models"
    base_path_vocab = "src/model/others"

    clf_model_obj = joblib.load(f"{base_path_model}/model{model_signature}.pkl")
    vector_vocab_obj = joblib.load(f"{base_path_vocab}/vocab{model_signature}.pkl")

    data = vector_vocab_obj.transform(data).toarray()
    results:list = clf_model_obj.predict(data)
    # results.append(clf_model_obj.accuracy)

    return results





