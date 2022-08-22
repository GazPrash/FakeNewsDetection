import pandas as pd
import time, secrets, joblib
from utils import vitals
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class Classifier:
    def __init__(self) -> None:
        self.clf_model = MultinomialNB()
        self.vectorizer = CountVectorizer()

        self.dataset = None
        self.backup_dataset = None
        self.accuracy = None
        self.selection_data = []


    def load_samples(self, primary_datasets:list, file_signature, return_data = False):
        data = vitals.load_data(
            primary_datasets,
            file_signature, 
            orientation="rows"
        )
        data_samples = vitals.prepare_data_for_t_n_t(data, samples = 2)
        if self.dataset is None and self.backup_dataset is None:
            self.dataset, self.backup_dataset = data_samples

        if return_data:
            return self.dataset

    def generate_corpus(self, data, fitted:bool):
        if not fitted:
            corpus_matrix = self.vectorizer.fit_transform(data).toarray()
        else:
            corpus_matrix = self.vectorizer.transform(data).toarray()

        return corpus_matrix
        
    def filter_classes(self, feature_label:str, target_label:str):
        # In this model, the feature will remain a 1D Array as it undergoes CVectorizer afterwards
        if self.dataset is not None:
            features, target = self.dataset[feature_label], self.dataset[target_label]
        
        return (features, target)
    
    def train_and_test_data(self):
        x, y = self.filter_classes(
            feature_label="text", 
            target_label="fraudulent"
        )

        self.selection_data = train_test_split(x.values, y.values)

        corpus_training = self.generate_corpus(self.selection_data[0], fitted = False)
        corpus_testing = self.generate_corpus(self.selection_data[1], fitted = True)

        y_train, y_testing = self.selection_data[2:]
        self.clf_model.fit(corpus_training, y_train)
        if self.accuracy is None:
            self.accuracy = self.clf_model.score(corpus_testing, y_testing)

        return self.accuracy


    def predict(self, test_case, *args, **kwargs):
        returnArray = False
        if isinstance(test_case, list): 
            returnArray = True
        
        if returnArray:
            results = []
            for case in test_case:
                corpus = self.generate_corpus([case], fitted=True)
                prediction = self.clf_model.predict(corpus)
                results.append(prediction)
            return results

        else:
            corpus = self.generate_corpus([test_case], fitted = True)
            return self.clf_model.predict(corpus)

    def save_model(self):
        file_signature = f"{secrets.token_hex(8)}_{int(time.time())}"
        if self.clf_model is not None:
            joblib.dump(self.clf_model, f"src/model/saved_models/model{file_signature}.pkl")

        return file_signature
    
    def save_vector_vocabulary(self, corres_model_signature):
        joblib.dump(self.vectorizer, f"src/model/others/vocab{corres_model_signature}.pkl")


if __name__ == "__main__":
    clf = Classifier()
    clf.load_samples(None, None)
    acc = clf.train_and_test_data()
    print(f"Training & Testing Complete. Accuracy : {acc}")
    file_signature = clf.save_model()
    clf.save_vector_vocabulary(corres_model_signature=file_signature)
    print(f"Model [{file_signature}] was sucessfully saved.")


