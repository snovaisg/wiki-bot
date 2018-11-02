import string  # to process standard python strings

import nltk
from nltk.corpus import stopwords


class baseline_tokenizer():
    def __init__(self, language="English"):
        self.stopwords = set(stopwords.words(language))

    def __call__(self, raw_text):
        return self.transform(raw_text)

    def transform(self, raw_text):
        def LemTokens(lem, tokens):
            return [lemmer.lemmatize(token) for token in tokens if token not in self.stopwords]

        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

        lemmer = nltk.stem.WordNetLemmatizer()

        return LemTokens(lemmer, nltk.word_tokenize(raw_text.lower().translate(remove_punct_dict)))