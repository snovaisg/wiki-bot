import random

from nltk.corpus import stopwords


class baseline_greeter():
    def __init__(self, language="English"):
        self.stopwords = set(stopwords.words(language))

    def __call__(self, raw_text):
        return self.response(raw_text)

    def response(self, raw_text):
        GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
        GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
        for word in raw_text.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)
        return None
