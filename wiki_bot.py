import nltk
import wikipedia
from sklearn.feature_extraction.text import \
    TfidfVectorizer  # convert a collection of raw documents to a matrix of TF-IDF features
from sklearn.metrics.pairwise import cosine_similarity

from greeters.baseline_greeter import baseline_greeter
from tokenizers.baseline_tokenizer import baseline_tokenizer


class wiki_bot():

    def __init__(self, tokenizer=baseline_tokenizer, greeter=baseline_greeter, page="Nova University Lisbon"):
        self.page = wikipedia.page(page).content
        self.sent_tokens = nltk.sent_tokenize(self.page)  # converts to list of sentences
        self.word_tokens = nltk.word_tokenize(self.page)  # converts to list of words
        self.tokenizer = tokenizer()
        self.greeter = greeter()

    # this tokenizer should receive raw text and output a list of documents
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer()

    # this method should receive a raw text and decide whether to and how to greet
    def set_greeter(self, greeter):
        self.greeter = greeter()

    def response(self, docs):
        robo_response = ''

        TfidfVec = TfidfVectorizer(tokenizer=self.tokenizer)
        tfidf = TfidfVec.fit_transform(docs)
        vals = cosine_similarity(tfidf[-1], tfidf)
        best_match_idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if (req_tfidf == 0):
            robo_response = robo_response + "I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response + self.sent_tokens[best_match_idx]
            return robo_response

    def play(self):

        flag = True
        print("ROBO: My name is Robo. What do you want to know? (if you want to exit, press \"bye\")")
        while (flag == True):
            user_response = input()
            user_response = user_response.lower()
            if (user_response != 'bye'):
                if (user_response == 'thanks' or user_response == 'thank you'):
                    flag = False
                    print("ROBO: You are welcome..")
                else:
                    if self.greeter.__call__(user_response) != None:
                        print("ROBO: " + self.greeter.__call__(user_response))
                    else:
                        self.sent_tokens.append(user_response)
                        print("ROBO: ", end="")
                        print(self.response(self.sent_tokens))
                        self.sent_tokens.remove(user_response)
            else:
                flag = False
                print("ROBO: Bye! take care..")
