import pandas as pd
import numpy as np
import re

test_data = [
    ["Free money now", 1],
    ["Hello how are you", 0],
    ["Win a prize today", 1],
    ["Meeting at 3pm", 0],
    ["Urgent click here", 1],
    ["Your account statement", 0],
    ["Limited time offer", 1],
    ["Lunch tomorrow?", 0],
    ["You've won!", 1],
    ["Project update", 0],
    ["Act fast deal", 1],
    ["Team building event", 0],
    ["Cash bonus inside", 1],
    ["Weekly report", 0],
    ["Risk-free investment", 1],
    ["Coffee break?", 0],
    ["Guaranteed earnings", 1],
    ["Vacation photos", 0],
    ["No credit needed", 1],
    ["Dinner plans", 0]
]

test_df = pd.DataFrame(test_data, columns=["email", "spam"])

class NaiveBayesClassifier:
    def __init__(self, df: pd.DataFrame):
        self.probabilities = NaiveBayesClassifier._calculate_probabilities(df)
        self.spam_probability = NaiveBayesClassifier._get_spam_probability(df)
    @staticmethod
    def _calculate_probabilities(df: pd.DataFrame) -> dict:
        appearance = {}
        total = len(df)
        spam_num = df["spam"].sum()
        not_spam_num = total - spam_num
        def calculate(row: pd.Series):
            email = row["email"]
            email_words = re.sub(r'[^a-z ]', '', email.lower()).split()
            for word in email_words:
                if word not in appearance:
                    appearance[word] = {"spam": 0, "not spam": 0}
                appearance[word]["spam"] += row["spam"]
                appearance[word]["not spam"] += 1 - row["spam"]
        df.apply(calculate, axis=1)
        probabilities = {}
        for word in appearance:
            probability = {"spam": total*appearance[word]["spam"]/spam_num,
                           "not spam": total*appearance[word]["not spam"] / not_spam_num}
            probabilities[word] = probability
        return probabilities

    @staticmethod
    def _get_spam_probability(df: pd.DataFrame) -> float:
        return df["spam"].sum()/len(df)

    def classify(self, email: str) -> int:
        email_words = re.sub(r'[^a-z ]', '', email.lower()).split()
        numerator = np.prod([self.probabilities[word]["spam"] for word in email_words
                             if word in self.probabilities])*self.spam_probability
        denominator = numerator + (1-self.spam_probability) * \
                       np.prod([self.probabilities[word]["not spam"] for word in email_words
                                if word in self.probabilities])
        result = numerator/denominator if denominator != 0 else self.spam_probability
        return 1 if result >= 0.5 else 0

if __name__ == '__main__':
    nbc = NaiveBayesClassifier(test_df)
    print(nbc.classify("Free money sex viagra penis prize won!!!"))
    print(nbc.classify("Meeting dinner project hello money yes yes buisness"))