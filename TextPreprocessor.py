import re

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from TextSummarizer import TextSummarizer


class TextAnalyzer:
    def __init__(self):
        self.similarityArray = []

    def extract_paragraphs(self, text):
        return re.split(r'\n+', text)

    def preprocess_paragraph(self, paragraph):
        # lowercase
        paragraph = paragraph.lower()
        # remove numbers
        paragraph = re.sub(r'\d', '', paragraph)
        # remove punctuation
        paragraph = re.sub(r'[^\w\s]', '', paragraph)
        # tokenize words
        words = word_tokenize(paragraph)
        # remove stop words
        words = [word for word in words if word not in stopwords.words('english')]
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # join words back into a single string
        # preprocessed_paragraph = ' '.join(words)
        return words

    def preprocess_query(self, query):
        # lowercase
        query = query.lower()
        # remove numbers
        query = re.sub(r'\d', '', query)
        # remove punctuation
        query = re.sub(r'[^\w\s]', '', query)
        # tokenize words
        words = word_tokenize(query)
        # remove stop words
        words = [word for word in words if word not in stopwords.words('english')]
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # join words back into a single string
        preprocessed_query = ' '.join(words)
        return preprocessed_query


    def evaluate_correlation(self, query, paragraphs):
        # Preprocess the query and paragraphs
        preprocessed_query = self.preprocess_query(query)
        preprocessed_paragraphs = self.preprocess_paragraph(paragraphs)

        # Create a Tf-Idf matrix of the paragraphs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs)

        # Compute cosine similarity between the query and each paragraph
        query_vector = vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix)

        # Return the index of the most similar paragraph
        return similarities.argmax()

    #runner class
    def runnerClass(self, textArr, userQ):
        for sent in textArr:
            self.similarityArray.append(self.evaluate_correlation(userQ, sent))

        senten_similarity_dict = {}

        for i in range(len(textArr)):
            key = textArr[i]
            val = self.similarityArray[i]
            print(val)
            if val != 0:
                senten_similarity_dict[key] = val

        sorted_dict = sorted(senten_similarity_dict.items(), key=lambda item: item[1], reverse=True)

        completeSent = ""
        for sent in sorted_dict:
            completeSent += sent[0]

        summarizer = TextSummarizer()
        summarizer.summarizeSentence(completeSent)
        summarizedSentence = summarizer.returntheSummarizePara()
        return summarizedSentence


# analyzer = TextAnalyzer()
# # #
# # #
# # #
# textArr = ["However, building and operating quantum computers is extremely challenging due to the fragile nature of qubits and the difficulty of controlling and measuring them. One of the key challenges in building quantum computers is maintaining the coherence of the qubits. This refers to the ability of the qubits to remain in a superposition of states without being disturbed by their environment. In order to maintain coherence, researchers use a variety of techniques such as cooling the qubits to extremely low temperatures, shielding them from external electromagnetic fields, and developing error-correcting codes. Another challenge is the difficulty of performing operations on the qubits, which requires extremely precise control and measurement techniques. Despite these challenges, quantum computing is a rapidly advancing field with many exciting possibilities for the future."]
# userQ = "Quantum Computing"
# #
# print(analyzer.runnerClass(textArr,userQ))

# extract_para = analyzer.extract_paragraphs(text)
# val = analyzer.evaluate_correlation(extract_para[0],"Administration")
# print(val)

# sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
# footballers_goals = {'Eusebio': 120, 'Cruyff': 104, 'Pele': 150, 'Ronaldo': 132, 'Messi': 125}
#
# sorted_footballers_by_goals = sorted(footballers_goals.items(), key=lambda x:x[1])
# print(sorted_footballers_by_goals)