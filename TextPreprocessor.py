import re

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


class TextAnalyzer:
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
        preprocessed_paragraph = ' '.join(words)
        return preprocessed_paragraph

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


    def evaluate_correlation(self, query, paragraphs, TfidfVectorizer=None):
        # Preprocess the query and paragraphs
        preprocessed_query = self.preprocess_query(query)
        preprocessed_paragraphs = [self.preprocess_paragraph(para) for para in paragraphs]

        # Create a Tf-Idf matrix of the paragraphs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs)

        # Compute cosine similarity between the query and each paragraph
        query_vector = vectorizer.transform([preprocessed_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix)

        # Return the index of the most similar paragraph
        return similarities.argmax()

    # create an instance of TextAnalyzer
analyzer = TextAnalyzer()

# process text
text = "The Daman and Diu administration on Wednesday withdrew a circular that asked women staff to tie rakhis on male colleagues after the order triggered a backlash from employees and was ripped apart on social media.The union territory?s administration was forced to retreat within 24 hours of issuing the circular that made it compulsory for its staff to celebrate Rakshabandhan at workplace.?It has been decided to celebrate the festival of Rakshabandhan on August 7. In this connection, all offices/ departments shall remain open and celebrate the festival collectively at a suitable time wherein all the lady staff shall tie rakhis to their colleagues,? the order, issued on August 1 by Gurpreet Singh, deputy secretary (personnel), had said.To ensure that no one skipped office, an attendance report was to be sent to the government the next evening.The two notifications ? one mandating the celebration of Rakshabandhan (left) and the other withdrawing the mandate (right) ? were issued by the Daman and Diu administration a day apart. The circular was withdrawn through a one-line order issued late in the evening by the UT?s department of personnel and administrative reforms.?The circular is ridiculous. There are sensitivities involved. How can the government dictate who I should tie rakhi to? We should maintain the professionalism of a workplace? an official told Hindustan Times earlier in the day. She refused to be identified.The notice was issued on Daman and Diu administrator and former Gujarat home minister Praful Kodabhai Patel?s direction, sources said.Rakshabandhan, a celebration of the bond between brothers and sisters, is one of several Hindu festivities and rituals that are no longer confined of private, family affairs but have become tools to push politic al ideologies.In 2014, the year BJP stormed to power at the Centre, Rashtriya Swayamsevak Sangh (RSS) chief Mohan Bhagwat said the festival had ?national significance? and should be celebrated widely ?to protect Hindu culture and live by the values enshrined in it?. The RSS is the ideological parent of the ruling BJP.Last year, women ministers in the Modi government went to the border areas to celebrate the festival with soldiers. A year before, all cabinet ministers were asked to go to their constituencies for the festival."
query = "administration"

extract_para = analyzer.extract_paragraphs(text)
print(extract_para)