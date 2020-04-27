from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from rouge import Rouge
import json
import nltk
import os
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim import corpora
import numpy as np
from scipy.sparse import coo_matrix
import lda

# summary of entire document and calculating the recall from rouge_l and rouge_1 for entire dataset
LANGUAGE = "english"
SENTENCES_COUNT = 5
topics_directory = "/content/drive/My Drive/Text_Summarisation/topics"
      
if __name__ == "__main__":
    count = 0
    total_recall_l = 0
    total_recall_1 = 0
    total_recall_2 = 0
    r = Rouge()
    for filename in os.listdir(topics_directory):
      try:
        filepath = str(topics_directory) + "/" + str(filename)
        parser = PlaintextParser.from_file(filepath, Tokenizer(LANGUAGE))
        
        stemmer = Stemmer(LANGUAGE)
        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        system_summary = ""
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            system_summary += str(sentence)
        
        filepath = filename
        filepath = os.path.splitext(filepath)[0]
        filepath = os.path.splitext(filepath)[0]
        gold_directory = "/content/drive/My Drive/Text_Summarisation/summaries-gold/" + filepath
        file_recall_l = 0
        file_recall_1 = 0
        file_recall_2 = 0
        for goldfile in os.listdir(gold_directory):
            gold_summary = open(str(gold_directory) + "/" + str(goldfile), "r").read()
            scores = r.get_scores(system_summary, gold_summary)
            recall_l = scores[0]['rouge-l']['r']
            recall_1 = scores[0]['rouge-1']['r']
            recall_2 = scores[0]['rouge-2']['r']
            file_recall_l += recall_l
            file_recall_1 += recall_1
            file_recall_2 += recall_2
        file_recall_l = file_recall_l/5
        file_recall_1 = file_recall_1/5
        file_recall_2 = file_recall_2/5
        total_recall_l += file_recall_l
        total_recall_1 += file_recall_1
        total_recall_2 += file_recall_2
      except UnicodeDecodeError:
        count+=1
    print("Files containing 0x92 char: " + str(count))
    count = 51 - count
    print("Total recall from rouge_l: " + str(total_recall_l/count))
    print("Total recall from rouge_1: " + str(total_recall_1/count))
    print("Total recall from rouge_2: " + str(total_recall_2/count)) #rouge-2 is not a good measure for extractive summarisation