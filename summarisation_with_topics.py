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

def find_summary_def(topic_text):
  LANGUAGE = "english"
  SENTENCES_COUNT = 2

  parser2 = PlaintextParser.from_string(topic_text, Tokenizer(LANGUAGE))
  stemmer2 = Stemmer(LANGUAGE)
  summarizer2 = Summarizer(stemmer2)
  summarizer2.stop_words = get_stop_words(LANGUAGE)
  system_summary2 = ""
  for sentence in summarizer2(parser2.document, 2):
    system_summary2 += str(sentence)
    system_summary2 += '\n'
  return system_summary2

  # dividing text into topics using LDA and then summarising each topic, joining the summaries into one, and calculating rouge score
LANGUAGE = "english"
SENTENCES_COUNT = 5
topics_directory = "/content/drive/My Drive/Text_Summarisation/topics"
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

if __name__ == "__main__":
    count = 0
    total_recall_l = 0
    total_recall_1 = 0
    total_recall_2 = 0
    r = Rouge()
    for filename in os.listdir(topics_directory):
      try:
        filepath = str(topics_directory) + "/" + str(filename)
        text = open(filepath, "r").read()
        # split the document into individual sentences
        a_list = nltk.tokenize.sent_tokenize(text)
        # tokenise each sentence
        lst = []
        for sentence in a_list:
          lst.append(sentence.split())
        # remove punctuation and convert all letters to lowercase, remove words with less than 3 chars, remove stopwords, lemmatise and stem words
        tokenised_list = []
        index = 0
        for tokenised_sentence in lst:
          tokenised_list.append([])
          for i in range(0, len(tokenised_sentence)):
            if tokenised_sentence[i] in punctuations:
              tokenised_sentence[i] = ""
            else:
              tokenised_sentence[i] = tokenised_sentence[i].lower()
              if len(tokenised_sentence[i]) > 2 and tokenised_sentence[i] not in stopwords.words():
                tokenised_sentence[i] = lemmatizer.lemmatize(tokenised_sentence[i])
                tokenised_sentence[i] = porter.stem(tokenised_sentence[i])                
                tokenised_list[index].append(tokenised_sentence[i])
          index += 1
        #creating a dictionary from tokenised_list
        docs = {}
        for i in range(0,len(tokenised_list)):
          docs[i] = tokenised_list[i]
        # create DTM
        vocab = set([])
        n_nonzero = 0 # number of unique terms
        for docterms in docs.values():
          unique_terms = set(docterms)    # all unique terms of this doc
          vocab |= unique_terms           # set union: add unique terms of this doc
          n_nonzero += len(unique_terms)  # add count of unique terms in this doc
        # make a list of document names
        # the order will be the same as in the dict
        docnames = list(docs.keys())
        docnames = np.array(docnames)
        vocab = np.array(list(vocab))  
        vocab_sorter = np.argsort(vocab)    # indices that sort "vocab"
        ndocs = len(docnames)
        nvocab = len(vocab) # dimensions of dtm will be ndocs by nvocab
        data = np.empty(n_nonzero, dtype=np.intc)     # all non-zero term frequencies at data[k]
        rows = np.empty(n_nonzero, dtype=np.intc)     # row index for kth data item (kth term freq.)
        cols = np.empty(n_nonzero, dtype=np.intc)     # column index for kth data item (kth term freq.)
        
        ind = 0     # current index in the sparse matrix data
        # go through all documents with their terms
        for docname, terms in docs.items():
          # find indices into  such that, if the corresponding elements in  were
          # inserted before the indices, the order of  would be preserved
          # -> array of indices of  in 
          term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

          # count the unique terms of the document and get their vocabulary indices
          uniq_indices, counts = np.unique(term_indices, return_counts=True)
          n_vals = len(uniq_indices)  # = number of unique terms
          ind_end = ind + n_vals  #  to  is the slice that we will fill with data

          data[ind:ind_end] = counts                  # save the counts (term frequencies)
          cols[ind:ind_end] = uniq_indices            # save the column index: index in 
          doc_idx = np.where(docnames == docname)     # get the document index for the document name
          rows[ind:ind_end] = np.repeat(doc_idx, n_vals)  # save it as repeated value

          ind = ind_end  # resume with next document -> add data to the end

        dtm = coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)
        
        # run LDA on the dtm
        model = lda.LDA(n_topics=3, n_iter=1500, random_state=1)
        model.fit(dtm)
        topic_word = model.topic_word_
        n_top_words = 5

        for i, topic_dist in enumerate(topic_word):
          topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
          # print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        
        # find document topic distribution (which sentence belongs to which topic)
        doc_topic = model.doc_topic_
        topic1 = ""
        topic2 = ""
        topic3 = ""
        for i in range(len(docnames)):
          #print("{} (top topic: {})".format(docnames[i], doc_topic[i].argmax()))
          if doc_topic[i].argmax() == 0:
            topic1 += a_list[i]
            topic1 += '\n'
          elif doc_topic[i].argmax() == 1:
            topic2 += a_list[i]
            topic2 += '\n'
          else:
            topic3 += a_list[i]
            topic3 += '\n'
        
        # find summary of each topic
        final_summary = ""
        final_summary += find_summary_def(topic1)
        final_summary += find_summary_def(topic2)
        final_summary += find_summary_def(topic3)
        
        # find rouge score
        filepath = filename
        filepath = os.path.splitext(filepath)[0]
        filepath = os.path.splitext(filepath)[0]
        gold_directory = "/content/drive/My Drive/Text_Summarisation/summaries-gold/" + filepath
        file_recall_l = 0
        file_recall_1 = 0
        file_recall_2 = 0
        for goldfile in os.listdir(gold_directory):
            gold_summary = open(str(gold_directory) + "/" + str(goldfile), "r").read()
            scores = r.get_scores(final_summary, gold_summary)
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