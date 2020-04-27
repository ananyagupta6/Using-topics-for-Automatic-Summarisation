# Using-topics-for-Automatic-Summarisation

In this project, I have performed automatic extractive summarisation of a text by making use of topic modeling.

Here are some brief details about this project:

## Dataset:
I have used Opinions dataset. This dataset contains sentences extracted from reviews on a given item. Example reviews are “performance of Toyota Camry” and “sound quality of ipod nano”, etc. There are 51 such documents in this dataset and the opinions are obtained from Tripadvisor(hotels), Edmunds.com(cars) and Amazon.com(various electronics). There are approximately 100 sentences per document. This dataset also contains gold standard summaries. These gold summaries are crucial for evaluating a summary. There are 5 gold summaries available per document. 

Link to Opinions dataset:
http://kavita-ganesan.com/opinosis-opinion-dataset/#.XqbuTy0Q3X9

This dataset was used for the following paper:
Kavita Ganesan, ChengXiang Zhai, and Jiawei Han, "Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions", Proceedings of the 23rd International Conference on Computational Linguistics (COLING 2010), Beijing, China, 2010.

## Methodology:
•	First, I found the summary of each document in the dataset as a whole using Sumy library in python. Let’s say this is summary#1 for each document.

•	Then, I have divided the text into topics using LDA algorithm, found individual summaries of text in each topic using Sumy, and then combined these summaries into one summary. This is summary#2. I plan on doing the same using LSA and NMF algorithms as well to compare the results of each against each other.

## Evaluation of summaries:
•	I have applied the above 2 approaches for each document in the dataset, and evaluated each summary against the gold summaries using ROUGE score, cosine similarity and unit overlap metrics. Then I have averaged these scores for all the documents in the dataset to get the final result of performance.

•	The scores indicate that the result was better using LDA topics, compared to without using topics.

Hence, the summarisation system with LDA topics performed somewhat better than without using any topics.




