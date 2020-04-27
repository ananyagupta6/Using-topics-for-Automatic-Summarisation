# Using-topics-for-Automatic-Summarisation

In this project, I have tried to implement the paper "Using Rhetorical Topics for Automatic Summarisation" by Natalie M. Schrimpf by adding in my own changes and modifications. 

Here are some brief details about this project:

## Dataset:
In the paper, author has used RST Discourse Treebank. This corpus contains 385 Wall Street Journal articles that have been annotated with RST structure. Most RST research is performed on this dataset only since it requires these RST annotations. This dataset also contains gold standard summaries. These gold summaries are crucial for evaluating a summary. This dataset is however, paid and only available on LDC (Linguistics Data Consortium).

Hence, since I could not use this dataset, and there is a special need for a dataset with gold summaries for any summarization task, I used Opinions dataset. This dataset contains sentences extracted from reviews on a given item. Example reviews are “performance of Toyota Camry” and “sound quality of ipod nano”, etc. There are 51 such documents in this dataset and the opinions are obtained from Tripadvisor(hotels), Edmunds.com(cars) and Amazon.com(various electronics). There are approximately 100 sentences per document. There are 5 gold summaries also available per document. However, since this dataset is not RST annotated and I could not find any such datasets for free, I could not implement the RST topic modeling part of this paper. 

## Methodology:
•	First, the author found the summary of each document in the dataset as a whole using Sumy library in python. Let’s say this is summary#1 for each document. I have implemented this for my dataset.

•	Then, the author has divided the text into RST topics (algorithm in the paper), found individual summaries of text in each topic using Sumy, and then combined these summaries into one summary. This is summary#2. I have implemented this using LDA algorithm instead of RST. I plan on doing the same using LSA and NMF algorithms as well to compare the results of each against each other.

•	The author also tried a third approach. She divided the text into random sub-parts to check whether the results were affected due to RST topics, or that simply dividing the text into smaller chunks and summarising them individually is improving the performance. This gives summary#3. She found that this does not affect the performance and it is due to RST topics that there is improvement. I have not implemented this approach yet, however it does seem a bit obvious that this will not affect the performance.

## Evaluation of summaries:
•	The author has applied the above 3 approaches for each document in the dataset, and evaluated each summary against the gold summaries using ROUGE score, cosine similarity and unit overlap metrics (Details of how these metrics work are mentioned in the paper). Then the author has averaged these scores for all the documents in the dataset to get the final result of performance.

•	The author’s results indicate that the result was better using RST topics, compared to without using topics and with random division of text.

•	In my implementation, I have calculated the average ROUGE-L, ROUGE-1, ROUGE-2 scores for all documents in the dataset for approach 1(no topics) and 2(LDA topics) each. ROUGE-2 is not considered a good metric when performing extractive summarisation but I am including the result anyway. I will also calculate average cosine similarity and average unit overlap next.

## Results:
METRIC	WITHOUT TOPICS	WITH LDA TOPICS

ROUGE-L	0.3912	0.4192

ROUGE-1	0.4358	0.4643

ROUGE-2	0.0660	0.0730

Hence, the summarisation system with LDA topics performed somewhat better than without using any topics.




