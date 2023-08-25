import os
import pandas as pd
import re
import gensim
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import datetime

stop_words = stopwords.words('english')
#this is where you can put in words to remove from the search
stop_words = stop_words + ["phd", "op", "like", "know", "would", "also", "i'm", "i've", "i¢m", "really", "one", "make",
                           "well", "take", "using", "much", "got", "use", "get", "im", "getting", "feel", "dont", "&amp;x200b"]

def remove_stopwords(texts):
    return_list = []
    for word in texts:
        if word not in stop_words:
            return_list.append(word)
    return return_list

if __name__ == '__main__':
    path = path = os.getcwd() + "\\"
    keywords = []
    new_keyword = ""
    while (new_keyword != "done"):
        if (new_keyword != ""):
            keywords.append(new_keyword)
        new_keyword = input("Please enter a keyword (enter done with finished): ")
    permalink_dict = dict()
    folders = os.listdir(path)
    i = 0
    j = 0
    csv_dict = dict()
    comment_set = set()
    comment_dictionary = dict()
    for f in folders:
        print(i, "out of", len(folders))
        i = i + 1
        if ("csv" in f):
            json_data = pd.read_csv(path + f, encoding="latin1")
            json_data = json_data.fillna("")
            try:
                json_data['Body'] = json_data['Body'].map(lambda x: x.lower())
                json_data['Title'] = json_data['Title'].map(lambda x: x.lower())
                json_data['Selftext'] = json_data['Selftext'].map(lambda x: x.lower())
                json_data['Body'] = json_data['Body'].map(lambda x: re.sub('[\r\n]', ' ', x))
                json_data['Title'] = json_data['Title'].map(lambda x: re.sub('[\r\n]', ' ', x))
                json_data['Selftext'] = json_data['Selftext'].map(lambda x: re.sub('[\r\n]', ' ', x))
                for keyword in keywords:
                    json_data = json_data[json_data["Body"].str.contains(keyword) | json_data["Title"].str.contains(keyword) | json_data["Selftext"].str.contains(keyword)]
                json_data['Body'] = json_data['Body'].map(lambda x: re.sub('[\\\¢\+\=\*\-\#\(\)ã°â,\.!?]', '', x))
                json_data['Title'] = json_data['Title'].map(lambda x: re.sub('[\\\¢\+\=\*\-\#\(\)ã°â,\.!?]', '', x))
                json_data['Selftext'] = json_data['Selftext'].map(lambda x: re.sub('[\\\¢\+\=\*\-\#\(\)ã°â,\.!?]', '', x))
                json_data['Body'] = json_data['Body'].map(lambda x: re.sub('\s\s+', ' ', x))
                json_data['Title'] = json_data['Title'].map(lambda x: re.sub('\s\s+', ' ', x))
                json_data['Selftext'] = json_data['Selftext'].map(lambda x: re.sub('\s\s+', ' ', x))
                for index, row in json_data.iterrows():
                    comment_ls = [row["Title"], row["Selftext"], row["Body"]]
                    for c in comment_ls:
                        comment_set.add(c)
                        comment_dictionary[c] = dict()
                        comment_dictionary[c]["Date"] = datetime.datetime.fromtimestamp(row["Date"]).strftime('%c')
                        comment_dictionary[c]["Score"] = row["Score"]
                        comment_dictionary[c]["Permalink"] = row["Permalink"]
            except Exception as e:
                print("No data found in", f)
    comment_list = list(comment_set)
    data_words = []
    for c in comment_list:
            c = c.split(" ")
            data = remove_stopwords(c)
            data_words.append(data)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    # number of topics
    num_topics = 10
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    doc_lda = lda_model[corpus]
    visualisation = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
    # Visualize the topics
    # setup: get the model's topics in their native ordering...
    all_topics = lda_model.print_topics()
    # ...then create a empty list per topic to collect the docs:
    docs_per_topic = [[] for _ in all_topics]
    # now, for every doc...
    for doc_id, doc_bow in enumerate(corpus):
        # ...get its topics...
        doc_topics = lda_model.get_document_topics(doc_bow)
        # ...& for each of its topics...
        for topic_id, score in doc_topics:
            # ...add the doc_id & its score to the topic's doc list
            docs_per_topic[topic_id].append((doc_id, score))

    topics = lda_model.print_topics()
    i = 0
    for doc_list in docs_per_topic:
        current_topic = topics[i]
        i = i + 1
        current_topic = current_topic[1]
        current_topic = current_topic.split('"')
        j = 0
        print("salient topics: ", end="")
        for c in current_topic:
            if ((j % 2) == 1):
                print(c, end=", ")
            j = j + 1
        print("")
        doc_list.sort(key=lambda id_and_score: id_and_score[1], reverse=True)
        shortened_list = doc_list[:10] #this is where you will change the number of comments that you want to return
        for (s, t) in shortened_list:
            print("Date: ", comment_dictionary[comment_list[s]]["Date"], "Score: ",
                  comment_dictionary[comment_list[s]]["Score"], "Permalink: ",
                  comment_dictionary[comment_list[s]]["Permalink"])
            print(comment_list[s])
        print("********************")
