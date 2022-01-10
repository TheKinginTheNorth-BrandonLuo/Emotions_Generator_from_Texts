import re
import json
import pickle
import numpy
import pandas
import spacy
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
import nltk
import networkx as nx
nlp = spacy.load('en_core_web_lg')
nlp.max_length = 50000000
data = []
with open('depression') as f:
    for line in f:
        data.append(json.loads(line))


i = 0
j = 0
dp = []
for i in range(len(data)):
    dp.append(data[i][0])
    
pos = []
neg = []
for j in range(len(dp)):
    if dp[j]['label'] == 'control':
        neg.append(dp[j])
    else:
        pos.append(dp[j])

m = 0
n = 0
for m in range(len(pos)):
    pos[m] = pos[m]['posts']
    
for n in range(len(neg)):
    neg[n] = neg[n]['posts']
    
h=0
f=0
for h in range(len(pos)):
    for f in range(len(pos[h])):
        pos[h][f] = pos[h][f][1]
        
g=0
e=0
for g in range(len(neg)):
    for e in range(len(neg[g])):
        neg[g][e] = neg[g][e][1]

res = [''.join(ele) for ele in pos] 
ans = [''.join(ele) for ele in neg]

res_1 = ' '.join(res)

d_1 = {'date': [1],'email': res_1, 'sender': 'user1','time':[1]}
df_text = pd.DataFrame(data=d_1)


def read_json_file(fn):
    f = open(fn, 'r')
    data = json.load(f)
    f.close()
    return data


def save_json_file(data, fn):
    f = open(fn, 'w')
    json.dump(
        data, f,
        sort_keys=True,
        indent=4,
        separators=(',', ': '),
    )
    f.close()
    
    
def read_pickle_file(fn):
    f = open(fn, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def save_pickle_file(data, fn):
    f = open(fn, 'wb')
    pickle.dump(data, f, protocol=2)
    f.close()
    
# df_text = df_1
# nlp.max_length = len(df_1['email'][0])

def remove_errors_from_text(text):
    text = text.replace('=20\r\n', ' ')
    text = text.replace('=\r\n', '')
    text = text.replace('\r\n', ' ')
    text = text.replace('\n', ' ')
    return text

def process_text(df_text):
    data = []
    vocabulary = set([])
    
    for index,row in df_text.iterrows():
        text = remove_errors_from_text(row['email'])
        doc = nlp(text)
        sentences = []

        key_neg = 'neg'
        key_token = 'token'
        key_en = 'en'
        key_text = 'text'

        for s in doc.sents:
            sent = {key_neg:0, key_token:[],}
            for token in s:
                if token.is_alpha:
                    if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'VERB']:
                        if token.is_stop:
                            pass
                        else:
                            sent[key_token].append(token.lemma_)
                            vocabulary.add(token.lemma_)
                        if token.dep_ == 'neg':
                            sent[key_neg] += 1
            sent[key_text] = ' '.join(sent[key_token])
            sentences.append(sent)
        d = {'date':row['date'], 'sender':row['sender'], 'time':row['time']}
        d['text'] = sentences
        data.append(d)
    return data, vocabulary

enron_data, vocab = process_text(df_text)

def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
                
    synonyms = list(set(synonyms))
    synonyms.sort()
 
    return synonyms

wn_vocabulary = {}

for w in vocab:
    syn = get_synonyms(w)
    wn_vocabulary[w] = syn
    
def add_nodes(w, d):
    if w in d:
        d[w]['playcount'] += 1
    else:
        d[w] = {
            'name':w,
            "match": 1.0,
            "artist": w,
            "id": w,
            'playcount':1,
        }

def gather_network_json(wn_vocabulary):
    nodes = {}
    links = set()
    
    for w in wn_vocabulary:
        syn = wn_vocabulary[w]
        if len(syn) > 0:
            add_nodes(w, nodes)
            for s in syn:
                if s != w:
                    if ('_' not in s) and ('-' not in s):
                        add_nodes(s, nodes)
                        edge  = (w,s)
                        redge = (s,w)
                        if redge not in links:
                            links.add(edge)
                            nodes[w]['playcount'] += 1
                    
    nodes_list = []
    links_list = []
    for n in nodes:
        nodes_list.append(nodes[n])
    for w,s in list(links):
        links_list.append({
            "source": w,
            "target": s,
        })
        
    data = {'nodes':nodes_list, 'links':links_list}
    return nodes, links, data

wn_nodes, wn_links, wn_net = gather_network_json(wn_vocabulary)
G = nx.Graph()
G.add_edges_from(list(wn_links))

#dictionary_path = os.path.join(os.path.dirname(cwd), 'data')

#emo12_liwc = os.path.join(dictionary_path, 'twelve_emotions_liwc.json')
#emo12_liwc = read_json_file(emo12_liwc)
emo12_liwc = read_json_file('twelve_emotions_liwc.json')
#emo12_extd = read_pickle_file(os.path.join(dictionary_path, 'extended_emotions.pkl'))
emo12_extd = read_pickle_file('extended_emotions.pkl')
enron_words_list = [w for w in wn_nodes]
enron_words_str = ' '.join(enron_words_list)

def is_word(t):
    reg = r'[a-zA-Z]+'
    r = re.match(reg, t)
    if r is None:
        return False
    return True

def search_minimal_item(t, s):
    reg = t
    if '*' in reg:
        reg = reg.replace('*', '')
        if is_word(reg):
            reg = r'\s+('+r'{0}'.format(reg)+r'\S*)'
            r = re.findall(reg, s)
            return r
    else:
        if is_word(reg):
            return [reg] 
    return []

emo12_category = list(emo12_liwc.keys())
emo12_category.sort()

emo12_enron = {}

for category in emo12_category:
    group = emo12_liwc[category]
    group.sort()
    emo12_enron[category] = {}
    for w in group:
        extend_words = search_minimal_item(w, enron_words_str)
        extend_words = list(set(extend_words))
        extend_words.sort()
        emo12_enron[category][w] = extend_words

emo12_enron_list = {}
for category in emo12_enron:
    group = emo12_enron[category]
    emo12_enron_list[category] = []
    for key in group:
        ls = group[key]
        emo12_enron_list[category].extend(ls)

emo12_enron_list_neighbors = {}
emo12_enron_list_neighbors_dict = {}
word_nodes = G.nodes()

for category in emo12_category:
    group = emo12_enron_list[category]
    emo12_enron_list_neighbors[category] = []
    emo12_enron_list_neighbors_dict[category] = {}
    for w in group:
        if w in word_nodes:
            for i in G.neighbors(w):
                ii = i.lower()
                emo12_enron_list_neighbors_dict[category][w] = []
                if ii not in group:
                    emo12_enron_list_neighbors[category].append(ii)
                    emo12_enron_list_neighbors_dict[category][w].append(ii)
                    
                    
for category in emo12_category:
    emo12_enron_list_neighbors[category] = list(set(emo12_enron_list_neighbors[category]))
    emo12_enron_list_neighbors[category].sort()

enron_emo12_word_label = {}

def found_emotion(w, d):
    for c in d:
        if w in d[c]:
            return c
        
    return 'NA'

for w in wn_nodes:
    e = found_emotion(w, emo12_enron_list)
    if e == 'NA':
        e = found_emotion(w, emo12_extd)
        
    if e != 'NA':
        enron_emo12_word_label[w] = e
        
email_sent_data = []

def add_emotion(w, d, r, t):
    pre = 'Word-'
    
    if w not in d:
        return
    
    e = d[w]
    if e in r:
        r[e] += 1
        t[pre+e].append(w)
    else:
        r[e]  = 1
        t[pre+e] = [w]

for d in enron_data:
    d_sender  = d['sender']
    d_date    = d['date']
    d_time    = d['time']
    sentences = d['text']
    for sent in sentences:
        d_neg = sent['neg']
        words = sent['token']
        d_emo = {}
        d_tgt = {}
        for w in words:
            add_emotion(w, enron_emo12_word_label, d_emo, d_tgt)
        for k in d_tgt:
            d_tgt[k] = ' '.join(d_tgt[k])
        d_emo.update(d_tgt)
        d_emo['sender'] = d_sender
        d_emo['date']   = d_date
        d_emo['time']   = d_time
        d_emo['neg']    = d_neg
        d_emo['text']   = sent['text']
        
        email_sent_data.append(d_emo)
        
email_sent_data_df = pandas.DataFrame(email_sent_data)
email_sent_data_df.fillna(0, inplace=True)

email_sent_data_df.to_csv('guaji.csv', index=False)