---
title:  "Buffalo Data Science Talk"
date:   2017-04-29 15:04:23
categories: [data-analysis]
tags: [data-anlysis]
header:
  image: /images/bg.JPG
---

I recently gave a talk at a [Buffalo Data Science Meetup](https://bufdatascience.github.io/text-analytics-in-python/) on Text Analytics in Python. It's adapted from my post on [Feature Extraction from Text](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/) with some added material and an example.

<h2>Intro to Text Analytics in Python</h2>

* Terminology
* Bag of Words Model
* TF-IDF Model
* Preprocessing and Hyperparameters
* Example
* N-gram model

<h3>Terminology</h3>

* Document - a single string of text information

* Corpus - a collection of documents

* Token - a word, phrase or symbol derived from a document

* Tokenizer - function to split a document into a list of tokens


```python
# Example corpus
messages = ["Hey hey hey lets go get lunch today :)",
           "Did you go home?",
           "Hey!!! I need a favor"]
```


```python
# Example document
document = messages[0]
document
```




    'Hey hey hey lets go get lunch today :)'




```python
# Creating tokens
document.split(' ')
```




    ['Hey', 'hey', 'hey', 'lets', 'go', 'get', 'lunch', 'today', ':)']



<h3>Bag of Words Model</h3>

* need a numerical representation for our corpus
* will use CountVectorizer() from sci-kit learn library
* creates matrix of token counts


```python
# import and instantiate CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```

* next we will use fit() and transform() methods
* similar to fit() and predict() used in ML classifiers


```python
vect.fit(messages)
```




    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)




```python
# before transforming look at feature names (columns names)
print vect.get_feature_names()
print 'Number of tokens: {}'.format(len(vect.get_feature_names()))
```

    [u'did', u'favor', u'get', u'go', u'hey', u'home', u'lets', u'lunch', u'need', u'today', u'you']
    Number of tokens: 11
    

Things to note:
* all lowercase
* words less than two letters are excluded
* punctuation removed
* no duplicates

Next, we'll use the transform() method to create a document term matrix(DTM). This is the matrix of token counts we want to create.


```python
dtm = vect.transform(messages)
repr(dtm)
```




    "<3x11 sparse matrix of type '<type 'numpy.int64'>'\n\twith 13 stored elements in Compressed Sparse Row format>"




```python
print dtm
```

      (0, 2)	1
      (0, 3)	1
      (0, 4)	3
      (0, 6)	1
      (0, 7)	1
      (0, 9)	1
      (1, 0)	1
      (1, 3)	1
      (1, 5)	1
      (1, 10)	1
      (2, 1)	1
      (2, 4)	1
      (2, 8)	1
    

* Because each document has a column for every word that occurs in the corpus, DTM is predominatly filled with 0's
* Sparse format can store the DTM in a smaller amount of memory and can speed up operations
* a DTM of a large corpus can quickly balloon in size


```python
import pandas as pd
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>did</th>
      <th>favor</th>
      <th>get</th>
      <th>go</th>
      <th>hey</th>
      <th>home</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
      <th>today</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get total counts for corpus
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()).sum()
```




    did      1
    favor    1
    get      1
    go       2
    hey      4
    home     1
    lets     1
    lunch    1
    need     1
    today    1
    you      1
    dtype: int64



What happens if we get a new message?


```python
new_message = ['Hey lets go get a drink tonight']
new_dtm = vect.transform(new_message)
pd.DataFrame(new_dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>did</th>
      <th>favor</th>
      <th>get</th>
      <th>go</th>
      <th>hey</th>
      <th>home</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
      <th>today</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* only tokens from original fit appear as features(columns)
* need to refit with new message included


```python
messages.append(new_message[0])
messages
```




    ['Hey hey hey lets go get lunch today :)',
     'Did you go home?',
     'Hey!!! I need a favor',
     'Hey lets go get a drink tonight']




```python
dtm = vect.fit_transform(messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>did</th>
      <th>drink</th>
      <th>favor</th>
      <th>get</th>
      <th>go</th>
      <th>hey</th>
      <th>home</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
      <th>today</th>
      <th>tonight</th>
      <th>you</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<h3>TF-IDF Model</h3>

* term frequency inverse document frequency
* generally more popular than bag of words model
* numerical statistic to show how important an token is to a document
* TF-IDF = term frequency * (1 / document frequency)
* TF - how frequent a term(token) occurs in a document
* IDF - inverse of how frequent a term occurs across documents


```python
from sklearn.feature_extraction.text import TfidfVectorizer
def createDTM(messages):
    vect = TfidfVectorizer()
    dtm = vect.fit_transform(messages) # create DTM
    
    # create pandas dataframe of DTM
    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names()) 
```


```python
messages = ["Hey lets get lunch :)",
           "Hey!!! I need a favor"]
createDTM(messages)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favor</th>
      <th>get</th>
      <th>hey</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.534046</td>
      <td>0.379978</td>
      <td>0.534046</td>
      <td>0.534046</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.631667</td>
      <td>0.000000</td>
      <td>0.449436</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.631667</td>
    </tr>
  </tbody>
</table>
</div>



* `'hey'` has lowest value, only word that occurs in both documents
* `'favor'` and `'need'` have highest, occur in 1 document with fewest tokens


```python
# add repeats of 'hey' to first message
messages = ["Hey hey hey lets get lunch :)",
           "Hey!!! I need a favor"]
createDTM(messages)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favor</th>
      <th>get</th>
      <th>hey</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.363788</td>
      <td>0.776515</td>
      <td>0.363788</td>
      <td>0.363788</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.631667</td>
      <td>0.000000</td>
      <td>0.449436</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.631667</td>
    </tr>
  </tbody>
</table>
</div>



* TF for `'hey'` in first increases, but IDF for `'hey'` remains the same


```python
# remove 'hey' from second message
messages = ["Hey hey hey lets get lunch :)",
           "I need a favor"]
createDTM(messages)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favor</th>
      <th>get</th>
      <th>hey</th>
      <th>lets</th>
      <th>lunch</th>
      <th>need</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.288675</td>
      <td>0.866025</td>
      <td>0.288675</td>
      <td>0.288675</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.707107</td>
    </tr>
  </tbody>
</table>
</div>



* `'hey'` for first message is now the highest value
* `'favor'` and `'need'` also increase as there are now fewer tokens in the second message

<h3>Preprocessing and Hyperparameters</h3>

* max_features = n : only considers the top n words when ordered by term frequency
* min_df = n : ignores words with a document frequency below n
* max_df = n : ignores words with a document frequency above n
* stop_words = [''] : ignores common words like `'the'`, `'that'`, `'which'` etc.


```python
vect = CountVectorizer(stop_words='english')
print vect.get_stop_words()
```

    frozenset(['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through', 'yourselves', 'fify', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'go', 'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'your', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once'])
    


```python
# defining our own stopwords
my_words = ['buffalo','data','science']
vect = CountVectorizer(stop_words=my_words)
print vect.get_stop_words()
```

    frozenset(['data', 'buffalo', 'science'])
    

Word Stemming
* reduces a word down to its base/root form
* crude heuristic that works by chopping off end of word


```python
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokens = ['manufactured','manufacturing','manufacture']
```


```python
stems = [stemmer.stem(i) for i in tokens]
print stems
```

    [u'manufactur', u'manufactur', u'manufactur']
    

Word Lemmatization
* similar to stemming
* seeks to find base dictionary form
* more complex, may need to specify part of speech for accurate results


```python
from nltk import WordNetLemmatizer
lemmer = WordNetLemmatizer()
tokens = ['hands','women']
```


```python
lemmas = [lemmer.lemmatize(i) for i in tokens]
print lemmas
```

    [u'hand', u'woman']
    


```python
lemmer.lemmatize('manufacturing')
```




    'manufacturing'




```python
# specify it as a verb, default is noun
lemmer.lemmatize('manufacturing','v')
```




    u'manufacture'



<h2>Example</h2>

* dataset of song lyrics from 4 different artists (Beatles, Metallica, Eminem, Bob Dylan)
* we will use a vectorizer and then try to plot
* would expect similar songs to be close together


```python
df = pd.read_csv('lyrics.txt', sep='\t')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beatles</td>
      <td>Help!</td>
      <td>(When) When I was younger (When I was young) s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Beatles</td>
      <td>Ticket to Ride</td>
      <td>I think I'm gonna be sad, I think it's today, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beatles</td>
      <td>A Hard Days Night</td>
      <td>It's been a hard day's night, and I been worki...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beatles</td>
      <td>Cant Buy Me Love</td>
      <td>Can't buy me love, love Can't buy me love I'll...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beatles</td>
      <td>Eleanor Rigby</td>
      <td>Ah look at all the lonely people Ah look at al...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beatles</td>
      <td>I Want to Hold Your Hand</td>
      <td>Oh yeah, I'll tell you something I think you'l...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Beatles</td>
      <td>She Loves You</td>
      <td>She loves you, yeah, yeah, yeah She loves you,...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Beatles</td>
      <td>Yesterday</td>
      <td>Yesterday all my troubles seemed so far away. ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Metallica</td>
      <td>Nothing Else Matters</td>
      <td>So close no matter how far Couldn't be much mo...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Metallica</td>
      <td>Enter Sandman</td>
      <td>Say your prayers, little one Don't forget, my ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Metallica</td>
      <td>Master of Puppets</td>
      <td>End of passion play, crumbling away I’m your s...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Metallica</td>
      <td>The Unforgiven</td>
      <td>New blood joins this earth, And quickly he's s...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Metallica</td>
      <td>Fade to Black</td>
      <td>Life, it seems, will fade away Drifting furthe...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Metallica</td>
      <td>One</td>
      <td>I can’t remember anything Can’t tell if this i...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Metallica</td>
      <td>For Whom the Bell Tolls</td>
      <td>Make his fight on the hill in the early day Co...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Eminem</td>
      <td>The Real Slim Shady</td>
      <td>May I have your attention please? May I have y...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Eminem</td>
      <td>Till I Collapse</td>
      <td>'Cause sometimes you just feel tired, Feel wea...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Eminem</td>
      <td>Lose Yourself</td>
      <td>Look, if you had, one shot, or one opportunity...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Eminem</td>
      <td>Stan</td>
      <td>My tea's gone cold I'm wondering why I got out...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Eminem</td>
      <td>My Name Is</td>
      <td>Hi! My name is... (what?) My name is... (who?)...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Eminem</td>
      <td>Like Toy Soldiers</td>
      <td>Step by step, heart to heart, left right left ...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Eminem</td>
      <td>When I'm Gone</td>
      <td>Yeah... It's my life... My own words I guess.....</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Eminem</td>
      <td>Mockingbird</td>
      <td>Yeah I know sometimes things may not always ma...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Eminem</td>
      <td>Without Me</td>
      <td>Obie Trice/Real Name No Gimmicks [2x] two trai...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Bob Dylan</td>
      <td>Blowin in the Wind</td>
      <td>How many roads must a man walk down Before you...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Bob Dylan</td>
      <td>Mr Tambourin Man</td>
      <td>Hey ! Mr Tambourine Man, play a song for me I'...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Bob Dylan</td>
      <td>Its All Over Now Baby Blue</td>
      <td>You must leave now, take what you need, you th...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Bob Dylan</td>
      <td>The Times They are A-changin</td>
      <td>Come gather 'round people Wherever you roam An...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Bob Dylan</td>
      <td>Hurricane</td>
      <td>Pistols shots ring out in the barroom night En...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Bob Dylan</td>
      <td>It aint me babe</td>
      <td>Go 'way from my window Leave at your own chose...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Bob Dylan</td>
      <td>Maggies Farm</td>
      <td>I ain't gonna work on Maggie's farm no more No...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Bob Dylan</td>
      <td>A Hard Rains A-gonna Fall</td>
      <td>Oh, where have you been, my blue-eyed son? And...</td>
    </tr>
  </tbody>
</table>
</div>




```python
vect = TfidfVectorizer(stop_words='english',max_df=0.7)
dtm = vect.fit_transform(df['lyrics'])
```


```python
repr(dtm)
```




    "<32x1984 sparse matrix of type '<type 'numpy.float64'>'\n\twith 3471 stored elements in Compressed Sparse Row format>"



* we can't plot 1984 dimensions in an effective way
* need to reduce dimensionality to 2 dimensions 
* use Principle Component Analysis (PCA)
* describes data using smaller number of dimensions
* trys to retain variance and 'structure' of the data


```python
# Principle Component Analysis (PCA) to reduce down to two dimensions
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(dtm.toarray())
```


```python
df['A'] = X_pca[:,0]
df['B'] = X_pca[:,1]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>lyrics</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Beatles</td>
      <td>Help!</td>
      <td>(When) When I was younger (When I was young) s...</td>
      <td>-0.204841</td>
      <td>0.018931</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Beatles</td>
      <td>Ticket to Ride</td>
      <td>I think I'm gonna be sad, I think it's today, ...</td>
      <td>0.107004</td>
      <td>-0.258968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Beatles</td>
      <td>A Hard Days Night</td>
      <td>It's been a hard day's night, and I been worki...</td>
      <td>0.044426</td>
      <td>-0.247208</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Beatles</td>
      <td>Cant Buy Me Love</td>
      <td>Can't buy me love, love Can't buy me love I'll...</td>
      <td>0.085107</td>
      <td>-0.354033</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Beatles</td>
      <td>Eleanor Rigby</td>
      <td>Ah look at all the lonely people Ah look at al...</td>
      <td>-0.232241</td>
      <td>0.069967</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.lmplot(x='A', y='B', data=df,fit_reg=False, hue='artist')
plt.show()
```


![png](/images/Python_Text_Analytics_files/Python_Text_Analytics_54_0.png)


If we had used CountVectorizer instead of TfidfVectorizer:
![png](/images/Python_Text_Analytics_files/countvect.png)

<h3>N-gram Model</h3>

* n-gram is a sequence of n words
* bag of words model is actually a specific case of the N-gram model where n=1
* Consider the string `'Buffalo Data Science Meetup'`
 * n=1 (unigram) :  `'Buffalo'`,`'Data'`,`'Science'`,`'Meetup'`  (Bag of words model)
 * n=2 (bigram) : `'Buffalo Data'`, `'Data Science'`, `'Science Meetup'`
 * n=3 (trigram) : `'Buffalo Data Science'`,`'Data Science Meetup`'
* using n-gram model info about order of tokens


```python
messages = ["Hey hey hey lets go get lunch today :)",
           "Hey!!! I need a favor"]
```


```python
# look at bigrams
vect = CountVectorizer(ngram_range=(2,2))
dtm = vect.fit_transform(messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>get lunch</th>
      <th>go get</th>
      <th>hey hey</th>
      <th>hey lets</th>
      <th>hey need</th>
      <th>lets go</th>
      <th>lunch today</th>
      <th>need favor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# look at trigrams
vect = CountVectorizer(ngram_range=(3,3))
dtm = vect.fit_transform(messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>get lunch today</th>
      <th>go get lunch</th>
      <th>hey hey hey</th>
      <th>hey hey lets</th>
      <th>hey lets go</th>
      <th>hey need favor</th>
      <th>lets go get</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# looking at unigrams, bigrams, and trigrams
vect = CountVectorizer(ngram_range=(1,3))
dtm = vect.fit_transform(messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>favor</th>
      <th>get</th>
      <th>get lunch</th>
      <th>get lunch today</th>
      <th>go</th>
      <th>go get</th>
      <th>go get lunch</th>
      <th>hey</th>
      <th>hey hey</th>
      <th>hey hey hey</th>
      <th>...</th>
      <th>hey need</th>
      <th>hey need favor</th>
      <th>lets</th>
      <th>lets go</th>
      <th>lets go get</th>
      <th>lunch</th>
      <th>lunch today</th>
      <th>need</th>
      <th>need favor</th>
      <th>today</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 23 columns</p>
</div>




```python
# Can also use with tf-idf
vect = TfidfVectorizer(ngram_range=(2,2))
dtm = vect.fit_transform(messages)
pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>get lunch</th>
      <th>go get</th>
      <th>hey hey</th>
      <th>hey lets</th>
      <th>hey need</th>
      <th>lets go</th>
      <th>lunch today</th>
      <th>need favor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.707107</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.707107</td>
    </tr>
  </tbody>
</table>
</div>



Questions?
