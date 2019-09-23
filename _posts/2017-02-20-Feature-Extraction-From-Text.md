---
title:  "Feature Extraction from Text"
date:   2017-02-20 15:04:23
categories: [machine-learning, nlp]
tags: [machine-learning, nlp]
header:
  image: /images/bg.JPG
---

This posts serves as an simple introduction to feature extraction from text to be used for a machine learning model using Python and sci-kit learn. I'm assuming the reader has some experience with sci-kit learn and creating ML models, though it's not entirely necessary. Most machine learning algorithms can't take in straight text, so we will create a matrix of numerical values to represent our text. We'll go over the differences between two common ways of doing this: CountVectorizer and TfidfVectorizer.

There are a few terms we'll define right off the bat. 
* document - refers to a single piece of text information. This could be a text message, tweet, email, book, lyrics to a song. This is equivalent to one row or observation.
* corpus - a collection of documents. This would be equivalent to a whole data set of rows/observations.
* token - this is a word, phrase, or symbols derived from a document through the process of tokenization. This will happen behind the scenes so we won't need to worry too much about it and for our purposes it essentially means a word. For example the document `'How are you'` would have tokens of `'How'`, `'are'`, and `'you'`

Let's start by defining a corpus of a few different sample text messages.


```python
messages = ["Hey hey hey lets go get lunch today :)",
           "Did you go home?",
           "Hey!!! I need a favor"]
```

<h2>CountVectorizer</h2>

First, we'll use CountVectorizer() from ski-kit learn to create a matrix of numbers to represent our messages. CountVectorizer() takes what's called the Bag of Words approach. Each message is seperated into tokens and the number of times each token occurs in a message is counted.

We'll import CountVectorizer from sklearn and instantiate it as an object, similar to how you would with a classifier from sklearn. In fact the usage is very similar. Instead of using fit() and then predict() we will use fit() then transform().


```python
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
```

Using the fit method, our CountVectorizer() will "learn" what tokens are being used in our messages.


```python
vect.fit(messages)
```




    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)



By using the get_feature_names() method, we can see what features have been created from our messages. (or what tokens have been "learned" by CountVectorizer)


```python
vect.get_feature_names()
```




    [u'did',
     u'favor',
     u'get',
     u'go',
     u'hey',
     u'home',
     u'lets',
     u'lunch',
     u'need',
     u'today',
     u'you']



There's a few things to note here. 
* Everything is lowercase
* Words less than two letters have not been included (notice there is no `'a'`)
* Punctuation has been removed
* There are no duplicates

By changing from the default arguments when CountVectorizer is instantiated, you can change what was mentioned in the first two bullet points if wanted.

Next, lets transform our CountVectorizer object. This will create matrix populated with token counts to represent our messages. This is often referred to as a [document term matrix](https://en.wikipedia.org/wiki/Document-term_matrix). We'll name the output from this as `dtm` to reflect this. We'll also print out some information about the matrix as well as the matrix itself.


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
    

Since we've been throwing around the word matrix this whole time you might be a bit confused by the output. From the repr() command we can see it's stored in compressed sparse row format(aka a sparse matrix). Each of our messages only contain 3-6 unique tokens and we have 11 different features created from all of our messages. This means each row will mostly be filled with zeros. In order to save space/computational power a sparse matrix is created. This means that only the location and value of non-zero values is saved. We'll convert it to pandas dataframe(a dense matrix version) for better intuition.


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



For example, the 3rd entry in our sparse matrix is:

```
(0,4)    3
```

Which corresponds to 1st message and the 5th feature, `'hey'` (remember zero indexing). The entry is 3 because our 1st message had the word `'hey'` three times in it.

Now you're ready to feed you document term matrix into your ML classifier or whatever else you had planned. You do not need to convert it into a pandas dataframe before use. Sci-kit learn will accept the sparse matrix representation or the pandas dataframe. Though it's advisable to keep it in sparse form especially when working with a large corpus.

Just to give an example, a [Kaggle competition](https://www.kaggle.com/c/whats-cooking) I did had a corpus of different recipes. Each recipe only contained about 10 ingredients each. But since there were several thousand recipes with some unique ingredients the resulting number of features in my document term matrix was over 6000. So each row representing a recipe was 99% filled with zeros.


There is one thing I'd like to make a note of. Let's say you got another message soon after you created your document term matrix and want to add it in. We'll transform it to a document term matrix using our CountVectorizer() object we fit earlier.


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



Now even though it contained 6 unique tokens (excluding `'a'`) there is only 4 entries in our DTM. The tokens `'drink'` and `'tonight'` are not represented. This is because our original messages used to fit CountVectorizer() did not have these tokens. We can append our new message to our original collection and then refit and transform to make sure we don't lose this information. This time we'll use the fit_transform method combining fit and transform just to show an alternative.


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



For this simple example refitting and transforming seems to be the correct thing to do. But when you're creating a model to predict something you'll have two sets of data, a training set and a testing set. You'll create a CountVectorizer() object and fit it to the training set. Then you'll create a DTM for both data sets, each transformed using the same fit. It's likely that the testing set contains tokens not included in the training set. Therefore the DTM for the testing set doesn't have features for those tokens that don't overlap between the two data sets. It may seem like an issue at first but it's actually nothing to be concerned about. You wouldn't want to create a new fit for the testing set as it would create new features(and maybe lose some) that the model wasn't trained on. If the training set DTM had columns for features included in the testing set but not in itself, the whole column would be filled with zeros anyway and offer no predictive insight. This may be a bit confusing but below I have some psuedo code on how this would be implemented for a logistic regression model that might make it more clear.

```python
# creating DTMs
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# creating and training logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
y_predicted = logreg.predict(X_test_dtm)  # predicting
```

<h2>TfidfVectorizer</h2>

An alternative to CountVectorizer is something called TfidfVectorizer. It also creates a document term matrix from our messages. However, instead of filling the DTM with token counts it calculates term frequency-inverse document frequency value for each word(TF-IDF). The TF-IDF is the product of two weights, the term frequency and the inverse document frequency(who would've guessed?). 

To generalize:    `TF-IDF = term frequency * (1 / document frequency)`

Or:    `TF-IDF = term frequency *  inverse document frequency`

Term frequency is a weight representing how often a word occurs in a document. If we have several occurences of the same word in one document we can expect the TF-IDF to increase. 

Inverse document frequency is another weight representing how common a word is across documents. If a word is used in many documents then the TF-IDF will decrease.

There are many ways to calculate the TF-IDF, but all essentially calculate the same concept. [If you're interested the wikipedia page goes over some of the ways it's calculated](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

With the definition out of the way we'll go through a few examples to see how it works. Since the usage is pretty much identical to CountVectorizer and we'll be going through a few examples we'll make a function to create a DTM from our messages to make things a bit easier and clearer.


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



As you can see the word `'hey'` has the lowest value as it's the only word that occurs in both messages(documents). The words `'favor'` and `'need'` have the highest values because they only occur in the second message and there are only 3 unique words in the second message so they have a higher term frequency.

Now let's change our messages a bit. We'll change the first message from `'Hey lets get lunch :)'` to `'Hey hey hey lets get lunch :)'`. We should expect the term frequency for `'hey'` to increase and therefore the TF-IDF value for hey in the first message to increase.


```python
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



The value for `'hey'` in the first message went up just as expected. There are two things worth noticing here. First, the values for other words in the first message have decreased. Their term frequency has decreased as there are now more words in the message so the TF-IDF will decrease as well. Also, the value for `'hey'` in the second message is unchanged from our first example. This is because we haven't done anything to  change the IDF portion of the TF-IDF. Both examples contain `'hey'` in both messages.


So next, lets try manipulate the messages to change the IDF portion of the TF-IDF. We'll change our second message from `'Hey!!! I need a favor'` to `I need a favor'`. Now the word `'hey'` only occurs in one message so we should expect its value to increase as its IDF value is increasing.


```python
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



Sure enough the TF-IDF value for `'hey'` in the first message has increased. It now has the highest TF-IDF has it has the highest DF and all words shared the same IDF now.

<h2>Picking a Vectorizer and Arguments</h2>

You may be wondering now which to use. Like most things in creating a model the best way to figure out which is best is to just try both. Both are simple to implement and will likely have cases where one may outperform the other. Maybe even some combination of both may work.

When instantiating your vectorizer there are a few arguments to include that I've found can help. 
* max_features = n : only considers the top n words orderd by term frequency
* min_df = n : ignores words with a document frequency below n
* max_df = n : ignores words with a document frequency above n
* stop_words = [' '] : ignores common words like `'the'`, `'that'`, `'which'`, etc. You'll need to define in a list what words you want to include. There are lists of common stop words available online, the NLTK library also has a list of stop words built into it.

<h2>Further Reading</h2>

* [Wikipedia: Bag of Words model](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [Wikipedia: TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [Wikipedia: Stop words](https://en.wikipedia.org/wiki/Stop_words)
* [Sci-kit learn docs: text feature extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [Sci-kit learn docs: CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [Sci-kit learn docs: TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [SO When does TF-IDF reduce accuracy?](http://stackoverflow.com/questions/39152229/in-general-when-does-tf-idf-reduce-accuracy)
* [SO Combining TfidfVectorizer and CountVectorizer?](http://stackoverflow.com/questions/27496014/does-it-make-sense-to-use-both-countvectorizer-and-tfidfvectorizer-as-feature-ve)
* [Kevin Markham's tutorial on Machine Learning with Text](https://github.com/justmarkham/pycon-2016-tutorial)
    * I highly recommend everything this guy does. Much of what I learned about this came from him.
