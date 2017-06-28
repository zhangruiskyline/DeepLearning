<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Recommender Systems](#recommender-systems)
- [Collaborative Filtering](#collaborative-filtering)
  - [Process](#process)
  - [Challenge](#challenge)
  - [Item based Collaborative Filtering](#item-based-collaborative-filtering)
  - [Content-based vs Collaborative Filtering (CF)](#content-based-vs-collaborative-filtering-cf)
  - [SVD/Matrix Factorization](#svdmatrix-factorization)
    - [How to calculate SVD/MF](#how-to-calculate-svdmf)
- [Ranking](#ranking)
- [Embedding](#embedding)
  - [Theory of Embedding](#theory-of-embedding)
  - [Code for embedding](#code-for-embedding)
    - [embedding in keras example](#embedding-in-keras-example)
- [Distance and similarity](#distance-and-similarity)
- [Explicit vs Implicit Feedback](#explicit-vs-implicit-feedback)
  - [Explicit Feedback](#explicit-feedback)
    - [The usual architecture for explicit feedback:](#the-usual-architecture-for-explicit-feedback)
    - [The Deep NN model](#the-deep-nn-model)
    - [The embedding illustration](#the-embedding-illustration)
    - [Embedding similarity](#embedding-similarity)
    - [Using item metadata in the model](#using-item-metadata-in-the-model)
  - [Implicit Feedback:](#implicit-feedback)
    - [The usual architecture for Triplet Loss:](#the-usual-architecture-for-triplet-loss)
    - [The Deep NN model](#the-deep-nn-model-1)
- [General Framework](#general-framework)
- [Ethical Considerations](#ethical-considerations)
- [Applications](#applications)
  - [Spotify Music Recommendation](#spotify-music-recommendation)
  - [Deep Neural Networks for YouTube Recommendation](#deep-neural-networks-for-youtube-recommendation)
  - [Quora: Semantic Question Matching with Deep Learning](#quora-semantic-question-matching-with-deep-learning)
  - [Quora Recommendation Model: Ranking answers](#quora-recommendation-model-ranking-answers)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Recommender Systems

> Recommender Systems

> Embeddings

> Architecture and Regularization

# Collaborative Filtering
* Typically, the workflow of a collaborative filtering system is:

A user expresses his or her preferences by rating items (e.g. books, movies or CDs) of the system. These ratings can be viewed as an approximate representation of the user's interest in the corresponding domain.

The system matches this user's ratings against other users' and finds the people with most "similar" tastes.

With similar users, the system recommends items that the similar users have rated highly but not yet being rated by this user (presumably the absence of rating is often considered as the unfamiliarity of an item)

## Process

  * Look for users who share the same rating patterns with the active user (the user whom the prediction is for).
  * Use the ratings from those like-minded users found in step 1 to calculate a prediction for the active user

![Collaborative Filtering](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/cf.png)

and the weight matrix can be

![Collaborative Filtering similarity](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/cf2.png)

## Challenge

* Cold start
* Popular bias
* Sparsity:
* Scalability: computation grow as number of users and number of items growing
* Pool relationship among minded but sparse user

> Next? Use latent to capture user and item in reduced dimension

## Item based Collaborative Filtering

* Looks items targeted user rated
* Compute how similar they are to target item

![Item CF](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/item_cf.png)

## Content-based vs Collaborative Filtering (CF)

* Content-based: user metadata (gender, age, location...) and item metadata (year, genre, director, actors)
* Collaborative Filtering: passed user/item interactions: stars, plays, likes, clicks
* Hybrid systems: CF + metadata to mitigate the cold-start problem

## SVD/Matrix Factorization
matrix factorization can be used to discover latent features underlying the interactions between two different kinds of entities. (Of course, you can consider more than two kinds of entities and you will be dealing with tensor factorization, which would be more complicated.) And one obvious application is to predict ratings in collaborative filtering.

![Matrix Factorization](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/mf.png)

### How to calculate SVD/MF
iterate/ gradient decent

http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

# Ranking


# Embedding

Can be used for Symbolic variable like: item ids, user ids in Recommender Systems. or Any categorical descriptor: tags, movie genres, visited URLs, skills on a resume, product categories

## Theory of Embedding

One Hot encoding:__[0,0,1,...,0]∈{0,1}|V|__

* Sparse, discrete, large dimension |V|
* Each axis has a meaning
* Symbols are equidistant from each other

Translate one hot encoding to Embedding:

* Continuous and dense
* Can represent a huge vocabulary in low dimension, typically: d∈{16,32,...,4096}
* Axis have no meaning a priori
* Embedding metric can capture semantic distance
* Can be seen as embedding(x)=onehot(x).W (one-hot encoding multiplied by a weight matrix W∈ℝn×d)

## Code for embedding

```python
import numpy as np

embedding_size = 4
vocab_size = 10

embedding = np.arange(embedding_size * vocab_size, dtype='float')
embedding = embedding.reshape(vocab_size, embedding_size)
print(embedding)

##Out:
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]
 [ 16.  17.  18.  19.]
 [ 20.  21.  22.  23.]
 [ 24.  25.  26.  27.]
 [ 28.  29.  30.  31.]
 [ 32.  33.  34.  35.]
 [ 36.  37.  38.  39.]]

# compute a one-hot encoding of i
# simply index (slice) the embedding matrix by  i, using numpy indexing
i = 3
onehot = np.zeros(10)
onehot[i] = 1.
onehot:
#out: array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.])

embedding_vector = np.dot(onehot, embedding)
print(embedding_vector)
# [ 12.  13.  14.  15.]
```

### embedding in keras example

In Keras, embeddings have an extra parameter, input_length which is typically used when having a sequence of symbols as input (think sequence of words). In our case, the length will always be 1.

```python
from keras.layers import Embedding

embedding_layer = Embedding(output_dim=embedding_size, input_dim=vocab_size,input_length=1,name='my_embedding')
from keras.layers import Input
from keras.models import Model

x = Input(shape=[1], name='input')
embedding = embedding_layer(x)
# Keras embedding model:
# keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
model = Model(input=x, output=embedding)
model.output_shape
# Out: (None, 1, 4)
model.get_weights() #The embedding weights are randomly initialized:
# Out
[array([[-0.01936201,  0.01061306, -0.00502486,  0.02421001],
        [-0.0071108 , -0.00054221, -0.00888319, -0.01043938],
        [ 0.02423747, -0.03029217,  0.04300025,  0.04906172],
        [ 0.02151183, -0.03897278, -0.02845473,  0.03577714],
        [ 0.02908627, -0.00221761,  0.04954894, -0.0006169 ],
        [-0.04881933, -0.00207155,  0.01976109, -0.03441812],
        [-0.04364808, -0.03180511,  0.00863915,  0.0453595 ],
        [ 0.02288247, -0.00607735, -0.04348791, -0.03095592],
        [-0.03011045,  0.02573893, -0.01666873, -0.0387949 ],
        [-0.04212544,  0.00182465, -0.00680971,  0.04644271]], dtype=float32)]

model.predict([[0], [3]])
#Out:
array([[[-0.01936201,  0.01061306, -0.00502486,  0.02421001]],

       [[ 0.02151183, -0.03897278, -0.02845473,  0.03577714]]], dtype=float32)

# The output of an embedding layer is then a 3-d tensor of shape (batch_size, sequence_length, embedding_size) To remove the sequence dimension, useless in our case, we use the Flatten() layer

from keras.layers import Flatten

x = Input(shape=[1], name='input')

# Add a flatten layer to remove useless "sequence" dimension
y = Flatten()(embedding_layer(x))

model2 = Model(input=x, output=y)
model2.output_shape
# Out: (None, 4)
model2.predict([[0],[3]])
#Out
array([[-0.01936201,  0.01061306, -0.00502486,  0.02421001],
       [ 0.02151183, -0.03897278, -0.02845473,  0.03577714]], dtype=float32)
```

# Distance and similarity

> Euclidean distance d(x,y)=||x−y||2

Simple with good properties. Dependent on norm (embeddings usually unconstrained)

> Cosine similarity: cosine(x,y)=x⋅y/||x||⋅||y||

Angle between points, regardless of norm. cosine(x,y)∈(−1,1). Expected cosine similarity of random pairs of vectors is 0.

if both have unit norms:
||x−y||2=2⋅(1−cosine(x,y))

or

cosine(x,y)=1−||x−y||2/2

Alternatively, dot product (unnormalized) is used in practice as a pseudo similarity


# Explicit vs Implicit Feedback

> Explicit: positive and negative feedback

* Examples: review stars and votes.
* Regression metrics: Mean Square Error, Mean Absolute Error...

> Implicit: positive feedback only

* Examples: page views, plays, comments...
* Ranking metrics: ROC AUC, precision at rank, NDCG...

> Comparison

* Implicit feedback much more abundant than explicit feedback
* Explicit feedback does not always reflect actual user behaviors
* Implicit feedback can be negative: like Page view with very short dwell time ,Click on "next" button

## Explicit Feedback


### The usual architecture for explicit feedback:
![Explicit Feedback architecture](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rec_archi_1.png)

The Code to implement
```python
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')
embedding_size = 32
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)
# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)
## Simply use dot product for user embedding and item embedding
y = merge([user_vecs, item_vecs], mode=dot_mode, output_shape=(1,))
model = Model(input=[user_id_input, item_id_input], output=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train])
squared_differences = np.square(initial_train_preds - rating_train.values)
absolute_differences = np.abs(initial_train_preds - rating_train.values)

## monitor run via Training the model
history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, nb_epoch=6, validation_split=0.1,
                    shuffle=True, verbose=2)

# Apply to test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))                    
```

### The Deep NN model

![Deep NN model for explicit feedback](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rec_archi_2.png)


```python
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 50
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = merge([user_vecs, item_vecs], mode='concat')

input_vecs = Dropout(0.2)(input_vecs)

x = Dense(64, activation='relu')(input_vecs)
## output dimension for 1-d rating
## tanh activation squashes the outputs between -1 and 1
## when we want to predict values between 1 and 5
y = Dense(1)(x)
model = Model(input=[user_id_input, item_id_input], output=y)
## A binary crossentropy loss is only useful for binary
## classification, while we are in regression (use mse or mae)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train])

history = model.fit([user_id_train, item_id_train], rating_train,
                    batch_size=64, nb_epoch=5, validation_split=0.1,
                    shuffle=True, verbose=2)

test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))
```

* The NN structure looks like following

```python
# weights and shape
weights = model.get_weights()
[w.shape for w in weights]

#Output:
[(944, 50), (1683, 50), # input size for user and item
(100, 64), # concat embedding from both user and item as input layer and output is 64 in Dense
(64,), (64, 1), (1,)]
```

### The embedding illustration

* This is trained embedding:

```python
user_embeddings = weights[0]
item_embeddings = weights[1]
print("First item name from metadata:", items["name"][1])
print("Embedding vector for the first item:")
print(item_embeddings[1])
print("shape:", item_embeddings[1].shape)

#Output:
First item name from metadata: Toy Story (1995)
Embedding vector for the first item:
[-0.09698112  0.08855848  0.0487658  -0.05941751 -0.00156418  0.12242204
  0.08545233  0.0240779  -0.05879579 -0.06360796  0.09601892 -0.05764137
  0.08753403 -0.06028035  0.08934461 -0.05766458 -0.07108436  0.06516211
  0.00763744  0.03659184  0.07006893  0.04200531  0.04824194 -0.05603793
 -0.05869192  0.04269543 -0.06628406  0.07491404 -0.03653983 -0.03171537
 -0.05951712  0.12271345 -0.054116    0.09354898 -0.0655614  -0.09088151
  0.10239661 -0.08552022 -0.03371907  0.10928621 -0.04839545 -0.0637596
 -0.00298041  0.07896615  0.10997456 -0.08374112  0.04906891  0.06700497
  0.06132839 -0.0625702 ]
shape: (50,)
```

### Embedding similarity

```python
# %load solutions/similarity.py
EPSILON = 1e-07

def cosine(x, y):
    dot_pdt = np.dot(x, y.T)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    return dot_pdt / (norms + EPSILON)

# Computes cosine similarities between x and all item embeddings
def cosine_similarities(x):
    dot_pdts = np.dot(item_embeddings, x)
    norms = np.linalg.norm(x) * np.linalg.norm(item_embeddings, axis=1)
    return dot_pdts / (norms + EPSILON)

# Computes euclidean distances between x and all item embeddings
def euclidean_distances(x):
    return np.linalg.norm(item_embeddings - x, axis=1)

# Computes top_n most similar items to an idx,
def most_similar(idx, top_n=10, mode='euclidean'):
    sorted_indexes=0
    if mode=='euclidean':
        dists = euclidean_distances(item_embeddings[idx])
        sorted_indexes = np.argsort(dists)
        idxs = sorted_indexes[0:top_n]
        return list(zip(items["name"][idxs], dists[idxs]))
    else:
        sims = cosine_similarities(item_embeddings[idx])
        # [::-1] makes it possible to reverse the order of a numpy
        # array, this is required because most similar items have
        # a larger cosine similarity value
        sorted_indexes = np.argsort(sims)[::-1]
        idxs = sorted_indexes[0:top_n]
        return list(zip(items["name"][idxs], sims[idxs]))

# sanity checks:
print("cosine of item 1 and item 1: "
      + str(cosine(item_embeddings[1], item_embeddings[1])))
euc_dists = euclidean_distances(item_embeddings[1])
print(euc_dists.shape)
print(euc_dists[1:5])

# Test on movie 181: Return of the Jedi
print("Items closest to 'Return of the Jedi':")
for title, dist in most_similar(181, mode="euclidean"):
    print(title, dist)

plt.hist(euc_dists)
```
> * Conclusion

We observe that the embedding is poor at representing similarities between movies, as most distance/similarities are very small/big. One may notice a few clusters though it's interesting to plot the distributions.

The reason for that is that the number of ratings is low and the embedding does not automatically capture semantic relationships in that context. Better representations arise with higher number of ratings, and less overfitting models

### Using item metadata in the model

Using a similar framework as previously, we will build another deep model that can also leverage additional metadata. The resulting system is therefore an Hybrid Recommender System that does both Collaborative Filtering and Content-based recommendations.

```python
# transform the date (string) into an int representing the release year
parsed_dates = [int(film_date[-4:])
                for film_date in items["date"].tolist()]

items['parsed_date'] = pd.Series(parsed_dates, index=items.index)
max_date = max(items['parsed_date'])
min_date = min(items['parsed_date'])

from sklearn.preprocessing import scale

items['scaled_date'] = scale(items['parsed_date'].astype('float64'))
item_meta_train = items["scaled_date"][item_id_train]
item_meta_test = items["scaled_date"][item_id_test]

len(item_meta_train), len(item_meta_test)

```

Then the model becomes:

```python
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')
meta_input = Input(shape=[1], name='meta_item')

embedding_size = 50
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)


# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = merge([user_vecs, item_vecs, meta_input], mode='concat')

x = Dense(64, activation='relu')(input_vecs)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
y = Dense(1)(x)

model = Model(input=[user_id_input, item_id_input, meta_input], output=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train, item_meta_train])

```

> __A recommendation function for a given user_id_train__

Once the model is trained, the system can be used to recommend a few items for a user, that he/she hasn't already seen:

* we use the model.predict to compute the ratings a user would have given to all items
* we build a recommendation function that sorts these items and exclude those the user has already seen
* This is prediction the rate as regression problem(similarity)

```python
def recommend(user_id, top_n=10):
    item_ids = range(1, max_item_id)
    seen_movies = list(all_ratings[all_ratings["user_id"]==user_id]["item_id"])
    item_ids = list(filter(lambda x: x not in seen_movies, item_ids))

    print("user "+str(user_id) +" has seen "+str(len(seen_movies)) + " movies. "+
          "Computing ratings for "+str(len(item_ids))+ " other movies")

    item_ids = np.array(item_ids)
    user = np.zeros_like(item_ids)
    user[:]=user_id
    items_meta = items["scaled_date"][item_ids].values

    rating_preds = model.predict([user, item_ids, items_meta])

    item_ids = np.argsort(rating_preds[:,0])[::-1].tolist()
    rec_items = item_ids[:top_n]
    return [(items["name"][movie], rating_preds[movie][0]) for movie in rec_items]


##Call recommend
recommend(3)
```

> __Predicting ratings as a classification problem__

we can help the model by forcing it to predict those values by treating the problem as a multiclassification problem. The only required changes are:
* setting the final layer to output class membership probabities using a softmax activation with 5 outputs;
* optimize the categorical cross-entropy classification loss instead of a regression loss such as MSE or MAE.


```Python
# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 50
dense_size = 128
dropout_embedding = 0.5
dropout_hidden = 0.2

user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = merge([user_vecs, item_vecs], mode='concat')
input_vecs = Dropout(dropout_embedding)(input_vecs)

x = Dense(dense_size, activation='relu')(input_vecs)
x = Dropout(dropout_hidden)(x)
x = Dense(dense_size, activation='relu')(x)
y = Dense(output_dim=5, activation='softmax')(x)

model = Model(input=[user_id_input, item_id_input], output=y)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

initial_train_preds = model.predict([user_id_train, item_id_train]).argmax(axis=1) + 1
print("Random init MSE: %0.3f" % mean_squared_error(initial_train_preds, rating_train))
print("Random init MAE: %0.3f" % mean_absolute_error(initial_train_preds, rating_train))


history = model.fit([user_id_train, item_id_train], rating_train - 1,
                    batch_size=64, nb_epoch=6, validation_split=0.1,
                    shuffle=True, verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('loss');

test_preds = model.predict([user_id_test, item_id_test]).argmax(axis=1) + 1
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))
```

And predict using classification way
```python
def recommend_classify(user_id, top_n=10):
    item_ids = range(1, max_item_id)
    seen_movies = list(all_ratings[all_ratings["user_id"]==user_id]["item_id"])
    item_ids = list(filter(lambda x: x not in seen_movies, item_ids))

    print("user "+str(user_id) +" has seen "+str(len(seen_movies)) + " movies. "+
          "Computing ratings for "+str(len(item_ids))+ " other movies")

    item_ids = np.array(item_ids)
    user = np.zeros_like(item_ids)
    user[:]=user_id
    items_meta = items["scaled_date"][item_ids].values

    rating_preds = model.predict([user, item_ids]).argmax(axis=1) + 1
    print(rating_preds)
    item_ids = np.argsort(rating_preds[:])[::-1].tolist()
    print(item_ids)
    rec_items = item_ids[:top_n]
    return [(items["name"][movie], rating_preds[movie]) for movie in rec_items]

recommend_classify(3)
```

* Comparison: Regression and classification

regression looks more accurate since there is only 5 classes and not fine grained


## Implicit Feedback:

* The similarity score between a user and an item is defined by the unormalized dot products of their respective embeddings.

* The matching scores can be use to rank items to recommend to a specific user.

Training of the model parameters is achieved by randomly sampling negative items not seen by a pre-selected anchor user. We want the model

  * embedding matrices to be such that the __similarity between the user vector and the negative vector is smaller than the similarity between the user vector and the positive item vector__.
  * Furthermore we use a __margin__ to further move appart the negative from the anchor user. Train goal is to max the margin


```shell
Gather a set of positive pairs user i and item j
While model has not converged:
  Shuffle the set of pairs (i,j)
  For each (i,j):
    Sample item k a negative item for user i uniformly at random
    Train model on triplet (i,j,k)
```

* To assess their quality we do the following for each user:

  * compute matching scores for items (except the movies that the user has already seen in the training set),
  * compare to the positive feedback actually collected on the test set using the ROC AUC ranking metric,
  * average ROC AUC scores across users to get the average performance of the recommender model on the test set.

```python
def average_roc_auc(match_model, data_train, data_test):
    """Compute the ROC AUC for each user and average over users"""
    max_user_id = max(data_train['user_id'].max(), data_test['user_id'].max())
    max_item_id = max(data_train['item_id'].max(), data_test['item_id'].max())
    user_auc_scores = []
    for user_id in range(1, max_user_id + 1):
        pos_item_train = data_train[data_train['user_id'] == user_id]
        pos_item_test = data_test[data_test['user_id'] == user_id]

        # Consider all the items already seen in the training set
        all_item_ids = np.arange(1, max_item_id + 1)
        items_to_rank = np.setdiff1d(all_item_ids, pos_item_train['item_id'].values)

        # Ground truth: return 1 for each item positively present in the test set
        # and 0 otherwise.
        expected = np.in1d(items_to_rank, pos_item_test['item_id'].values)

        if np.sum(expected) >= 1:
            # At least one positive test value to rank
            repeated_user_id = np.empty_like(items_to_rank)
            repeated_user_id.fill(user_id)

            predicted = match_model.predict([repeated_user_id, items_to_rank],
                                            batch_size=4096)
            user_auc_scores.append(roc_auc_score(expected, predicted))

    return sum(user_auc_scores) / len(user_auc_scores)
```


Here is the [code](https://github.com/maciejkula/triplet_recommendations_keras) for Triplet loss

### The usual architecture for Triplet Loss:

Here is the architecture of such a triplet architecture. The triplet name comes from the fact that the loss to optimize is defined for triple (anchor_user, positive_item, negative_item):

![Triplet Loss](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rec_archi_implicit.png)

We call this model a triplet model with bi-linear interactions because the similarity between a user and an item is captured by a dot product of the first level embedding vectors. This is therefore not a deep architecture.

```python
import tensorflow as tf

def identity_loss(y_true, y_pred):
    """Ignore y_true and return the mean of y_pred

    This is a hack to work-around the design of the Keras API that is
    not really suited to train networks with a triplet loss by default.
    """
    return tf.reduce_mean(y_pred + 0 * y_true)
def margin_comparator_loss(inputs, margin=1.):
    """Comparator loss for a pair of precomputed similarities

    If the inputs are cosine similarities, they each have range in
    (-1, 1), therefore their difference have range in (-2, 2). Using
    a margin of 1. can therefore make sense.

    If the input similarities are not normalized, it can be beneficial
    to use larger values for the margin of the comparator loss.
    """
    positive_pair_sim, negative_pair_sim = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + margin, 0)

```

### The Deep NN model

![Deep NN model for triplet loss](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/rec_archi_implicit_2.png)

The code to build the loss function and model implements a deep_match_model, deep_triplet_model pair of models for the architecture described in the schema.

The last layer of the embedded __Multi Layer Perceptron__ outputs a single scalar that encodes the similarity between a user and a candidate item.

Evaluate the resulting model by computing the per-user average ROC AUC score on the test feedback data.
AUC ROC score is close to 0.50 for a randomly initialized model.
can reach at least 0.91 ROC AUC with this deep model.

```python
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Dense, Dropout, merge
from keras.regularizers import l2


def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,
                         dropout=0, l2_reg=None):
    """Build the shared multi layer perceptron"""
    mlp = Sequential()
    if n_hidden == 0:
        # Plug the output unit directly: this is a simple
        # linear regression model. Not dropout required.
        mlp.add(Dense(1, input_dim=input_dim,activation='relu', W_regularizer=l2_reg))
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim,activation='relu', W_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        for i in range(n_hidden - 1):
            mlp.add(Dense(hidden_size, activation='relu',W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
        mlp.add(Dense(1, activation='relu', W_regularizer=l2_reg))
    return mlp


def build_models(n_users, n_items, user_dim=32, item_dim=64,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    """Build models to train a deep triplet network"""
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')

    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1,name='user_embedding', W_regularizer=l2_reg)

    # The following embedding parameters will be shared to encode both
    # the positive and negative items.
    item_layer = Embedding(n_items, item_dim, input_length=1,name="item_embedding", W_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))


    # Similarity computation between embeddings using a MLP similarity
    positive_embeddings_pair = merge([user_embedding, positive_item_embedding],mode='concat',
            name="positive_embeddings_pair")
    positive_embeddings_pair = Dropout(dropout)(positive_embeddings_pair)
    negative_embeddings_pair = merge([user_embedding, negative_item_embedding],mode='concat',
                                     name="negative_embeddings_pair")
    negative_embeddings_pair = Dropout(dropout)(negative_embeddings_pair)

    # Instanciate the shared similarity architecture
    interaction_layers = make_interaction_mlp(user_dim + item_dim, n_hidden=n_hidden, hidden_size=hidden_size,
        dropout=dropout, l2_reg=l2_reg)

    positive_similarity = interaction_layers(positive_embeddings_pair)
    negative_similarity = interaction_layers(negative_embeddings_pair)

    # The triplet network model, only used for training
    triplet_loss = merge([positive_similarity,
    negative_similarity],mode=margin_comparator_loss,output_shape=(1,),name='comparator_loss')

    deep_triplet_model = Model(input=[user_input, positive_item_input,negative_item_input],
                               output=triplet_loss)

    # The match-score model, only used at inference
    deep_match_model = Model(input=[user_input, positive_item_input],
                             output=positive_similarity)

    return deep_match_model, deep_triplet_model


hyper_parameters = dict(
    user_dim=32,
    item_dim=64,
    n_hidden=2,
    hidden_size=128,
    dropout=0.1,
    l2_reg=0
)
deep_match_model, deep_triplet_model = build_models(n_users, n_items,**hyper_parameters)


deep_triplet_model.compile(loss=identity_loss, optimizer='adam')
fake_y = np.ones_like(pos_data_train['user_id'])

n_epochs = 15

for i in range(n_epochs):
    # Sample new negatives to build different triplets at each epoch
    triplet_inputs = sample_triplets(pos_data_train, max_item_id,random_seed=i)

    # Fit the model incrementally by doing a single pass over the
    # sampled triplets.
    deep_triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                           batch_size=64, nb_epoch=1, verbose=2)

    # Monitor the convergence of the model
    test_auc = average_roc_auc(deep_match_model, pos_data_train, pos_data_test)
    print("Epoch %d/%d: test ROC AUC: %0.4f"
          % (i + 1, n_epochs, test_auc))

```


# General Framework

In general, you can formulate any deterministic machine learning algorithm in a neural network framework. Matrix factorisation is an embedding model that embeds both user and item in a shared latent space and predicts the rating as an inner product of the embedding. See [Github Code Implementations](https://github.com/mesuvash/TFMF/blob/master/TFMF.ipynb)


Further, for more general NN framework refer "AutoRec: Autoencoders Meet Collaborative Filtering" (http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf) and [Code](https://github.com/mesuvash/NNRec). In this framework, you can define different models based on input and target i.e

* Input = User vector in item space; Target = User vector in item space: This is equivalent to U-AutoRec model.
* Input = Item vector in user space; Target = Item vector in user space, This is equivalent to I-AutoRec model.
* Input = One Hot encoding of User; Target = User vector in item space, This is equivalent to MF model.
* Input = User features ; Target = User vector in item space, This is equivalent to user feature based recommender.
* Input = Item features ; Target = Item vector in user space, This is equivalent to item feature based recommender.

# Ethical Considerations

* Amplification of existing discriminatory and unfair behaviors / bias

Example: gender bias in ad clicks (fashion / jobs)
Using the firstname as a predictive feature

* Amplification of the filter bubble and opinion polarization

People tend to unfollow people they don't agree with
Ranking / filtering systems can further amplify this issue
Optimizing for short-term clicks can promote clickbait contents

# Applications

## Spotify Music Recommendation

 [Spotify Music recommendation](http://benanne.github.io/2014/08/05/spotify-cnns.html)

## Deep Neural Networks for YouTube Recommendation

[Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html)

## Quora: Semantic Question Matching with Deep Learning

https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning

* Problem definition: To detect similar # QUESTION: More formally, the duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function:


```Math
f(q1, q2) → 0 or 1
```

* Current model

Our current production model for solving this problem is a random forest model with tens of handcrafted features, including the cosine similarity of the average of the word2vec embeddings of tokens, the number of common words, the number of common topics labeled on the questions, and the part-of-speech tags of the words.

* Improvement

We trained our own word embeddings using Quora's text corpus using LSTM, combined them to generate question embeddings for the two questions, and then fed those question embeddings into a representation layer. We then concatenated the two vector representation outputs from the representation layers and fed the concatenated vector into a dense layer to produce the final classification result.

![quora embeddings](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_1.png)

*  Tai, Socher empirically-motivated handcrafted features

  * the distance, calculated as the sum of squares of the difference of the two vectors, and * the “angle”, calculated as an element-wise multiplication of the two vector representations (denoted ʘ).

A neural network using the distance and angle as the two input neurons was then stacked on top

![quora embeddings 2](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_2.png)

* Attention Model

Finally, we tried an attention-based approach from Google Research [4] that combined neural network attention with token alignment, commonly used in machine translation. The most prominent advantage of this approach, relative to other attention-based approaches, was the small number of parameters. Similar to the previous two approaches, this model represents each token from the question with a word embedding. The process, shown in Figure 3, trained an attention model (soft alignment) for all pairs of words in the two questions, compared aligned phrases, and finally aggregated comparisons to get classification result.

![quora attention](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_attention.png)

## Quora Recommendation Model: Ranking answers

https://www.slideshare.net/xamat/recsys-2016-tutorial-lessons-learned-from-building-reallife-recommender-systems

https://engineering.quora.com/A-Machine-Learning-Approach-to-Ranking-Answers-on-Quora

* Ground truth data: upvotes/downvotes

some problem:
1. time sensitive
2. rich get richer
3. joke answer
4. good answer from not so active user

* Goal:

In ranking problems our goal is to predict a list of documents ranked by relevance. In most cases there is also additional context, e.g. an associated user that's viewing results, therefore introducing personalization to the problem.

* System architecture
![quora_ranking](https://github.com/zhangruiskyline/DeepLearning_Intro/blob/master/img/quora_ranking.png)

* Ground Truth Build

1. A/B Test online observation
2. offline: user survey. manually create ranking, starting from some already known good one

* Good answer standard

1. Answers the question that was asked.
2. Provides knowledge that is reusable by anyone interested in the question.
3. Answers that are supported with rationale.
4. Demonstrates credibility and is factually correct.
5. Is clear and easy to read.

* Features

  * text based features:
  * expertise-based features
  * upvote/downvote history

General rule is make sure our features generalize well. Text features can be particularly problematic in this regard.

Ensemble always works better

* Real production

  * rank the new answer on the question page as soon as possible to provide a smooth user experience

1. simple model with easy to calculate feature as approximate, once answer is added, recompute the more accurate score asynchronously

2. Question pages can contain hundreds of answers.

  1. cache answer scores so that the question page can load in a reasonable time
  2. cache all features
  3. All this data (answer scores and feature values) is stored in HBase, which is an open-source NoSQL datastore able to handle a large volume of data and writes.
  4.  cache score sometimes is not good when answer or feature changes: Consider a user who has tens of thousands of answers on Quora. If we depend on a feature like the number of answers added by an answer author, then every time this user adds an answer, we have to update the score of all of their answers at once.stopped updating feature values if that wouldn't impact the answer score at all.
