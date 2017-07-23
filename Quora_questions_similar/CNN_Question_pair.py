# coding: utf-8

# # Predicting Duplicate Questions

# In[5]:

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime, time, json
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, \
    Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import initializers
from keras import backend as K
from keras.optimizers import SGD
from collections import defaultdict

# In[6]:

train = pd.read_csv("train.csv")[:100]
test = pd.read_csv("test.csv")[:100]

# In[7]:

train.head(6)

# In[8]:

test.head()

# In[9]:

print(train.shape)
print(test.shape)

# In[10]:

# Check for any null values
print(train.isnull().sum())
print(test.isnull().sum())

# In[11]:

# Add the string 'empty' to empty strings
train = train.fillna('empty')
test = test.fillna('empty')

# In[12]:

print(train.isnull().sum())
print(test.isnull().sum())

# In[13]:

# Preview some of the pairs of questions
for i in range(6):
    print(train.question1[i])
    print(train.question2[i])
    print()

# In[14]:

stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
              'those', 'then',
              'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to',
              'What', 'Which',
              'Is', 'If', 'While', 'This']


# In[191]:

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Convert words to lower case and split them
    # text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


# In[192]:

def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list) / len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))


# In[193]:

train_question1 = []
process_questions(train_question1, train.question1, 'train_question1', train)

# In[194]:

train_question2 = []
process_questions(train_question2, train.question2, 'train_question2', train)

# In[165]:

test_question1 = []
process_questions(test_question1, test.question1, 'test_question1', test)

# In[166]:

test_question2 = []
process_questions(test_question2, test.question2, 'test_question2', test)

# In[195]:

# Preview some transformed pairs of questions
i = 0
for i in range(i, i + 10):
    print(train_question1[i])
    print(train_question2[i])
    print()

# In[168]:

# Find the length of questions
lengths = []
for question in train_question1:
    lengths.append(len(question.split()))

for question in train_question2:
    lengths.append(len(question.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

# In[169]:

lengths.counts.describe()

# In[170]:

print(np.percentile(lengths.counts, 99.0))
print(np.percentile(lengths.counts, 99.4))
print(np.percentile(lengths.counts, 99.5))
print(np.percentile(lengths.counts, 99.9))

# In[171]:

# tokenize the words for all of the questions
all_questions = train_question1 + train_question2 + test_question1 + test_question2
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_questions)
print("Fitting is complete.")
train_question1_word_sequences = tokenizer.texts_to_sequences(train_question1)
print("train_question1 is complete.")
train_question2_word_sequences = tokenizer.texts_to_sequences(train_question2)
print("train_question2 is complete")

# In[172]:

test_question1_word_sequences = tokenizer.texts_to_sequences(test_question1)
print("test_question1 is complete.")
test_question2_word_sequences = tokenizer.texts_to_sequences(test_question2)
print("test_question2 is complete.")

# In[173]:

word_index = tokenizer.word_index
print("Words in index: %d" % len(word_index))

# In[174]:

# Pad the questions so that they all have the same length.

max_question_len = 36

train_q1 = pad_sequences(train_question1_word_sequences,
                         maxlen=max_question_len)
print("train_q1 is complete.")

train_q2 = pad_sequences(train_question2_word_sequences,
                         maxlen=max_question_len)
print("train_q2 is complete.")

# In[175]:

test_q1 = pad_sequences(test_question1_word_sequences,
                        maxlen=max_question_len,
                        padding='post',
                        truncating='post')
print("test_q1 is complete.")

test_q2 = pad_sequences(test_question2_word_sequences,
                        maxlen=max_question_len,
                        padding='post',
                        truncating='post')
print("test_q2 is complete.")

# In[30]:

y_train = train.is_duplicate

# In[31]:

# Load GloVe to use pretrained vectors

# Note for Kaggle users: Uncomment this - it couldn't be used on Kaggle

# From this link: https://nlp.stanford.edu/projects/glove/
# embeddings_index = {}
# with open('glove.840B.300d.txt', encoding='utf-8') as f:
#    for line in f:
#        values = line.split(' ')
#        word = values[0]
#        embedding = np.asarray(values[1:], dtype='float32')
#        embeddings_index[word] = embedding
#
# print('Word embeddings:', len(embeddings_index)) #151,250


# In[176]:

# Need to use 300 for embedding dimensions to match GloVe's vectors.
embedding_dim = 300

# Note for Kaggle users: Uncomment this too, because it relate to the code for GloVe.

nb_words = len(word_index)
# word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
# for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        word_embedding_matrix[i] = embedding_vector
#
# print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0)) #75,334


# In[177]:

units = 128  # Number of nodes in the Dense layers
dropout = 0.25  # Percentage of nodes to drop
nb_filter = 32  # Number of filters to use in Convolution1D
filter_length = 3  # Length of filter for Convolution1D
# Initialize weights and biases for the Dense layers
weights = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
bias = bias_initializer = 'zeros'

model1 = Sequential()
model1.add(Embedding(nb_words + 1,
                     embedding_dim,
                     # weights = [word_embedding_matrix], Commented out for Kaggle
                     input_length=max_question_len,
                     trainable=False))

model1.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))

model1.add(Flatten())

model2 = Sequential()
model2.add(Embedding(nb_words + 1,
                     embedding_dim,
                     # weights = [word_embedding_matrix],
                     input_length=max_question_len,
                     trainable=False))

model2.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))

model2.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))

model2.add(Flatten())

model3 = Sequential()
model3.add(Embedding(nb_words + 1,
                     embedding_dim,
                     # weights = [word_embedding_matrix],
                     input_length=max_question_len,
                     trainable=False))
model3.add(TimeDistributed(Dense(embedding_dim)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(dropout))
model3.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,)))

model4 = Sequential()
model4.add(Embedding(nb_words + 1,
                     embedding_dim,
                     # weights = [word_embedding_matrix],
                     input_length=max_question_len,
                     trainable=False))

model4.add(TimeDistributed(Dense(embedding_dim)))
model4.add(BatchNormalization())
model4.add(Activation('relu'))
model4.add(Dropout(dropout))
model4.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,)))

modela = Sequential()
modela.add(Merge([model1, model2], mode='concat'))
modela.add(Dense(units * 2, kernel_initializer=weights, bias_initializer=bias))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))

modela.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))

modelb = Sequential()
modelb.add(Merge([model3, model4], mode='concat'))
modelb.add(Dense(units * 2, kernel_initializer=weights, bias_initializer=bias))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))

modelb.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))

model = Sequential()
model.add(Merge([modela, modelb], mode='concat'))
model.add(Dense(units * 2, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# In[178]:

# save the best weights for predicting the test question pairs
save_best_weights = 'question_pairs_weights.h5'

t0 = time.time()
callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
history = model.fit([train_q1, train_q2, train_q1, train_q2],
                    y_train,
                    batch_size=256,
                    epochs=2,  # Use 100, I reduce it for Kaggle,
                    validation_split=0.15,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

# In[179]:

# Aggregate the summary statistics
summary_stats = pd.DataFrame({'epoch': [i + 1 for i in history.epoch],
                              'train_acc': history.history['acc'],
                              'valid_acc': history.history['val_acc'],
                              'train_loss': history.history['loss'],
                              'valid_loss': history.history['val_loss']})

# In[180]:

summary_stats

# In[181]:

plt.plot(summary_stats.train_loss)  # blue
plt.plot(summary_stats.valid_loss)  # green
plt.show()

# In[182]:

# Find the minimum validation loss during the training
min_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
print('Minimum loss at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(min_loss))
min_loss = round(min_loss, 4)

# In[183]:

# Make predictions with the best weights
model.load_weights(save_best_weights)
predictions = model.predict([test_q1, test_q2, test_q1, test_q2], verbose=True)

# In[184]:

# Create submission
submission = pd.DataFrame(predictions, columns=['is_duplicate'])
submission.insert(0, 'test_id', test.test_id)
file_name = 'submission_{}.csv'.format(min_loss)
submission.to_csv(file_name, index=False)

# In[185]:

submission.head(10)