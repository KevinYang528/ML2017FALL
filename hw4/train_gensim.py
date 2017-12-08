import sys
import numpy as np
import pickle
import random
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding,Dense,Flatten,LSTM,Dropout,Bidirectional
from keras.models import Sequential,Model
from keras.activations import sigmoid
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from gensim.models import Word2Vec

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def read_data(filename, train):
    texts, labels = [], []
    if train:
        with open(filename, encoding='UTF-8') as f:
            for line in f:
                label, symbol, text = line.strip('\n').split(' ', 2)
                texts.append(text)
                labels.append(label)
        return texts, labels
    else:
        with open(filename, encoding='UTF-8') as f:
            f.readline()
            for line in f:
                id, text = line.strip('\n').split(',', 1)
                texts.append(text)
        return texts  

def read_data_nolabel(filename):
    texts = []
    with open(filename, encoding='UTF-8') as f:
        for line in f:
            texts.append(line)
    return texts  

train_list, train_label = read_data(sys.argv[1], train=True)
nolabel_list = read_data_nolabel(sys.argv[2])
# test_list = read_data(sys.argv[3], train=False)

train_label = np.array(train_label,dtype='float32')

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(train_list + test_list + nolabel_list)


with open('tokenizer/tokenizer_mark.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

train_seq = tokenizer.texts_to_sequences(train_list)
# nolabel_seq = tokenizer.texts_to_sequences(nolabel_list)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# # semi-supervise
# model = load_model('model/model-029-0.83725.h5')
# pred_unlabeled = model.predict(nolabel,batch_size=1024,verbose=1)
# pred_class = np.round(pred_unlabeled)
# semi_list = []
# semi_label = []
# confidence = 0.95

# for i, y in enumerate(pred_unlabeled):
#     if y >= confidence:
#         semi_list.append(nolabel_list[i])
#         semi_label.append(1)
#     elif y <= 1 - confidence:
#         semi_list.append(nolabel_list[i])
#         semi_label.append(0)
# semi_seq = tokenizer.texts_to_sequences(semi_list)
# semi = pad_sequences(semi_seq, maxlen=maxlen)

# train = np.concatenate((train, semi), axis=0)
# train_label = np.concatenate((train_label, semi_label), axis=0)

maxlen = 40
train = pad_sequences(train_seq, maxlen=maxlen)
# nolabel = pad_sequences(nolabel_seq, maxlen=maxlen)
model = Word2Vec.load('w2v/word2vec_mark.pkl')
weights = model.wv.syn0

vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

# word to vector dict 
embeddings_index = {}
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    vector = vocab_list[i][1]
    embeddings_index[word] = vector

top_word = len(word_index) + 1

embeddings_matrix = np.zeros((top_word, 256))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector
        
train_x, valid_x, train_y, valid_y = train_test_split(train, train_label, test_size=0.1, random_state=42)    

model = Sequential()

embedding_layer = Embedding(top_word, 256, weights=[embeddings_matrix], input_length=maxlen, trainable=False)

model.add(embedding_layer)
model.add(Bidirectional(LSTM(256,return_sequences=True,dropout=0.5,recurrent_dropout=0.5)))
model.add(Bidirectional(LSTM(256,dropout=0.5,recurrent_dropout=0.5)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

callbacks = [] 
# callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1))
callbacks.append(ModelCheckpoint('model-{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', verbose=1, save_best_only=True))

history = model.fit(train_x, train_y, epochs=30, validation_data=(valid_x,valid_y), batch_size=1024, callbacks=callbacks)