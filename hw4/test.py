import sys
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

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

maxlen = 40

test_list = read_data(sys.argv[1], train=False)

with open('tokenizer/tokenizer_mark.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_seq = tokenizer.texts_to_sequences(test_list)
test = pad_sequences(test_seq, maxlen=maxlen) 

model = load_model('model/model-029-0.83725.h5')

prediction = model.predict_classes(test,batch_size=1024,verbose=1)

# save prediction
of = open(sys.argv[2],'w')
out = 'id,label\n'

for i in range(len(prediction)):
    out = out + str(i) + ',' + str(prediction[i]).replace('[','').replace(']','') + '\n'

of.write(out)
of.close