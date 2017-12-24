import pandas as pd
import numpy as np
import sys
from keras.models import load_model


test = pd.read_csv(sys.argv[1]);
test_x = test.feature.str.split(' ').tolist()
test_x = np.array(test_x).astype('float32')
test_x = test_x / 255

test_x = test_x.reshape(test_x.shape[0],48,48,1)

models = ['model_0.67846.h5',
          'model_0.67679.h5',
          'model_0.67093.h5', 
          'model_0.66620.h5',
          'model_0.66453.h5']

pred = 0.0
for model_name in models:
    model = load_model(model_name)
    pred = model.predict(test_x) + pred

pred_en = np.argmax(pred, axis=-1)

of = open(sys.argv[2],'w')
output = 'id,label\n'

for i in range(len(pred_en)):
    output = output + str(i) + ',' + str(pred_en[i]) + '\n'

of.write(output)
of.close