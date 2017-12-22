import os
import sys
import argparse
import numpy as np
import pandas as pd
from models import build_mf_model, build_dnn_model, rate

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):

    test_data = pd.read_csv(args[1], usecols=['UserID', 'MovieID'])
    print('{} testing data loaded.'.format(test_data.shape[0]))

    trained_model = build_mf_model(max_userid, max_movieid, latent_dim)
    trained_model.summary()

    trained_model.load_weights(model_path)

    recommendations = pd.read_csv(args[1])

    # y_std = 1.116897661146206
    # y_mean = 3.5817120860388076

    rating_pred = trained_model.predict([test_data['UserID'], test_data['MovieID']])

    rating_pred = np.clip(rating_pred, 1.0, 5.0)

    with open(str(args[2]).rstrip(),'w') as f:
        f.write('TestDataID,Rating\n')  
        i = 0
        while i < rating_pred.shape[0]:
            f.write(str(i + 1) + ',' + str(rating_pred[i, 0]) + '\n')   
            i += 1
    print('output.csv done')


if __name__ == '__main__':
   
    model_path = './model/model-029-0.86059.h5'

    latent_dim = 64
    max_userid = 6040
    max_movieid = 3952

    main(sys.argv)