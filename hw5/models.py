import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout
from keras.layers import Reshape, Flatten
from keras.layers.merge import concatenate, dot, add
from keras.models import Model
from keras import backend as K



def build_mf_model(n_users, n_movies, latent_dim):
    u_input = Input(shape=(1,))
    m_input = Input(shape=(1,))

    u = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(u_input)
    u = Flatten()(u)
    u = Dropout(0.5)(u)
   
    m = Embedding(n_movies, latent_dim, embeddings_initializer='random_normal')(m_input)
    m = Flatten()(m)
    m = Dropout(0.5)(m)
    
    u_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(u_input)
    u_bias = Flatten()(u_bias)

    m_bias = Embedding(n_movies, 1, embeddings_initializer='zeros')(m_input)
    m_bias = Flatten()(m_bias)

    out = dot([u, m], axes=-1)
    out = add([out, u_bias, m_bias])

    model = Model(inputs=[u_input, m_input], outputs=out)
    return model


def build_dnn_model(n_users, n_movies, latent_dim):
    u_input = Input(shape=(1,))
    m_input = Input(shape=(1,))

    u = Embedding(n_users, latent_dim)(u_input)
    u = Flatten()(u)
    
    m = Embedding(n_movies, latent_dim)(m_input)
    m = Flatten()(m)

    out = concatenate([u, m])
    out = Dropout(0.1)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(0.15)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1)(out)


    model = Model(inputs=[u_input, m_input], outputs=out)
    return model

def rate(model, user_id, item_id):
    preds = model.predict([np.array([user_id]), np.array([item_id])], batch_size=256, verbose=2)[0][0]
    return np.clip(preds, 1.0, 5.0)