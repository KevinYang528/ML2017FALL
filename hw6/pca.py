import numpy as np
import skimage
from skimage import io
import sys
import os

def reconstruct(filename, face_avg, U, pc):
    a = io.imread(filename)
    a = a - face_avg
    a = a.reshape(-1, 1)    
    weights = np.dot(a.T, U)

    M = np.dot(U[:,:pc], weights[:,:pc].T)
    M = M.reshape([600, 600, 3]) + face_avg
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    io.imsave('reconstruction.jpg', M)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# load images
images = load_images_from_folder(sys.argv[1])
print('load images done!')

face_avg = np.mean(images, axis=0)

X = (images - face_avg).reshape(len(images), -1)

# SVD
U, s, V = np.linalg.svd(X.T, full_matrices=False)
print('SVD done!')

# reconstruct
reconstruct(os.path.join(sys.argv[1], sys.argv[2]), face_avg, U, pc=4)