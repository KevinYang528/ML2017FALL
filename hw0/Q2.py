import sys
from PIL import Image

im = Image.open(sys.argv[1])
pixel_im = im.load()

im2 = Image.new(im.mode, im.size)
pixel_im2 = im2.load()

for i in range(im2.size[0]):
    for j in range(im2.size[1]):
        
        [r, g, b] = pixel_im[i, j]
        
        r2 = int(r - r / 2)
        g2 = int(g - g / 2)
        b2 = int(b - b / 2)
        
        pixel_im2[i, j] = (r2, g2, b2)
        
im.close()

im2.save('Q2.png')
im2.close()