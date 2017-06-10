from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import pickle
import re
import csv
import json
import glob
from misc.utils import get_image
import scipy.misc
import misc.skipthoughts as skipthoughts


#########################################
# Input
#  Data/paintings
#   - imgs
#   - paintings.csv
#########################################


LR_HR_RATIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
INPUT_DATA_DIR="Data/paintings"
NUM_TRAIN=1000

numbers = re.compile(r'(\d+)')
def numerical_sort(value):
    parts = numbers.split(value)
    print(parts)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_filenames():
    files = glob.glob(os.path.join(INPUT_DATA_DIR, 'imgs/*.jpg'))

    return sorted(files, key=numerical_sort)

def save_data_list(outpath, filenames):
    hr_images = []
    lr_images = []
    lr_size = int(LOAD_SIZE / LR_HR_RATIO)
    cnt = 0
    for f_name in filenames:
        img = get_image(f_name, LOAD_SIZE, is_crop=False)
        img = img.astype('uint8')
        hr_images.append(img)
        lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic')
        lr_images.append(lr_img)
        cnt += 1
        if cnt % 50 == 0:
            print('Load %d........' % cnt)

    print('images', len(hr_images), len(lr_images), hr_images[0].shape, lr_images[0].shape)

    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)

    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)

    outfile = outpath + 'filenames.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump(filenames, f_out)
        print('save to: ', outfile)

    # TODO Use fake classes for now, switch to real data when it's available
    outfile = outpath + 'class_info.pickle'
    with open(outfile, 'wb') as f_out:
        pickle.dump([0 for i in range(len(filenames))], f_out)
        print('save to: ', outfile)



def convert_images_to_pickle():
    filenames = get_filenames()

    train_dir = os.path.join(INPUT_DATA_DIR, "train/")
    test_dir = os.path.join(INPUT_DATA_DIR, "test/")

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    save_data_list(train_dir, filenames[:NUM_TRAIN])
    save_data_list(test_dir, filenames[NUM_TRAIN:])

class Encoder():
    def encode(self, a):
        return 0

def save_embeddings_csv(outpath, encoder):
    embeddings = []
    captions_file = os.path.join(INPUT_DATA_DIR, "paintings.csv")
    with open(captions_file) as f:
        data = csv.reader(f)
        i = 1
        for row in data:
            assert str(i) == row[0], \
                   "image id not match, expect %s got %s" % (str(i), row[0])
            caption = row[1]
            print("[From CSV] Encode %s" % caption)
            vector = encoder.encode(np.array([caption]))
            embeddings.append(vector)
            i += 1

    outfile = outpath + "/skip-thought-embeddings.pickle"
    with open(outfile, 'wb') as f_out:
        pickle.dump(embeddings, f_out)
        print('save to: ', outfile)

def save_embeddings_json(outpath, encoder):
    embeddings = []
    captions_file = os.path.join(INPUT_DATA_DIR, "paintings.json")
    with open(captions_file) as f:
        data = json.load(f)

        i = 1
        for row in data:
            assert str(i) == row['image_id'], \
                   "image id not match, expect %s got %s" % (str(i), row['image_id'])
            caption = row['caption']
            print("[From JSON] Encode %s" % caption)
            vector = encoder.encode(np.array([caption]))
            embeddings.append(vector)
            i += 1

    outfile = outpath + "/skip-thought-embeddings.pickle"
    with open(outfile, 'wb') as f_out:
        pickle.dump(embeddings, f_out)
        print('save to: ', outfile)

def convert_captions_to_pickle():
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    #encoder = Encoder()
    save_embeddings_csv(INPUT_DATA_DIR, encoder)

def gen_caption_files():
    captions_path = os.path.join(INPUT_DATA_DIR, "text_c10")
    if not os.path.exists(captions_path):
        os.mkdir(captions_path)

    captions_file = os.path.join(INPUT_DATA_DIR, "corrected_vis.csv")
    with open(captions_file) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            with open(os.path.join(captions_path, "img%s.jpg.txt" % row[0]), 'w') as cf:
                cf.write(row[1])

if __name__ == '__main__':
    #print(get_filenames())
    convert_images_to_pickle()
    convert_captions_to_pickle()
    gen_caption_files()
