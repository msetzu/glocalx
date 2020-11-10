import sys
sys.path.append('/usr/local/opt/opencv/lib/python3.7/site-packages')

import cv2
import glob
import json

from ilorem import ILOREM
from util import record2str, neuclidean

from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from datamanager import *


def img4dnn(img):
    res = image.img_to_array(img)
    res = np.expand_dims(res, axis=0)
    res = inception_v3.preprocess_input(res)
    return res


def main():

    random_state = 0
    path = '/Users/riccardo/Documents/PhD/LOREM/'
    path_data = path + 'dataset/'

    imlist = list()
    for filename in sorted(glob.glob(path_data + 'ILSVRC/*')):
        img = cv2.imread(filename)
        imlist.append(img)

    class_map = json.load(open(path_data + 'imagenet_class_index.json'))
    class_map = {int(k): v for k, v in class_map.items()}
    inv_class_map = {v[0]: k for k, v in class_map.items()}
    class_values = [''] * len(class_map)
    for k, v in class_map.items():
        class_values[k] = v[1]

    K = None
    class_name = 'class'

    bb = inception_v3.InceptionV3()

    def bb_predict(imgs):
        pred = bb.predict(np.vstack([img4dnn(img) for img in imgs]))
        pred = np.array([inv_class_map[p[0][0]] for p in decode_predictions(pred)])
        return pred

    i2e = 0
    img = imlist[i2e]

    bb_outcome = bb_predict([img])[0]
    bb_outcome_str = class_values[bb_outcome]

    print('bb(img) = { %s }' % bb_outcome_str)
    print('')

    explainer = ILOREM(K, bb_predict, class_name, class_values, neigh_type='lime', use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, segmentation_type='quickshift', random_state=random_state,
                       verbose=True)

    exp = explainer.explain_instance(img, num_samples=20, use_weights=True, metric='cosine', hide_color=0)

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    # for crule in exp.crules:
    #     print(crule)
    # print(exp.bb_pred, exp.dt_pred, exp.fidelity)


if __name__ == "__main__":
    main()
