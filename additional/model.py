
import sys
import warnings
import cv2
import numpy as np
from keras.models import Model, load_model
import tensorflow.keras.backend as K
from PIL import Image
import glob
warnings.filterwarnings('ignore')

#seed = 56
#IMAGE_HEIGHT = IMAGE_WIDTH = 256
#NUM_CHANNELS = 3

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


class test:
    def __init__(self):
        self.weight_path = "C:/Users/gekal/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/road_vect/additional/weights/road_mapper_final.h5"

    def process(self):
        tiles = []
        names = []
        files = glob.glob('tiles/'+'*.*')
        model = load_model(self.weight_path, custom_objects={'dice_coef_loss': soft_dice_loss, 'dice_loss': iou_coef})
        if len(tiles) >= 5:
            for i in range(0,len(files)):
                image = cv2.imread(files[i])
                names.append(files[i].split('\\')[-1])
                tiles.append(image)
                if i % 5 == 0:
                    image_tiles = np.array(tiles)
                    tiles.clear()
                    predictions = model.predict(image_tiles, verbose=1)
                    for prediction in predictions:
                        im1 = Image.fromarray(np.squeeze(prediction[:,:,0])*255).convert("L")
                        im1.save("tiles_segmented/" + names.pop(0))
                if tiles:
                    image_tiles = np.array(tiles)
                    tiles.clear()
                    predictions = model.predict(image_tiles, verbose=1)
                    for prediction in predictions:
                        im1 = Image.fromarray(np.squeeze(prediction[:,:,0])*255).convert("L")
                        im1.save("tiles_segmented/" + names.pop(0))
        else:
            for i in range(0,len(files)):
                image = cv2.imread(files[i])
                names.append(files[i].split('\\')[-1])
                tiles.append(image)
                image_tiles = np.array(tiles)
                tiles.clear()
                predictions = model.predict(image_tiles, verbose=1)
                for prediction in predictions:
                    im1 = Image.fromarray(np.squeeze(prediction[:,:,0])*255).convert("L")
                    im1.save("tiles_segmented/" + names.pop(0))

obj = test()
obj.process()

if __name__ == "__main__":
    print(sys.version)
    obj = test()
    obj.process()