import unittest
import os
import cv2
import numpy as np
import unittest
import tensorflow as tf

class TestStringMethods(unittest.TestCase):
    def setUp(self):   
        self.base_dir = '../..'  
        self.train_dir = os.path.join(self.base_dir, 'Training')
        self.test_dir = os.path.join(self.base_dir, 'Test')
        self.saved_files = 'output_files'  

        if not os.path.exists(self.saved_files):
            os.makedirs(self.saved_files)
        self.labels = os.listdir(self.train_dir)
        self.num_classes = len(self.labels)
        self.model_out_dir = os.path.join(self.saved_files, 'vegeModel')
        if not os.path.exists(self.model_out_dir):
            print("No saved model found")
            exit(0)
        self.model = tf.keras.models.load_model(self.model_out_dir + "/model.h5")
        self.imgpath = ''

    def test_modelExist(self):
        self.assertTrue(os.path.exists(self.model_out_dir), 'No saved model found')

    def test_outputDirExist(self):
        self.assertTrue(os.path.exists(self.saved_files), 'No output directory found')

    def test_tomato1(self):
        self.imgpath = self.test_dir + '/TestImg/T.jpg'
        image1 = cv2.imread(self.imgpath)
        image1 = cv2.resize(image1, (100, 100))
        data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
        image_array1 = np.asarray(image1)
        data[0] = image_array1
        y_pred = self.model.predict(data, 1)
        self.assertEqual('Tomato 1', self.labels[y_pred.argmax(axis=-1)[0]], 'Did not accurately predict vegetable.')


    def test_onion(self):
        self.imgpath = self.test_dir + '/TestImg/O.jpg'
        image1 = cv2.imread(self.imgpath)
        image1 = cv2.resize(image1, (100, 100))
        data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
        image_array1 = np.asarray(image1)
        data[0] = image_array1
        y_pred = self.model.predict(data, 1)
        self.assertEqual('Onion White', self.labels[y_pred.argmax(axis=-1)[0]], 'Did not accurately predict vegetable.')


    def test_cucumber(self):
        self.imgpath = self.test_dir + '/TestImg/C.jpg'
        image1 = cv2.imread(self.imgpath)
        image1 = cv2.resize(image1, (100, 100))
        data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
        image_array1 = np.asarray(image1)
        data[0] = image_array1
        y_pred = self.model.predict(data, 1)
        self.assertEqual('Cucumber Ripe', self.labels[y_pred.argmax(axis=-1)[0]], 'Did not accurately predict vegetable.')

if __name__ == '__main__':
    unittest.main()