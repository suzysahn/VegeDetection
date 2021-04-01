import os
import cv2
import numpy as np

import tensorflow as tf

##############################################
use_label_file = False  # set this to true if you want load the label names from a file; uses the label_file defined below; the file should contain the names of the used labels, each label on a separate line
label_file = 'labels.txt'
base_dir = '../..'  # relative path to the Fruit-Images-Dataset folder
train_dir = os.path.join(base_dir, 'Training')
test_dir = os.path.join(base_dir, 'Test')
saved_files = 'output_files'  # root folder in which to save the the output files; the files will be under output_files/model_name
##############################################

if not os.path.exists(saved_files):
    os.makedirs(saved_files)

if use_label_file:
    with open(label_file, "r") as f:
        labels = [x.strip() for x in f.readlines()]
else:
    labels = os.listdir(train_dir)
num_classes = len(labels)


# Create a custom layer that converts the original image from
# RGB to HSV and grayscale and concatenates the results
# forming in input of size 100 x 100 x 4
def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez

def scan_image():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            #print(check) #prints true as long as the webcam is running
            # print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            imgpath = "/home/pi/shared/VegeDetection/src/image_classification/imgCapture.jpg"
            key = cv2.waitKey(1)
            if key == ord('s'): 

                cv2.imwrite(imgpath, img=frame)
                
                webcam.release()
                cv2.waitKey(1650)
                cv2.destroyAllWindows()

                img_ = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR)
                img_ = cv2.resize(img_,(100,100))
                img_resized = cv2.imwrite("/home/pi/shared/VegeDetection/src/image_classification/imgCap.jpg", img=img_)
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break


def test_model(name=""):
    imgpath = test_dir + '/Potato White/99_100.jpg'
    model_out_dir = os.path.join(saved_files, name)
    if not os.path.exists(model_out_dir):
        print("No saved model found")
        exit(0)
    model = tf.keras.models.load_model(model_out_dir + "/model.h5")
    #imgfile = scan_image()
    print("os path exists? ")
    print(os.path.exists(imgpath))

    # image = cv2.imread(imgpath, cv2.IMREAD_ANYCOLOR)
    # #print(image.shape())
    # image = cv2.resize(image, (100, 100))
    # cv2.imshow("image",image)
    # cv2.waitKey(1300)
    # data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
    # image_array = np.asarray(image)
    # data[0] = image_array
    # y_pred = model.predict(data, 1)
    # print("Prediction probabilities: " + str(y_pred))
    # print("Predicted class index: " + str(y_pred.argmax(axis=-1)))
    # print("Predicted class label: " + labels[y_pred.argmax(axis=-1)[0]])

    image1 = cv2.imread(test_dir + '/Potato White/99_100.jpg')
    image1 = cv2.resize(image1, (100, 100))
    data = np.ndarray(shape=(1, 100, 100, 3), dtype=np.int)
    image_array1 = np.asarray(image1)
    data[0] = image_array1
    y_pred = model.predict(data, 1)
    print("Prediction probabilities: " + str(y_pred))
    print("Predicted class index: " + str(y_pred.argmax(axis=-1)))
    print("Predicted class label: " + labels[y_pred.argmax(axis=-1)[0]])

#scan_image()
test_model(name='fruit-360 model')

