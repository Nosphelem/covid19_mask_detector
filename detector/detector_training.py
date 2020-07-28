# LIBRARIES IMPORT

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# CATEGORIES AND DIRECTORY PARAMETERS

categories = ['mask_off', "mask_on"]
directory = r"./datasets/"


# INITIALIZATION : BATCH SIZE, INITIAL LEARNING RATE, EPOCHS NUMBER

batch_size = 32
initial_lr = 1e-4
epochs_number = 20

# LOADING THE LIST OF DATA USED FOR TRAINING

print("PROCESS - Loading datasets ...")

data = []
labels = []

for category in categories :
    path = os.path.join(directory, category)
    for image in os.listdir(path) :
        image_path = os.path.join(path, image)
        images = load_img(image_path, target_size = (224, 224))
        images = img_to_array(images)
        images = preprocess_input(images)
        data.append(images)
        labels.append(category)


# ENCODING ON LABELS

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype = "float32")
labels = np.array(labels)


# TRAIN_TEST_SPLIT PARAMETERS

(train_x, test_x, train_y, test_y) = train_test_split(
    data, 
    labels, 
    test_size = 0.20,
    stratify = labels,
    random_state = 19
)


# CREATING TRAINING IMAGE GENERATOR FOR DATA AUGMENTATION

augment = ImageDataGenerator(
    rotation_range = 25,
    zoom_range = 0.20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)


# LOADING MOBILENETV2 - ENSURING FC LAYER SETS ARE LEFT OFF

base_model = MobileNetV2(
    weights = "imagenet",
    include_top = False,
    input_tensor = Input(shape = (224, 224, 3))
    )


# CONSTRUCT HEAD MODEL PLACED TOP BASE MODEL

head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7, 7))(head_model)
head_model = Flatten(name = "flatten")(head_model)
head_model = Dense(128, activation = "relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation = "softmax")(head_model)


# PLACE HEAD MODEL TOP OF BASE MODEL

model = Model(inputs = base_model.input, outputs = head_model)


# LOOP OVER AND FREEZE BASE MODEL LAYERS

for layer in base_model.layers : 
    layer.trainable = False


# COMPILE MODEL

print("PROCESS - Compiling model ...")

optimizer = Adam(lr = initial_lr, decay = initial_lr / epochs_number)
model.compile(loss = "binary_crossentropy", optimizer = optimizer)


# TRAINING HEAD OF MODEL 
print("PROCESS - Training head ...")

T = model.fit(
    augment.flow(train_x, train_y, batch_size = batch_size),
    steps_per_epoch = len(train_x) // batch_size,
    validation_data = (test_x, test_y),
    validation_steps = len(test_x) // batch_size,
    epochs = epochs_number
)


# PREDICTIONS ON TESTING SET

print("PROCESS - Evaluating network ...")

predIdxs = model.predict(test_x, batch_size = batch_size)


# FIND LABEL INDEX CORRESPONDING LARGEST PREDICTED PROBABILITY FOR EACH IMAGE

predIdxs = np.argmax(predIdxs, axis = 1)


# PRINT REPORT

print(classification_report(test_y.argmax(axis = 1), predIdxs, target_names = label_binarizer.classes_))


# SAVING MODEL

print("PROCESS - Saving model ...")

model.save("mask_detector.model", save_format = "h5")


# PLOT TRAINING LOSS & ACCURACY

n = epochs_number

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, n), T.history["loss"], label = "train_loss")
plt.plot(np.arange(0, n), T.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, n), T.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, n), T.history["val_accuracy"], label = "val_acc")

plt.title("Training loss & accuracy")
plt.xlabel("Epoch number")
plt.ylabel("Loss/accuracy")
plt.legend(loc = "lower right")

plt.savefig("plot.png")


############# 