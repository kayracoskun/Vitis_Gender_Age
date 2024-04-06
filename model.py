import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import itertools
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.utils import plot_model


"""
def custom_data_generator(directory, target_size, batch_size):
    gender_labels = {'female': 0, 'male': 1}
    age_labels = ['age_0-9', 'age_10-19', 'age_20-29', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69', 'age_70-79', 'age_80-89', 'age_90-99', 'age_100-116']
    age_label_encoder = LabelEncoder()
    age_label_encoder.fit(age_labels)
    
    while True:
        image_batch = []
        gender_label_batch = []
        age_label_batch = []
        for gender, age_group in itertools.product(gender_labels.keys(), age_labels):
            folder_path = os.path.join(directory, gender, age_group)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = load_img(img_path, target_size=target_size)
                img = img_to_array(img) / 255.0  # Normalize image

                image_batch.append(img)
                gender_label_batch.append(gender_labels[gender])
                age_label_batch.append(age_label_encoder.transform([age_group])[0])
                
                if len(image_batch) == batch_size:
                    yield (np.array(image_batch), {'gender_output': to_categorical(gender_label_batch, num_classes=len(gender_labels)), 'age_output': to_categorical(age_label_batch, num_classes=len(age_labels))})
                    image_batch, gender_label_batch, age_label_batch = [], [], []
"""


def custom_data_generator(directory, target_size, batch_size):
    gender_labels = {'female': 0, 'male': 1}
    
    while True:
        image_batch = []
        gender_label_batch = []
        for gender in gender_labels.keys():
            gender_folder_path = os.path.join(directory, gender)
            # Iterate through each sub-folder in the gender directory
            for age_group in os.listdir(gender_folder_path):
                age_group_folder_path = os.path.join(gender_folder_path, age_group)
                # Check if it's indeed a directory
                if not os.path.isdir(age_group_folder_path):
                    continue
                for img_file in os.listdir(age_group_folder_path):
                    img_path = os.path.join(age_group_folder_path, img_file)
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img) / 255.0  # Normalize image

                    image_batch.append(img)
                    gender_label_batch.append(gender_labels[gender])

                    if len(image_batch) == batch_size:
                        # Only yield gender labels for classification
                        yield (np.array(image_batch), to_categorical(gender_label_batch, num_classes=len(gender_labels)))
                        image_batch, gender_label_batch = [], []  # Reset for the next batch


# Define the input size for the model
input_shape = (224, 224, 3)

# Define the number of gender classes
num_gender_classes = 2

# Define the input layer
input_layer = Input(shape=input_shape)

# First convolutional block
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Second convolutional block
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Third convolutional block
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)

# Flatten the output
x = Flatten()(x)

# Gender classification branch
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
gender_output = Dense(num_gender_classes, activation='softmax', name='gender_output')(x)

# Define the model
model = Model(inputs=input_layer, outputs=gender_output)

"""
# Compile the model
model.compile(optimizer='adam',
              loss={'gender_output': 'categorical_crossentropy', 'age_output': 'categorical_crossentropy'},
              metrics={'gender_output': 'accuracy', 'age_output': 'mae'})
"""

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = custom_data_generator(
    directory='./data/UTKFace',
    target_size=input_shape[:2],
    batch_size=32
)

model.summary()

model.fit(train_generator, epochs=2, steps_per_epoch=200)  # Adjust steps_per_epoch based on your dataset size

plot_model(model, to_file='gender_age_model.png', show_shapes=True, show_layer_names=True)

N = 500  # Number of images you want to extract
image_count = 0
X_train = []  # Initialize an empty list to store your images

for X_batch, _ in train_generator:
    # Add the images in the current batch to the X_train list
    X_train.extend(X_batch)
    image_count += len(X_batch)
    
    # Break the loop once we have collected enough images
    if image_count >= N:
        break

X_train = X_train[:N]
X_train = np.array(X_train)

print("Shape of X_train:", X_train.shape)

# Quantize and store AI models
quantizer = vitis_quantize.VitisQuantizer(model)

# Number of data sets to be passed is between 100-1000 without labels, so let's use 500
quantized_model = quantizer.quantize_model(calib_dataset=X_train[0:500])

# Save the quantized model
quantized_model.save("quantized_gender_model.h5")
