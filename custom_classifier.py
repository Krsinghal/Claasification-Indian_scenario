import warnings
warnings.filterwarnings("ignore")
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras.preprocessing import image


classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(16, (3, 3),bias_regularizer=l2(0.01),kernel_regularizer=l2(0.01),input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 32,bias_regularizer=l2(0.001),kernel_regularizer=l2(0.001), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 16,bias_regularizer=l2(0.01),kernel_regularizer=l2(0.01), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 6, activation = 'softmax'))

classifier.summary()

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/Dataset/train',
                                                 target_size = (128, 128), batch_size = 40,class_mode='categorical')
test_set = test_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/Dataset/test', 
                                            target_size = (128, 128),batch_size = 40,class_mode='categorical')
r=classifier.fit_generator(training_set,validation_data = test_set,epochs = 20,
                         steps_per_epoch = 80,validation_steps = 100)



#plotting accuracy and loss
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='train loss')
plt.plot(r.history['val_loss'],label='val loss')
plt.legend()
plt.show()
#plt.savefig('/content/drive/My Drive/Colab Notebooks/LossVal_lossC1.jpg')

plt.plot(r.history['accuracy'],label='train acc')
plt.plot(r.history['val_accuracy'],label='val acc')
plt.legend()
plt.show()
#plt.savefig('/content/drive/My Drive/Colab Notebooks/LossVal_accsC1.jpg')

# predicting
import cv2
from PIL import Image


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((224, 224))
    return np.array(image)
    # cv2.imshow()


def get_animal_name(label):
    if label == 0:
        return "cat"
    if label == 1:
        return "cow"
    if label == 2:
        return "dog"
    if label == 3:
        return "goat"
    if label == 4:
        return "horse"
    if label == 5:
        return "monkey"


def predict_animal(file):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a, verbose=1)
    print(score)
    label_index = np.argmax(score)
    print(label_index)
    acc = np.max(score)
    animal = get_animal_name(label_index)
    print(animal)
    print("The predicted Animal is a " + animal + " with accuracy =    " + str(acc))


# importing test image from dataset, can be performed for all classes of animals.
img = image.load_img('/content/drive/My Drive/Colab Notebooks/Dataset/test/dog/dog- (1909).jpg',
                     target_size=(299, 299))
predict_animal("/content/drive/My Drive/Colab Notebooks/Dataset/test/dog/dog- (1909).jpg")

# displaying image along with predicted class of animal
plt.figure()
imgplot = plt.imshow(img)

