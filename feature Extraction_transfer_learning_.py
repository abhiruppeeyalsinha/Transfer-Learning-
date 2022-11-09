import tensorflow as tf
import keras,os,cv2
# from keras.utils.image_
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.applications.efficientnet import EfficientNet
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator,image_dataset_from_directory
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



train_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\train"
test_path = r"G:\video tutorial\video tutorial\A.I\cat and dog\test"

# conv_base = VGG16(weights="imagenet", include_top=True,input_shape=(224,224,3))
conv_base = VGG16(weights="imagenet", include_top=False,input_shape=(224,224,3))
# print(conv_base.summary())


#feature Extraction-  

conv_base.trainable = False
 
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
# model.add(Dense(125,activation="relu"))
# model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
print(model.summary())

train_DataGenerator_datatSet = ImageDataGenerator( rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_DataGenerator_dataset = ImageDataGenerator(rescale=1./255)


# train_ds = image_dataset_from_directory(train_path,
# labels='inferred',
#     label_mode='int',
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(224, 224),)

# test_ds = image_dataset_from_directory(test_path,
# labels='inferred',
#     label_mode='int',
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(224, 224),)    

train_set = train_DataGenerator_datatSet.flow_from_directory(train_path,target_size=(224,224),
color_mode = "rgb",batch_size=32,shuffle=True,class_mode="binary")

test_set = test_DataGenerator_dataset.flow_from_directory(test_path,target_size=(224,224),
color_mode="rgb", batch_size=32,shuffle=True,class_mode="binary")


# def process(image, label):
#         img = tf.cast(image/255, tf.float32)   
#         return image,label

# train_ds = train_ds.map(process)
# test_ds = test_ds.map(process)


# print(train_ds)


# model.compile(optimizer="adam",loss="binary_crossentropy",
# metrics=["acc"])
# save_model = model.fit(train_ds,epochs=10,validation_data=test_ds)
# train_model = save_model.model.save(r"E:\CNN Project\TL\save model\Cat&Dog_re_trainModel_10.h5")

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["acc"]) 

save_model = model.fit_generator(train_set,
epochs=10,validation_data=test_set)


plt.plot(save_model.history["accuracy"],color="red",label="train")
plt.plot(save_model.history["val_accuracy"],color="blue",label="validation")

plt.plot(save_model.history["loss"],color="red",label="train")
plt.plot(save_model.history["val_loss"],color="blue",label="validation")
plt.legend()
plt.show()

save_model.model.save(r"E:\CNN Project\TL\save model\TL_feature_extraction_10.h5")



