from keras.models import load_model
import cv2
import numpy as np
import glob

model = load_model('fmodel.h5')

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

dict={0:"capsule",1:"circle",2:"football"}

for opt in ["capsule","circle","football"]:
    for name in glob.glob("./test2/"+opt+"/*"):
        img = cv2.imread(name)
        img = cv2.resize(img,(200,200))
        img = np.reshape(img,[1,200,200,3])
        classes = model.predict_classes(img)
        print(name,classes,"Shape :",dict[list(classes)[0]])
