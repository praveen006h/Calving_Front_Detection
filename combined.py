from simple_multi_unet_model import multi_unet_model
from simple_unet_model import simple_unet_model   
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

SIZE_X = 256 
SIZE_Y = 256
n_classes=4 


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights('test.hdf5')

img_name = "sq.png"

test_img = cv2.imread("output/"+img_name)
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


cv2.imwrite("output/zones_pred.png", predicted_img)

with Image.open("output/zones_pred.png") as img_rgb:
    img = img_rgb.convert("L")
    img_array = img.load()
    w,h = img.size
    for y in range(h):
        for x in range(w):
            pix_value = img_array[x, y]
            if pix_value == 0:
                img_array[x, y] = 0
            elif pix_value == 1:
                img_array[x, y] = 64
            elif pix_value == 2:
                img_array[x, y] = 127
            elif pix_value == 3:
                img_array[x, y] = 255
    img.save("output/zones_pred.png")



def get_model():
    return simple_unet_model(256, 256, 1)

model = get_model()

model = get_model()
model.load_weights('zones_to_fronts.hdf5')

test_img_zone = cv2.imread("output/zones_pred.png")
test_img_zone_norm=test_img_zone[:,:,0][:,:,None]
test_img_zone_input=np.expand_dims(test_img_zone_norm, 0)
prediction_front = (model.predict(test_img_zone_input)[0,:,:,0] > 0.2).astype(np.uint8)
cv2.imwrite("output/front.png", prediction_front)

img_gray = Image.open("output/"+img_name)
front = Image.open("output/front.png")
img = img_gray.convert('RGB')
sar_array = img.load()
front_array = front.load()
for y in range(256):
    for x in range(256):
        if front_array[x,y] == 1:
            sar_array[x,y] = (255,0,0)
            sar_array[x+1,y] = (255,0,0)
            sar_array[x,y+1] = (255,0,0)
            sar_array[x+1,y+1] = (255,0,0)
img.save("output/sar_front.png")

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Input Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Zone prediction on input image')
plt.imshow(predicted_img, cmap='gray')
plt.subplot(233)
plt.title('Calving Front prediction on input image')
plt.imshow(prediction_front,cmap='gray')
plt.subplot(234)
plt.title("Calving Front Visulalized on a SAR Image")
plt.imshow(img)


plt.show()
