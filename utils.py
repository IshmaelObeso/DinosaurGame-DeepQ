import torch
from PIL import Image
import cv2 #opencv
import numpy as np
from io import BytesIO
import base64
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def screenshot(browser):
    image_b64 = browser.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen) #process image as required
    return image

def process_img(image):

    all_obstacles_idx = image > 50
    unharmful_obstacles_idx = image > 85
    image[all_obstacles_idx] = 255
    image[unharmful_obstacles_idx] = 0


    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # rgb to grey scale
    image = image[:300, :500] # crop region of interest(ROI)
    image = cv2.resize(image, (80,80))
    image = np.reshape(image, (80,80,1))
    return image

def image_to_tensor(image):
    image = np.transpose(image, (2,0,1)) #84x84x1 to 1x84x84
    image_tensor = image.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available(): #put on GPU if CUDA is avaiable
        image_tensor = image_tensor.cuda()
    return image_tensor

def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = 'Dino Agent'
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resize(screen, (800,400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"