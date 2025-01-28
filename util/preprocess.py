import cv2

def load_image(path, size=(256, 256)):
    image = cv2.imread(path)
    image = cv2.resize(image, size)
    return image
