from keras.models import load_model
import os, cv2
import numpy as np

def preprocessImage(image):
    modelInput = image / 255
    modelInput = np.expand_dims(modelInput, axis=0)
    return modelInput

def postProcessSegmentation(segPrediction):
    segPred = np.argmax(segPrediction, axis=2)
    seg = np.zeros((224, 416, 3), dtype=np.uint8)

    seg[segPred == 0] = (255, 0, 0)
    seg[segPred == 1] = (0, 255, 0)
    seg[segPred == 2] = (0, 0, 255)

    return seg

def postProcessDepth(depthPrediction):
    depthImage = depthPrediction * 2
    depthImage = cv2.cvtColor(depthImage, cv2.COLOR_GRAY2RGB)
    depthImage = (depthImage * 255).astype(np.uint8)
    return depthImage

def postProcessLanes(lanesPrediction):
    lanesPred = np.argmax(lanesPrediction, axis=2)
    lanes = np.zeros((224, 416, 3), dtype=np.uint8)

    lanes[lanesPred == 0] = (255, 0, 0)
    lanes[lanesPred == 1] = (0, 255, 0)

    return lanes

def createSingleImageFromPredictions(image, segImage, laneImage, depthImage):
    topRow = np.concatenate([image, segImage], axis=1)
    bottomRow = np.concatenate([laneImage, depthImage], axis=1)
    finalImage = np.concatenate([topRow, bottomRow], axis=0)

    return finalImage

if __name__ == "__main__":

    imageSize = (416, 224)
    imageFilepath = "Images/TestImage.png"
    
    #Load the model
    model = load_model("CarVision.model", compile=False)

    #Load and resize the image
    image = cv2.imread(imageFilepath)
    image = cv2.resize(image, imageSize)
    
    #Preprocess the image for the neural network
    modelInput = preprocessImage(image)

    #Run the prediction
    prediction = model.predict([modelInput, np.random.rand(1, 95, 141, 3)])

    #Separate the predictions 
    segPrediction = prediction[0][0]
    lanesPrediction = prediction[1][0]
    test = np.argmax(lanesPrediction, axis=2)
    depthPrediction = prediction[2][0]

    #Create images for the predictions
    segImage = postProcessSegmentation(segPrediction)
    laneImage = postProcessLanes(lanesPrediction)
    depthImage = postProcessDepth(depthPrediction)

    #Create 2x2 image of input image, lanes, segmentation and depth
    finalImage = createSingleImageFromPredictions(image, segImage, laneImage, depthImage)

    #Display image
    cv2.imshow("Image", finalImage)
    cv2.waitKey(1)
