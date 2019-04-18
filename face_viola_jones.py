import cv2

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
faceCascade = cv2.CascadeClassifier('pth_/haarcascade_frontalface_default.xml')


OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600



def detectLargestFace():
    #Open the first webcame device
    capture = cv2.VideoCapture(0)

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)


    cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)


    cv2.startWindowThread()

    rectangleColor = (180,0,155)

    try:
        while True:
            #Retrieve the image
            rc,fullSizeBaseImage = capture.read()

            #Resize the image
            baseImage = cv2.resize( fullSizeBaseImage, ( 320, 240))



            #destroy all opencv windows and exit the application
            pressedKey = cv2.waitKey(2)
            if pressedKey == ord('Q'):
                cv2.destroyAllWindows()
                exit(0)
                
            resultImage = baseImage.copy()
            
            gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
            #Now use the haar cascade detector 
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            
            maxArea = 0
            x = 0
            y = 0
            w = 0
            h = 0


            #Loop over all faces and check if the area for this face is
            #the largest so far
            for (_x,_y,_w,_h) in faces:
                if  _w*_h > maxArea:
                    x = _x
                    y = _y
                    w = _w
                    h = _h
                    maxArea = w*h


            #draw the largest face present in the picture
            if maxArea > 0 :
                cv2.rectangle(resultImage,  (x-10, y-20),
                                            (x + w+10 , y + h+20),
                                            rectangleColor,2)



            #Show larger than the
            #original 320x240, resize the image again

            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

            #Finally,show the images on the screen
            cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)


    #check for the KeyboardInterrupt exception and destroy
    #all opencv windows and exit the application
    except KeyboardInterrupt as e:
        cv2.destroyAllWindows()
        exit(0)


if __name__ == '__main__':
    detectLargestFace()
