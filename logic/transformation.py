import cv2 as cv
import cv2 #as cv
import numpy as np
class ImageTrasformation:
    def __init__(self, img_path):
        self.img_path = img_path

    def rotate(self, show = True , write=False):
        # Reading the image
        image = cv2.imread(self.img_path)

        # dividing height and width by 2 to get the center of the image
        height, width = image.shape[:2]
        # get the center coordinates of the image to create the 2D rotation matrix
        center = (width/2, height/2)

        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

        # rotate the image using cv2.warpAffine
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        if(show):
            cv2.imshow('Original image', image)
            cv2.imshow('Rotated image', rotated_image)
        if(write):
            # wait indefinitely, press any key on keyboard to exit
            #cv2.waitKey(0)
            # save the rotated image to disk
            cv2.imwrite('rotated_image.jpg', rotated_image)

    def translate(self, show = True , write=False):
        pass

    def crop(self,resize, show = True , write=False):
        xy = resize
        img = cv2.imread(self.img_path)
        print(f"image shape : {img.shape}") # Print image shape
        cv2.imshow("original", img)

        # Cropping an image
        # cropped = img[start_row:end_row, start_col:end_col]
        cropped_image = img[xy[1]:xy[2], xy[3]:xy[4]]
        if(show):
            # Display cropped image
            cv2.imshow("cropped", cropped_image)
        if(write):
            # Save the cropped image
            cv2.imwrite("Cropped Image.jpg", cropped_image)

    def flip_image(self,show = True , write=False):
        img = cv.imread(self.img_path, cv.IMREAD_COLOR)
        cv.imshow('input', img)
        #  Flip up and down 
        dst1 = cv.flip(img, 0)
        res1 = np.vstack((img, dst1))
        #  Flip left and right 
        dst2 = cv.flip(img, 1)
        res2 = np.vstack((img, dst2))
        #  Turn diagonally 
        dst3 = cv.flip(img, -1)
        res3 = np.vstack((img, dst3))
        
        #  All flipping results are displayed in the same window 
        result = np.hstack((res1, res2, res3))
        if(show):
            cv.imshow('flip', result)
        if(write):
            cv.imwrite('images/result_flip.jpg', result)

    
    def rotate_image(self):
        """
        Rotated image , Introduce two rotation modes .
        1、 Specific angle rotation function , But only support 90、180、270 Rotate at such a special angle .
        2、 Arbitrary angle rotation function , Need to rotate the matrix M, There are two ways to get the rotation matrix M The way ： Manual configuration （ It can realize the rotating image without clipping ） And built-in functions to get 
        :param image_path:  Incoming image file 
        :return:  no return value 
        """
        img = cv.imread(self.img_path, cv.IMREAD_COLOR)
        cv.imshow('input', img)

        h, w, c = img.shape

        # ### The following rotation methods obtain the cropped rotating image #######
        # ########## Manually set the rotation matrix M#################
        #  Define an empty matrix 
        M = np.zeros((2, 3), dtype=np.float32)

        #  Set the rotation angle 
        alpha = np.cos(np.pi / 4.0)
        beta = np.sin(np.pi / 4.0)
        print('alpha: ', alpha)
        #  Initialize the rotation matrix 
        M[0, 0] = alpha
        M[1, 1] = alpha
        M[0, 1] = beta
        M[1, 0] = -beta

        #  The coordinates of the center of the picture 
        cx = w / 2
        cy = h / 2

        #  Variable width and height 
        tx = (1 - alpha) * cx - beta * cy
        ty = beta * cx + (1 - alpha) * cy

        M[0, 2] = tx
        M[1, 2] = ty

        #  The built-in function obtains the rotation matrix M, A positive value indicates a counter clockwise rotation , Suppose the upper left corner is the coordinate origin 
        M = cv.getRotationMatrix2D((w / 2, h / 2), 45, 1)
        #  Perform rotating ,  Rotate at any angle 
        result = cv.warpAffine(img, M, (w, h))

        # ####### Built in rotation function , Support only 90,180,270#################
        dst1 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        dst2 = cv.rotate(img, cv.ROTATE_180)
        dst3 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

        #  take 4 An image is displayed in a window , Be careful ： The four images have the same shape , Otherwise, an error will be reported 
        res = np.hstack((img, dst1, dst2, dst3))
        cv.imwrite('images/rotate4.jpg', res)
        cv.imshow('res', res)

        #  Display the rotation image result of manually setting the rotation angle 
        result = np.hstack((img, result))
        cv.imwrite('images/rotate2.jpg', result)
        cv.imshow('rotate center', result)

        # # # ####### Get a rotated image without clipping #########
        # #  Define an empty matrix 
        # M = np.zeros((2, 3), dtype=np.float32)
        # #  Set the rotation angle 
        # alpha = np.cos(np.pi / 4.0)
        # beta = np.sin(np.pi / 4.0)
        # print('alpha: ', alpha)
        # #  Initialize the rotation matrix 
        # M[0, 0] = alpha
        # M[1, 1] = alpha
        # M[0, 1] = beta
        # M[1, 0] = -beta
        # #  The coordinates of the center of the picture 
        # cx = w / 2
        # cy = h / 2
        #
        # #  Variable width and height 
        # tx = (1 - alpha) * cx - beta * cy
        # ty = beta * cx + (1 - alpha) * cy
        # M[0, 2] = tx
        # M[1, 2] = ty
        #
        # #  The rotated image is high 、 wide 
        # rotated_w = int(h * np.abs(beta) + w * np.abs(alpha))
        # rotated_h = int(h * np.abs(alpha) + w * np.abs(beta))
        #
        # #  Center position after movement 
        # M[0, 2] += rotated_w / 2 - cx
        # M[1, 2] += rotated_h / 2 - cy
        #
        # result = cv.warpAffine(img, M, (rotated_w, rotated_h))
        # cv.imshow('result', result)