'''
 * BSD 3-Clause License
 * @copyright (c) 2019, Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * @file    human.py
 * @author  Krishna Bhatu, Hrishikesh Tawade, Kapil Rawal
 * @version 1.0
 * @brief  Implemetation of Lucas-Kanade algorithm on human dataset 
 *
 '''

import cv2
import numpy as np
import math
import glob
import os

# Robust Error Weights
def getRobustError(errorVector):
    var = np.var(errorVector)
    sd = np.sqrt(var)
    mean = np.mean(errorVector)
    n,it = errorVector.shape
    q = np.zeros((n,it))
    in1,in2 = np.where(np.abs(mean - errorVector) <= var)
    q[in1,in2] = 0.5
    in3,in4 = np.where(np.abs(mean - errorVector) > var)
    q[in3,in4] = 0.05
    return q

# Warp function for affine
def getWfromP(p):
    W = np.array([[1+p[0,0],p[0,2],p[0,4]],
                    [p[0,1], 1+p[0,3],p[0,5]]])
    return W 

# Wrapping the image
def wrappingFunction(I,W,Tpoints):
    n,it = Tpoints.shape
    transformedImagePoints = np.empty([2,n])
    transformedImagePoints = np.matmul(W,Tpoints.T)
    transformedImagePoints = transformedImagePoints.T
    transformedImageIntensities = np.empty([n,1])
    transformedImageIntensities[:,0] = I[transformedImagePoints[:,1].astype(int),transformedImagePoints[:,0].astype(int)]
     
    return transformedImagePoints,transformedImageIntensities

# Wrapping the gradient image
def wrappingFunctionOfGrad(gradientX, gradientY ,IWpoints):
    n,it = IWpoints.shape
    gradXIntensities = np.empty([n,1])
    gradYIntensities = np.empty([n,1])
    gradXIntensities[:,0] = gradientX[IWpoints[:,1].astype(int),IWpoints[:,0].astype(int)]
    gradYIntensities[:,0] = gradientY[IWpoints[:,1].astype(int),IWpoints[:,0].astype(int)]
    return gradXIntensities, gradYIntensities

# Calculating change in parameters p
def clacChangeInParams(error, IWdx, IWdy, TPoints, weights):
    img1 = IWdx[:,0] * [TPoints[:,0]]
    img2 = IWdx[:,0] * [TPoints[:,1]]
    img3 = IWdy[:,0] * [TPoints[:,0]]
    img4 = IWdy[:,0] * [TPoints[:,1]]
    dIW = np.hstack((img1.T,img3.T,img2.T,img4.T,IWdx,IWdy))
    sumP = np.matmul(dIW.T,error * weights)
    sumHess = np.matmul(dIW.T,weights * dIW)
    sumP = np.matmul(np.linalg.pinv(sumHess), sumP)
    return sumP

# LucasKanadeTracker implementation
def lucasKanadeTracker(Tpoints, Tintensity, I, p, startingPoint, endPoint):
    threshold = 0.07
    changeP = 100
    gradientX = cv2.Sobel(I,cv2.CV_64F,1,0,ksize=3)
    gradientY = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=3)
    it = 0
    safeW,safep = getWfromP(p),p 
    while(changeP > threshold):
        it += 1
        W = getWfromP(p)
        IWpoints, IWi = wrappingFunction(I,W,Tpoints)
        error = Tintensity - IWi
        weights = getRobustError(error)
        IWdx, IWdy = wrappingFunctionOfGrad(gradientX, gradientY ,IWpoints)
        deltaP= clacChangeInParams(error, IWdx, IWdy,Tpoints, weights)
        changeP = np.linalg.norm(deltaP)
        p[0,0] += deltaP[0,0]
        p[0,1] += deltaP[1,0]
        p[0,2] += deltaP[2,0]
        p[0,3] += deltaP[3,0]
        p[0,4] += deltaP[4,0]        
        p[0,5] += deltaP[5,0]
        newStart = np.array([[startingPoint[0]],[startingPoint[1]],[1]])
        newend = np.array([[endPoint[0]],[endPoint[1]],[1]])
        s = np.matmul(W,newStart)
        e = np.matmul(W,newend)
        if (it > 300):
            return safeW,safep 
    return W,p

# Selecting template from the image
def selectRectangle(event, x, y, flags, param):
    global startingPoint, endPoint
    if event == cv2.EVENT_LBUTTONDOWN:
        startingPoint = [x,y]
    elif event == cv2.EVENT_LBUTTONUP:
        endPoint = [x,y]
        cv2.rectangle(frame11, (startingPoint[0], startingPoint[1]), (endPoint[0], endPoint[1]),  (255,255,255), 2)
        cv2.imshow("Mark", frame11)
        cv2.waitKey(0)

global startingPoint, endPoint

# Taking input images
imgs = glob.glob('human/*.jpg')

# Setting output folder
folder = 'weightedOutput'

# Creating folder directory if one doesn't exits
if not os.path.exists(folder):
    os.makedirs(folder)

# Taking user input
inp = input("Do you want to select the bounding box for template or use the tested bounding box for best result?('y' for Yes and 'n' for no) :")
if(inp == "y"):
    frame11 = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)
    cv2.namedWindow("Mark")
    cv2.setMouseCallback("Mark", selectRectangle)
    cv2.imshow("Mark", frame11)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    startingPoint = [263,291]
    endPoint = [282,359]

# Converting to gray
frame1 = cv2.imread(imgs[0], 0)

h1 = abs(startingPoint[1] - endPoint[1])
w1 = abs(startingPoint[0] - endPoint[0])
frame1Points = np.empty([h1*w1,3])
n = 0

# Getting frames points
for i in range(startingPoint[0],endPoint[0]):
    for j in range(startingPoint[1],endPoint[1]):
        frame1Points[n,0] = i
        frame1Points[n,1] = j
        frame1Points[n,2] = 1
        n+=1

frame1Intensities = np.empty([h1*w1, 1])
n = 0
# Setting new points with intensity from image
for i in frame1Points:
    frame1Intensities[n,0] = frame1[int(i[1]),int(i[0])]
    n += 1

p = np.zeros([1,6], dtype = np.float)
framesSeen = 0
it = 0

# Main loop for all image sequence
for img in imgs:    

    frame1c = cv2.imread(img)
    frame2 = cv2.imread(img, 0)
    frame2 = frame2.astype(float)

    # LucasKanadeTracker
    updatedParam,p = lucasKanadeTracker(frame1Points, frame1Intensities, frame2, p, startingPoint, endPoint)
 
    # Updating boundary points
    newstartPoint = np.array([[startingPoint[0]],[startingPoint[1]], [1]])
    newendPoint = np.array([[endPoint[0]],[endPoint[1]], [1]])
    newStart = np.matmul(updatedParam, newstartPoint)
    newEnd = np.matmul(updatedParam, newendPoint)
    newstartrigth = np.array([[startingPoint[0]],[endPoint[1]], [1]])
    newstartleft = np.array([[endPoint[0]],[startingPoint[1]], [1]])
    newStartrigth = np.matmul(updatedParam, newstartrigth)
    newStartleft = np.matmul(updatedParam, newstartleft)
    print(it)
    framesSeen += 1
    if(framesSeen == 1000):
        p = np.zeros([1,6], dtype = np.float)
        startingPoint[0], startingPoint[1] = int(newStart[0]), int (newStart[1])
        endPoint[0], endPoint[1] = int(newEnd[0]), int(newEnd[1])
        h1 = abs(startingPoint[1] - endPoint[1])
        w1 = abs(startingPoint[0] - endPoint[0])
        frame1Points = np.empty([h1*w1,3])
        n = 0
        for i in range(startingPoint[0],endPoint[0]):
            for j in range(startingPoint[1],endPoint[1]):
                frame1Points[n,0] = i
                frame1Points[n,1] = j
                frame1Points[n,2] = 1
                n+=1

        frame1Intensities = np.empty([h1*w1, 1])
        n = 0
        for i in frame1Points:
            frame1Intensities[n,0] = frame1[int(i[1]),int(i[0])]
            n += 1
        framesSeen = 0

    # Creating bounding box
    cv2.line(frame1c, (newStart[0],newStart[1]), (newStartrigth[0], newStartrigth[1]), (0,0,255),2)
    cv2.line(frame1c, (newStart[0],newStart[1]), (newStartleft[0], newStartleft[1]), (0,0,255),2)
    cv2.line(frame1c, (newStartleft[0], newStartleft[1]), (newEnd[0], newEnd[1]), (0,0,255),2)
    cv2.line(frame1c, (newStartrigth[0], newStartrigth[1]), (newEnd[0], newEnd[1]), (0,0,255),2)
    
    cv2.imshow("Frameaa", frame1c)
    cv2.waitKey(1)

    # Writing the image
    cv2.imwrite(folder + '/' + img, frame1c)
    it += 1

print('All frames processed')
print('\nPress \'q\' to destroy window')
cv2.waitKey(0)
cv2.destroyAllWindows()