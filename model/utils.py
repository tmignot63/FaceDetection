import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def getBigRectangles(recognisedFacesCenters, recognisedFacesPercentages, recognisedFacesCenterSizes):
    ''' Arrays containing only the final frame'''
    recognisedBigFacesCenters = []
    recognisedBigFacesCentersSizes = []
    recognisedBigFacesPercentages = []

    ''' Putting the higest probability frame in the final array by default '''
    maxvalueCenters = max(recognisedFacesPercentages)
    maxposCenters = recognisedFacesPercentages.index(maxvalueCenters)

    recognisedBigFacesCenters.append(recognisedFacesCenters[maxposCenters])
    recognisedBigFacesCentersSizes.append(
        recognisedFacesCenterSizes[maxposCenters])
    recognisedBigFacesPercentages.append(
        recognisedFacesPercentages[maxposCenters])

    ''' Purging initial arrays of the values  we just put in the final arrays'''
    recognisedFacesCenters.pop(maxposCenters)
    recognisedFacesPercentages.pop(maxposCenters)
    recognisedFacesCenterSizes.pop(maxposCenters)

    for i in range(len(recognisedFacesCenters)):
        #print('ok ', i)
        maxvalueCenters = max(recognisedFacesPercentages)
        maxposCenters = recognisedFacesPercentages.index(maxvalueCenters)
        # print(maxposCenters)
        # print(recognisedFacesCenters[maxposCenters])
        # print(recognisedFacesCenterSizes[maxposCenters])
        test = getTowCornersOfRectangle(
            recognisedFacesCenters[maxposCenters], recognisedFacesCenterSizes[maxposCenters], recognisedBigFacesCenters, recognisedBigFacesCentersSizes)
        ''' If the area are not overlapping then add the tested frame into the final arrays '''
        if(test == 1):
            recognisedBigFacesCenters.append(
                recognisedFacesCenters[maxposCenters])
            recognisedBigFacesCentersSizes.append(
                recognisedFacesCenterSizes[maxposCenters])
            recognisedBigFacesPercentages.append(
                recognisedFacesPercentages[maxposCenters])
        ''' Purging initial arrays of the tested values'''
        recognisedFacesCenters.pop(maxposCenters)
        recognisedFacesPercentages.pop(maxposCenters)
        recognisedFacesCenterSizes.pop(maxposCenters)
    return [recognisedBigFacesCenters, recognisedBigFacesCentersSizes, recognisedBigFacesPercentages]


def getTowCornersOfRectangle(centerToTest, sizeToTest, recognisedFacesCenters, recognisedFacesCenterSizes):
    #print('centertottest ', centerToTest)
    #print('sizetottest ',sizeToTest)
    lToTest = (centerToTest[0]-int(sizeToTest/2),
               centerToTest[1]+int(sizeToTest/2))
    rToTest = (centerToTest[0]+int(sizeToTest/2),
               centerToTest[1]-int(sizeToTest/2))
    for i in range(len(recognisedFacesCenters)):
        #print('recognisedFacesCenters ', recognisedFacesCenters[i])
        #print('recognisedFacesCenterSizes ',recognisedFacesCenterSizes[i])
        lArray = (recognisedFacesCenters[i][0]-int(recognisedFacesCenterSizes[i]/2),
                  recognisedFacesCenters[i][1]+int(recognisedFacesCenterSizes[i]/2))
        rArray = (recognisedFacesCenters[i][0]+int(recognisedFacesCenterSizes[i]/2),
                  recognisedFacesCenters[i][1]-int(recognisedFacesCenterSizes[i]/2))

        if(overlappingArea(lToTest, rToTest, lArray, rArray) > 0.5):
            return -1
    return 1


def overlappingArea(l1, r1, l2, r2):
    x = 0
    y = 1

    # Area of 1st Rectangle
    area1 = abs(l1[x] - r1[x]) * abs(l1[y] - r1[y])

    # Area of 2nd Rectangle
    area2 = abs(l2[x] - r2[x]) * abs(l2[y] - r2[y])

    ''' Length of intersecting part i.e 
        start from max(l1[x], l2[x]) of 
        x-coordinate and end at min(r1[x],
        r2[x]) x-coordinate by subtracting 
        start from end we get required 
        lengths '''
    x_dist = (min(r1[x], r2[x]) -
              max(l1[x], l2[x]))

    y_dist = (-min(r1[y], r2[y]) +
              max(l1[y], l2[y]))
    areaI = 0
    if x_dist > 0 and y_dist > 0:
        areaI = x_dist * y_dist

    ''' get the smallest of the two areas'''
    if(area1 < area2):
        percentageArea = areaI/area1
    else:
        percentageArea = areaI/area2
    #print('l1', l1)
    #print('r1', r1)
    #print('l2', l2)
    #print('r2', r2)
    #print('x_dist', x_dist)
    #print('y_dist', y_dist)
    #print('areaI', areaI)
    #print('area1', area1)
    #print('area2', area2)
    return percentageArea
