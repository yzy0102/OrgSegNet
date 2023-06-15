from PIL import Image, ImageTk
import logging
import os
import sys
import numpy as np
from numpy import linalg
import glob
import skimage
from skimage import io, color, morphology, filters, transform, measure, exposure, restoration, feature, draw
from skimage.morphology import disk
from skimage.util import dtype
import scipy as sp
from scipy import ndimage, stats, spatial, cluster
import itertools
import pandas as pd
import networkx as nx
from packaging.version import Version
import math
import shapely
from shapely import geometry
import pickle
import time
import sklearn
from sklearn import decomposition
import matplotlib
import matplotlib.pyplot as plt

class VisGraphOther:

    def __init__(self, selectedImage, resolution, outputFolder, organelle, inputType='image', fileList=None):
        self.selectedImage = selectedImage
        self.outputFolder = outputFolder
        self.resolution = float(resolution)
        self.inputType = inputType
        self.fileList = fileList
        self.organelle = organelle
        self.sigmas = []
        self.shapeResultsTable = pd.DataFrame(columns=['File', 'LabeledImage', 'GraphNumber', '#Nodes', '#Edges', 'Complexity'])
        if os.path.isfile(self.outputFolder + '/visibilityGraphs.gpickle'):
            try:
                os.remove(self.outputFolder + '/visibilityGraphs.gpickle')
                os.remove(self.outputFolder + '/visibilityContours.gpickle')
                os.remove(self.outputFolder + '/shapeResultsTable.csv')
            except:
                pass

        if self.inputType == 'image':
            self.labeledImage, self.labels = self.label_binary_image(self.selectedImage)
            self.visibilityGraphsOther = self.visibility_graphs_other(self.labeledImage, self.labels, self.resolution)

            for graph in self.visibilityGraphsOther.keys():
                self.sigmas.append(self.add_data_to_table(self.visibilityGraphsOther[graph], self.selectedImage, graph, 'LabeledShapes.png', self.organelle))
            self.plot_labeled_image(self.labeledImage, self.outputFolder, self.labels, 'image', 1)
        else:
            graphIndex = 1
            self.visibilityGraphsOther = {}
            for fileIndex, file in enumerate(self.fileList):

                self.labeledImage, self.labels = self.label_binary_image(file)

                self.visibilityGraph = self.visibility_graphs_other(self.labeledImage, self.labels, self.resolution)
                labeledFile = self.plot_labeled_image(self.labeledImage, self.outputFolder, self.labels, 'folder', graphIndex)
                if len(self.visibilityGraph) == 1:
                    self.visibilityGraphsOther[graphIndex] = list(self.visibilityGraph.values())[0]
                    self.sigmas.append(self.add_data_to_table(list(self.visibilityGraph.values())[0], file, graphIndex, labeledFile, self.organelle))
                    graphIndex += 1
                else:
                    for graph in self.visibilityGraph.keys():
                        self.visibilityGraphsOther[graphIndex] = self.visibilityGraph[graph]
                        self.sigmas.append(self.add_data_to_table(self.visibilityGraph[graph], file, graphIndex, labeledFile, self.organelle))
                        graphIndex += 1

    def label_binary_image(self, selectedImage):
        """
        check if input is binary image and label all objects (white)
        """
        # rawImage = skimage.io.imread(selectedImage)
        # if len(rawImage.shape) == 2:
        #     if len(np.unique(rawImage)) == 2:
        #         binaryImage = rawImage > 0
        #         if (1 in binaryImage[0, :]) or (1 in binaryImage[-1, :]) or (1 in binaryImage[:, 0]) or (1 in binaryImage[:, -1]):
        #             show_Message("...Detected objects at image border. Added padding to binary image.")
        #             binaryImage = np.pad(binaryImage, pad_width=3, mode='constant', constant_values=0)
        #         labeledImage, labels = sp.ndimage.label(binaryImage, np.ones((3,3)))
        
        return (selectedImage, selectedImage.max())

    def visibility_graphs_other(self, labeledImage, labels, resolution):
        """
        create a visibility graph for all cells
        """
        visGraphsAll = {}
        cellContoursAll = {}
        for label in range(1, labels+1):
            visGraph, cellContour = self.create_visibility_graph(labeledImage, label, resolution)
            if visGraph != None:
                visGraphsAll[label] = visGraph
                # visGraphsOtherPickle = open(self.outputFolder + '/visibilityGraphs.gpickle', 'ab')
                # pickle.dump(visGraph, visGraphsOtherPickle)
                # visGraphsOtherPickle.close()
                # cellContoursOtherPickle = open(self.outputFolder + '/visibilityContours.gpickle', 'ab')
                # pickle.dump(cellContour, cellContoursOtherPickle)
                # cellContoursOtherPickle.close()

        return(visGraphsAll)

    def create_visibility_graph(self, labeledImage, label, resolution):
        """
        create visibilit graph from cell contour
        """
        visGraph = nx.Graph()
        cases = ['FFFF0F212','0FFF0F212','1FFF0F212','F0FF0F212','00FF0F212','10FF0F212','F1FF0F212']
        contourImage, cellContourOrdered = self.extract_cell_contour(label, labeledImage)
        if len(cellContourOrdered) != 0:
            pixelsOnContour = interpolate_contour_pixels(cellContourOrdered, resolution)
            if len(pixelsOnContour) != 0:
                for key in pixelsOnContour:
                    visGraph.add_node(key, pos=(pixelsOnContour[key][0], pixelsOnContour[key][1]))
                visGraph = self.add_edges_to_visGraph(pixelsOnContour, visGraph, cases)
            else:
                visGraph, cellContourOrdered = None, None
        else:
            visGraph, cellContourOrdered = None, None
        return(visGraph, cellContourOrdered)

    def extract_cell_contour(self, label, labeledImage):
        """
        extract the contour of a specified cell
        """
        cellImage = labeledImage == label
        contourImage = invert(cellImage)

        if np.all([np.all(cellImage[..., 0:, 0] == 0), np.all(cellImage[..., 0, 0:] == 0), np.all(cellImage[..., 0:, -1] == 0), np.all(cellImage[..., -1, 0:] == 0),]):
            cellContour = find_contour_of_object(cellImage)
            cellContourOrdered = marching_squares(cellContour, cellImage)
        else:
            cellImageBuffer = np.pad(cellImage, pad_width=2, mode='constant', constant_values=0)
            cellContour = find_contour_of_object(cellImageBuffer)
            cellContourOrdered = marching_squares(cellContour, cellImageBuffer)

        for xPos, yPos in cellContourOrdered:
            try:
                contourImage[xPos, yPos] = 1
            except:
                pass#print()

        return(contourImage, cellContourOrdered)

    def add_edges_to_visGraph(self, pixelsOnContour, visGraph, cases):
        """
        add edge to visGraph if the edge between two nodes lies inside the cell (concave)
        """
        Polygon = shapely.geometry.Polygon([[pixelsOnContour[key][1], pixelsOnContour[key][0]] for key in pixelsOnContour])
        Boundary = shapely.geometry.LineString(list(Polygon.exterior.coords))
        combs = itertools.combinations(range(len(pixelsOnContour)), 2)
        for node1, node2 in list(combs):
            line = shapely.geometry.LineString(((pixelsOnContour[node1][1], pixelsOnContour[node1][0]), (pixelsOnContour[node2][1], pixelsOnContour[node2][0])))
            DE9IM = line.relate(Polygon)

            if DE9IM in cases:
                intersection = Boundary.intersection(line)
                if DE9IM == '10FF0F212' and len(intersection) <= 3:
                    visGraph.add_edge(node1, node2, length=euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
                if DE9IM == 'F1FF0F212' and intersection.geom_type == 'LineString':
                    visGraph.add_edge(node1, node2, length=euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
                if DE9IM in cases[:5]:
                    visGraph.add_edge(node1, node2, length=euclidean(pixelsOnContour[node1], pixelsOnContour[node2]))
        return(visGraph)

    def add_data_to_table(self, visGraph, file, index, labeledFile, organelle):
        """
        summarize all results in a table
        """
        fileName = organelle # file.split('/')[-1]
        sigma = self.compute_graph_complexity(visGraph)
        dataAppend = [fileName, labeledFile, index, visGraph.number_of_nodes(), visGraph.number_of_edges(), sigma]
        self.shapeResultsTable.loc[0] = dataAppend
        # if not os.path.isfile(self.outputFolder + '/ShapeResultsTable.csv'):
        #     self.shapeResultsTable.to_csv(self.outputFolder + '/ShapeResultsTable.csv', mode='a', index=False)
        # else:
        #     self.shapeResultsTable.to_csv(self.outputFolder + '/ShapeResultsTable.csv', mode='a', index=False, header=False)
        return round(sigma, 4)


    def compute_graph_complexity(self, visGraph):
        """
        compute the complexity of the graph using the relative density of the clique
        """
        edgesCompleteGraph = (visGraph.number_of_nodes() * (visGraph.number_of_nodes() - 1)) * 0.5
        delta = visGraph.number_of_edges() / edgesCompleteGraph
        return(delta)

    def plot_labeled_image(self, labeledImage, outputFolder, labels, fileType, index):
        """
        plot the labeled image with labels
        """
        textPositions = []
        textString = []
        for idx in range(1, labels + 1):
            label = labeledImage == idx
            cmx, cmy = sp.ndimage.measurements.center_of_mass(label)
            textPositions.append([cmx, cmy])
            graphNumber = idx + index - 1
            textString.append(str(graphNumber))

        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        plt.imshow(labeledImage, cmap='viridis')
        axs.axes.get_yaxis().set_visible(False)
        axs.axes.get_xaxis().set_visible(False)
        for idx in range(len(textString)):
            plt.text(textPositions[idx][1], textPositions[idx][0], textString[idx], fontsize=8, color='black')
        if fileType == 'image':
            fig.savefig(outputFolder + '/LabeledShapes.png', bbox_inches='tight', dpi=300)
        else:
            fig.savefig(outputFolder + '/LabeledShapes_' + str(index) + '.png', bbox_inches='tight', dpi=300)
            return('LabeledShapes_' + str(index) + '.png')



##### general functions #####
def show_Message(msg):
    logging.info(msg)

def keep_labels_on_border(labeledImage):
    """
    modified version of skimage.segmentation.clear_border to keep only labels touching the image border
    """
    image = labeledImage
    # create borders with buffer_size
    borders = np.zeros_like(image, dtype=np.bool_)
    ext = 1
    slstart = slice(ext)
    slend   = slice(-ext, None)
    slices  = [slice(s) for s in image.shape]
    for d in range(image.ndim):
        slicedim = list(slices)
        slicedim[d] = slstart
        borders[tuple(slicedim)] = True
        slicedim[d] = slend
        borders[tuple(slicedim)] = True

    labels = skimage.measure.label(image, background=0)
    number = np.max(labeledImage) + 1
    # determine all objects that are connected to borders
    borders_indices = np.unique(labeledImage[borders])
    indices = np.arange(number + 1)
    # mask all label indices that are not connected to borders
    label_mask = ~np.in1d(indices, borders_indices)
    mask = label_mask[labeledImage.ravel()].reshape(labeledImage.shape)
    image[mask] = 0
    return(image)

def find_edge_contour(image):
    """
    extract contour of artificial edges
    """
    # add buffer to image to detect contour also on borders
    bufferedImage = skimage.util.pad(image, pad_width=2, mode='constant')
    edgeContour = find_contour_of_object(bufferedImage)
    return(edgeContour)

def find_contour_of_object(cellObject):
    """
    find contour of an object
    """
    contour = []
    coord = np.transpose(np.where(cellObject != 0))
    lenX, lenY = cellObject.shape[0] - 1, cellObject.shape[1] - 1

    for x,y in coord:

        xmin,xmax = bounds(x-1, 0, lenX), bounds(x+1, 0, lenX)
        ymin,ymax = bounds(y-1, 0, lenY), bounds(y+1, 0, lenY)
        if xmax != x and xmin != x and ymax != y and ymin != y:
            if cellObject[xmin, y] == 0 and [xmin, y] not in contour:
                    contour.append([xmin, y])
            if cellObject[x, ymin] == 0 and [x, ymin] not in contour:
                    contour.append([x, ymin])
            if cellObject[x, ymax] == 0 and [x, ymax] not in contour:
                    contour.append([x, ymax])
            if cellObject[xmax, y] == 0 and [xmax, y] not in contour:
                    contour.append([xmax, y])
        else:
            if [x, y] not in contour:
                contour.append([x, y])
    return(np.asarray(contour))

def bounds(x,xmin,xmax):
    """
    define bounds of image window
    """
    if (x <= xmin): #if x is smaller than xmin, set x to xmin
        x = xmin
    elif ( x >= xmax):   #if x is larger than xmax, set x to xmax
        x = xmax
    return(x)

def contour_orientation(window):
    """
    define the orientatin of the image pixel window
    """
    orient = []
    if np.sum(window[:, 0] > 0) == 3:
        orient.append('left')
    if np.sum(window[:, 2] > 0) == 3:
        orient.append('right')
    if np.sum(window[0, :] > 0) == 3:
        orient.append('top')
    if np.sum(window[2, :] > 0) == 3:
        orient.append('bottom')
    return(orient)

def measure_intensity_along_contour(image, x, y, orientation, listOfCorrectedPixels):
    """
    measure intensity along contour to detect intensity gradients
    """
    lx, ly = image.shape[0] - 1, image.shape[1] - 1
    if 'top' in orientation:
        xmin, xmax = bounds(x-10, 0, lx), bounds(x+1, 0, lx)
        ymin, ymax = bounds(y-1, 0, ly), bounds(y+2, 0, ly)
        new_window = image[xmin:xmax, ymin:ymax].astype('int')
        window_means = np.mean(new_window, axis=1)[::-1]

    if 'right' in orientation:
        xmin, xmax = bounds(x-1, x, lx), bounds(x+2, 0, lx)
        ymin, ymax = bounds(y, 0, ly), bounds(y+11, 0, ly)
        new_window = image[xmin:xmax, ymin:ymax].astype('int')
        window_means = np.mean(new_window, axis=0)

    if 'bottom' in orientation:
        xmin, xmax = bounds(x, 0, lx), bounds(x+11, 0, lx)
        ymin, ymax = bounds(y-1, 0, ly), bounds(y+2, 0, ly)
        new_window = image[xmin:xmax, ymin:ymax].astype('int')
        window_means = np.mean(new_window, axis=1)

    if 'left' in orientation:
        xmin, xmax = bounds(x-1, 0, lx), bounds(x+2, 0, lx)
        ymin, ymax = bounds(y-10, 0, ly), bounds(y+1, 0, ly)
        new_window = image[xmin:xmax, ymin:ymax].astype('int')
        window_means = np.mean(new_window, axis=0)[::-1]

    if len(window_means) > 5:
        window_percentage = window_means[1:] * 100 / window_means[0]
        if window_percentage[2] > 25 and [x, y] not in listOfCorrectedPixels:
            listOfCorrectedPixels.append([x, y])
    return(listOfCorrectedPixels)

def tube_filter(image, sigma):
    """
    enhance tube-like structures
    """
    if Version(skimage.__version__) < Version('0.14'):
        imH=skimage.feature.hessian_matrix(image, sigma=sigma, mode='reflect')
        imM=skimage.feature.hessian_matrix_eigvals(imH[0], imH[1], imH[2])
    else:
        imH=skimage.feature.hessian_matrix(image, sigma=sigma, mode='reflect', order='xy')
        imM=skimage.feature.hessian_matrix_eigvals(imH)
    imR = -1.0 * imM[1]
    imT = 255.0 * (imR - imR.min()) / (imR.max() - imR.min())
    imT = imT.astype('int')
    return(imT)

def correct_gaps_in_skeleton(skeletonImage):
    """
    find gaps in the skeleton and close them if the gap is small and both ends have the same direction/angle
    """
    skeletonImage = skeletonImage * 1
    correctedSkeletonImage = skeletonImage.copy()
    endpoints = detect_crossings_and_endpoints(skeletonImage, mode='endpoints', output='list')
    if len(endpoints) != 0:
        distanceBins = sort_coordinate_distances(endpoints)
        correctingEndpoints = np.transpose(np.where(distanceBins==1))
        imageEndpointsCrossings = detect_crossings_and_endpoints(skeletonImage, mode='both', output='image')

        for xPos, yPos in correctingEndpoints:
            angles, rows, columns = evaluate_angle(xPos, yPos, endpoints, imageEndpointsCrossings)
            if len(angles) != 0 and (np.max(angles) - np.min(angles) < 20):
                correctedSkeletonImage[rows, columns] = 2
                #print("Correction added: ", str(xPos), str(yPos))
    return(correctedSkeletonImage)

def evaluate_angle(x, y, endpoints, image):
    """
    evaluate whether the angles of both endpoints are similar
    """
    allAngles = []
    xPos1, yPos1 = endpoints[x]
    xPos2, yPos2 = endpoints[y]
    rows, columns = skimage.draw.line(xPos1, yPos1, xPos2, yPos2)
    if np.sum(image[rows[1:-1], columns[1:-1]]) == 0:
        angleEndpoint1 = measure_angle_of_endpoints(xPos1, yPos1, image)
        angleEndpoint2 = measure_angle_of_endpoints(xPos2, yPos2, image)
        if xPos2 < xPos1:
            angleBetweenEndpoints = angle180([yPos2 - yPos1, xPos2 - xPos1])
        else:
            angleBetweenEndpoints = angle180([yPos1 - yPos2, xPos1 - xPos2])
        allAngles = [angleEndpoint1, angleEndpoint2, angleBetweenEndpoints]
    return(allAngles, rows, columns)

def create_window(image, x, y, xUp, xDown, yLeft, yRight):
    """
    create a window from the specified coordinates in the image
    """
    lx, ly = image.shape[0] - 1, image.shape[1] - 1
    xmin, xmax = bounds(x - xUp, 0, lx), bounds(x + xDown, 0, lx)
    ymin, ymax = bounds(y - yLeft, 0, ly), bounds(y + yRight, 0, ly)
    window = image[xmin:xmax, ymin:ymax].copy()
    return(window, [xmin, xmax, ymin, ymax])

def sort_coordinate_distances(points):
    """
    sort the distances of different points into bins
    """
    distance = sp.spatial.distance_matrix(points, points)
    bins = [0, 1, 10, 20, 50, 100, 500, 1000, 9999]
    distance_bins = np.zeros((len(points), len(points))).astype('int')
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        ida = (distance >= b1) * (distance < b2)
        distance_bins[ida] = i
    distance_bins = np.tril(distance_bins)
    return(distance_bins)

def detect_crossings_and_endpoints(skeletonImage, mode='both', output='image'):
    """
    detect crossings and endpoints of the skeleton
    """
    skeletonImage = skeletonImage * 1
    detected_nodes = skeletonImage.copy()
    node_list = []
    coord = np.transpose(np.where(skeletonImage == 1))
    for x, y in coord:
        window, winBounds = create_window(skeletonImage, x, y, 1, 2, 1, 2)
        window[x - winBounds[0], y - winBounds[2]] = 0
        labeledWindow, L = sp.ndimage.label(window)
        if mode == 'both' or mode == 'endpoints':
            if L == 1 or L == 0:
                detected_nodes[x, y] = 3
                node_list.append([x, y])
        if mode == 'both' or mode == 'crossings':
            if L == 3 or L == 4:
                detected_nodes[x, y] = 2
                node_list.append([x, y])

    if output == 'image':
        return(detected_nodes)
    else:
        return(np.asarray(node_list))

def angle180(dxy):
    """
    calculate the angle between two points in 180 degree range
    """
    dx, dy = dxy
    rad2deg = 180.0 / np.pi
    angle = np.mod(np.arctan2(-dx, -dy) * rad2deg + 360.0, 360.0)
    if angle >= 270:
        angle = 360 - angle
    return(angle)

def measure_angle_of_endpoints(x, y, image):
    """
    measure the angle between two endpoint of the skeleton
    """
    window, winBounds = create_window(image, x, y, 5, 6, 5, 6)
    cleanedWindow = (window > 0) * 1
    labeledWindow, labelsWindow = sp.ndimage.label(cleanedWindow, np.ones((3, 3)))
    labelSkeleton = labeledWindow[x - winBounds[0], y - winBounds[2]]
    coordLabel = np.transpose(np.where(labeledWindow == labelSkeleton))
    newWindow = window.copy() * 0
    for s, t in coordLabel:
        newWindow[s, t] = window[s, t]

    if 2 in newWindow:
        coord = np.transpose(np.where(newWindow == 2))
        dista = []
        for s, t in coord:
            dista.append(euclidean([s, t], [x - winBounds[0], y - winBounds[2]]))
        w = np.argmin(dista)
        if x < coord[w][0] + winBounds[0]:
            angle = angle180([y - (coord[w][1] + winBounds[2]), x - (coord[w][0] + winBounds[0])])
        else:
            angle = angle180([(coord[w][1] + winBounds[2]) - y, (coord[w][0] + winBounds[0]) - x])
    else:
        coord = np.transpose(np.where(window == 1))
        dista = []
        for s, t in coord:
            dista.append(euclidean([s, t], [x - winBounds[0], y - winBounds[2]]))
        if len(dista) != 0:
            w = np.argmax(dista)
            if x < coord[w][0] + winBounds[0]:
                angle = angle180([y - (coord[w][1] + winBounds[2]), x - (coord[w][0] + winBounds[0])])
            else:
                angle = angle180([(coord[w][1] + winBounds[2]) - y, (coord[w][0] + winBounds[0]) - x])
        else:
            angle = 0
    return(angle)

def euclidean(x, y):
    """
    calculate the Euclidean distance between two points
    """
    dist = math.sqrt(((int(x[0]) - int(y[0])) ** 2) + ((int(x[1]) - int(y[1])) ** 2))
    return(dist)

def detect_branches(skeletonImage, mode='remove'):
    """
    remove skeleton branches by tracking from endpoints back to crossings
    """
    detected_nodes = detect_crossings_and_endpoints(skeletonImage, mode='both', output='image')
    branch_filament = (detected_nodes == 3).sum()
    while branch_filament > 0:
        branchless = track_or_remove_branches(detected_nodes, mode=mode)
        detected_nodes = detect_crossings_and_endpoints(branchless, mode='both', output='image')
        branch_filament = (detected_nodes == 3).sum()
    if mode == 'remove':
        return(detected_nodes > 0)
    else:
        return(detected_nodes)

def track_or_remove_branches(detected_nodes, mode):
    """
    depending on the mode either remove or track detected branches
    """
    branch_filament = (detected_nodes==3).sum()
    while branch_filament > 0:
        coord = np.transpose(np.where(detected_nodes==3))
        for x, y in coord:
            window, winBounds = create_window(detected_nodes, x, y, 1, 2, 1, 2)
            label_counts = np.sum(np.unique(window) > 0)
            label_sum = np.sum(window)
            label_number = np.sum(window > 0)
            if mode == 'remove':
                if label_counts == 2:
                    detected_nodes[winBounds[0]:winBounds[1], winBounds[2]:winBounds[3]] = np.where(window == 1, 3, window)
                detected_nodes[x, y] = 0
            else:
                if (label_counts <= 3 and label_sum < 9) or (label_sum == 9 and label_number == 4):
                    detected_nodes[winBounds[0]:winBounds[1], winBounds[2]:winBounds[3]] = np.where(window == 1, 3, window)
                detected_nodes[x, y] = 4
        branch_filament = (detected_nodes == 3).sum()
    if mode == 'remove':
        return(detected_nodes > 0)
    else:
        return(detected_nodes)

def create_labeled_and_tracked_image(skeletonImage, labeledImage):
    """
    create a labeled image, where background=0, skeleton=1, tracked branches=2 and cell labels>=3
    """
    labeledTrackedImage = labeledImage.copy() + 3
    trackedImage = detect_branches(skeletonImage, mode='track')
    trackedPixels = np.transpose(np.where(trackedImage ==4))
    skeletonPixels = np.transpose(np.where(labeledTrackedImage == 3))
    backgroundPixels = np.transpose(np.where(labeledTrackedImage == 4))

    labeledTrackedImage[backgroundPixels[:,0], backgroundPixels[:,1]] = 0
    labeledTrackedImage[skeletonPixels[:,0], skeletonPixels[:,1]] = 1
    labeledTrackedImage[trackedPixels[:,0], trackedPixels[:,1]] = 2
    return(labeledTrackedImage)

def marching_squares(contour, cellImage):
    """
    sort contour coordinates using marchin squares algorithm
    """
    contourCopy = contour.copy()
    orderedContour = np.empty(shape = [0, 2])
    xRight,yRight = find_rightmost_point(contour)
    contourImage = cellImage.copy() * 2
    contourImage[contour[:, 0], contour[:, 1]] = 1
    timeout = 120
    startTime = time.time()
    while len(contourCopy) > 0:
        timeDelta = time.time() - startTime
        if timeDelta >= timeout:
            show_Message('......Encountered timeout error while sorting the contour coordinates.')
            break
        window = contourImage[xRight:xRight+2, yRight:yRight+2]
        nextWindow, nextContourPixel = orientation(window)
        if len(nextContourPixel) > 0:
            for pixel in range(len(nextContourPixel)):
                xPos, yPos = xRight + nextContourPixel[pixel][0], yRight + nextContourPixel[pixel][1]
                arrayPosition = np.where((orderedContour == [xPos, yPos]).all(axis=1))[0]
                if len(arrayPosition) == 0:
                    index = find_index_of_coordinates([xPos, yPos], contourCopy, [0], 'index')
                    if len(index) != 0:
                        orderedContour = np.append(orderedContour, [[xPos, yPos]], axis=0)
                        contourCopy = np.delete(contourCopy, index[0], 0)
                        'orderedContour'
                else:
                    contourCopy = []
        if nextWindow == 'left':
            yRight = yRight - 1
        elif nextWindow == 'right':
            yRight = yRight + 1
        elif nextWindow == 'up':
            xRight = xRight- 1
        elif nextWindow == 'down':
            xRight = xRight + 1
    if len(orderedContour) != len(contour):
        clockwise = []
    else:
        clockwise = np.append([orderedContour[0]], orderedContour[-1:0:-1], axis=0)
        clockwise = clockwise.astype('int')
    return(clockwise)

def find_rightmost_point(contour):
    """
    return the rightmost point of a list of coordinates
    """
    index = np.where(contour[:, 1] == np.max(contour[:, 1]))[0]
    return(contour[index[0]][0], contour[index[0]][1])

def orientation(window):
    """
    define the direction of the shift for the next window according to the marching square algorithm
    """
    orient=''
    nextContourPixel = []
    if np.sum(window > 0) == 0:
        orient = 'right'
    elif np.sum(window > 0) == 1:
        if window[0, 0] != 0:
            orient = 'up'
        elif window[0, 1] != 0:
            orient = 'right'
        elif window[1, 0] != 0:
            orient = 'left'
        elif window[1, 1] != 0:
            orient = 'down'
    elif np.sum(window > 0) == 2:
        if window[0, 1] != 0 and window[1, 1] != 0:
            orient = 'down'
            nextContourPixel = [[1, 1]]
        elif window[0, 0] != 0 and window[0, 1] != 0:
            orient = 'right'
            nextContourPixel = [[0, 1]]
        elif window[0, 0] != 0 and window[1, 0] != 0:
            orient = 'up'
            nextContourPixel = [[0, 0]]
        elif window[1, 0] != 0 and window[1, 1] != 0:
            orient = 'left'
            nextContourPixel = [[1, 0]]
        elif window[0, 0] != 0 and window[1, 1] != 0:
            orient = 'up'
            nextContourPixel = [[0, 0]]
        elif window[0, 1] != 0 and window[1, 0] != 0:
            orient = 'left'
            nextContourPixel = [[1, 0]]
    elif np.sum(window > 0) == 3:
        if window[0, 0] != 0 and window[0, 1] != 0 and window[1, 1] != 0:
            orient = 'down'
            if window[0, 1] == 1:
                nextContourPixel = [[0, 1], [1, 1]]
            else:
                nextContourPixel = [[1, 1]]
        elif window[0, 0] != 0 and window[1, 0] != 0 and window[1, 1] != 0:
            orient = 'up'
            if window[1, 0] == 1:
                nextContourPixel = [[1, 0], [0, 0]]
            else:
                nextContourPixel = [[0, 0]]
        elif window[0, 1] != 0 and window[1, 0] != 0 and window[1, 1] != 0:
            orient = 'left'
            if window[1, 1] == 1:
                nextContourPixel = [[1, 1], [1, 0]]
            else:
                nextContourPixel = [[1, 0]]
        elif window[0, 0] != 0 and window[0, 1] != 0 and window[1, 0] != 0:
            orient = 'right'
            if window[0, 0] == 1:
                nextContourPixel = [[0, 0], [0, 1]]
            else:
                nextContourPixel = [[0, 1]]
    else:
        print('Error: too many pixels in window.')
    return(orient, nextContourPixel)

def find_index_of_coordinates(point, array, radius, output):
    """
    find position of point coordinates around radius in an array
    """
    foundPositions = []
    combinations = list(itertools.product(radius, repeat=2))
    for xRadius, yRadius in combinations:
        w = np.where((point[0]+xRadius == array[:, 0]) & (point[1]+yRadius == array[:, 1]))[0]
        if len(w) > 0:
            if output == 'index':
                foundPositions.append(w[0])
            else:
                foundPositions.append([xRadius, yRadius])
    return(foundPositions)

def calculate_pixel_distance(resolution):
    """
    calculate the optimal pixel distance between nodes along the contour from the image resolution
    """
    pixelDistance = int(np.round(1 / (resolution * 0.65)))
    return(pixelDistance)

def interpolate_contour_pixels(cellContour, pixelDistance):
    """
    determine all cell contour pixels which will be assigned as nodes according to the optimal pixel distance
    """
    pixelsOnContour = {}
    contourLength = len(cellContour)
    contourIndices = np.round(np.linspace(0, contourLength - pixelDistance, int((contourLength - pixelDistance) / pixelDistance))).astype('int')
    pixels = np.asarray(cellContour[contourIndices])
    for idx in range(len(pixels)):
        pixelsOnContour[idx] = (pixels[idx][0], pixels[idx][1])
    return(pixelsOnContour)

def invert(image):
    """
    invert image
    """
    if image.dtype == 'bool':
        return ~image
    else:
        return dtype.dtype_limits(image, clip_negative=False)[1] - image

def find_local_extrema(array):
    """
    find local minima and maxima in array
    """
    reverseArray = array[::-1]
    array = np.append(array, array[:1])
    reverseArray = np.append(reverseArray, reverseArray[:1])
    signsArray = calculate_consecutive_difference(array)
    signsReverseArray = calculate_consecutive_difference(reverseArray)[::-1]
    neckIndices = []
    lobeIndices = []
    for idx, sign in enumerate(signsArray):
        if sign == '-' and signsReverseArray[idx] == sign:
            lobeIndices.append(idx)
        elif sign == '+' and signsReverseArray[idx] == sign:
            neckIndices.append(idx)
    return(lobeIndices, neckIndices)

def calculate_consecutive_difference(sequence):
    """
    calculate the difference of consecutive elements in an array
    """
    difference = [elem1 - elem2 for elem1, elem2 in zip(sequence[:-1], sequence[1:])]
    signedDifference = np.sign(difference)
    signedSequence = convert_to_sign(signedDifference)
    return(signedSequence)

def convert_to_sign(sequence):
    """
    convert signed numbers into + and - signs
    """
    signs = []
    for idx in range(len(sequence)):
        if sequence[idx] > 0:
            signs.extend('+')
        elif sequence[idx] < 0:
            signs.extend('-')
        else:
            signs.extend('0')
    return(''.join(signs))

def get_key_from_value(dictionary, value):
    """
    get the dictionary key of a specified value
    """
    for key, val in dictionary.items():
        if val == value:
            return(key)




if __name__ == "__main__":

    selectedImage = r"C:\Users\user\Desktop\OrgSegNet\user_2\Nucl/Nucl__.tif"
    resolution = 30
    outputFolder = r"C:\Users\user\Desktop\OrgSegNet\user_2\Nucl/"

    fileList = None
    VG = VisGraphOther(selectedImage=selectedImage, resolution=resolution, outputFolder=outputFolder, inputType= "image", fileList=fileList)

