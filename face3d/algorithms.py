"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

from utils import intertial_axis, max_xy, min_xy, find_peak_start, find_peak_stop

from scipy.ndimage import filters, interpolation
from scipy.interpolate import griddata
from scipy import stats, signal, optimize, interpolate
from sklearn import metrics

import numpy
import scipy
import math
import pylab
import threading

def process(face, N, K):
    """ Apply the selected algorithms on a given face """

    # Normalize face
    smooth(face)
    repair(face)
    crop(face)
    zoom(face)
    key_points(face)
    rotate(face)
    key_points(face)
    fit(face)

    # Extract features
    features_histogram(face, N, K)

def smooth(face):
    """ Smooth data. Removes peaks """

    # Helper method
    def smooth_axis(axis):
        face.abs_file.data[axis] = filters.median_filter(face.abs_file.data[axis], size=4)
        face.abs_file.data[axis] = filters.gaussian_filter(face.abs_file.data[axis], sigma=1, mode='nearest')

    # Smooth it
    smooth_axis('X')
    smooth_axis('Y')
    smooth_axis('Z')

def repair(face):
    """ Fill missing data by interpolating """

    # Helper method
    def interpolate_axis(axis):
        A = face.abs_file.data[axis]

        # Calculate parameters
        mask = numpy.isfinite(A)
        points = mask.nonzero()
        values = A[points]
        grid_coords = numpy.meshgrid(numpy.arange(0, len(A[:, 0]), 1), numpy.arange(0, len(A[0, :]), 1))

        # Apply interpolation
        face.abs_file.data[axis] = griddata(points, values, grid_coords, method='linear').T

    interpolate_axis('X')
    interpolate_axis('Y')
    interpolate_axis('Z')

def key_points(face, 
    d_nose_x1=30, d_nose_x2=5, d_nose_y=5,
    d_lip_y1=25, d_lip_y2=70, d_lip_y3=4, d_lip_x1=50,
    d_chin_x=3, d_chin_y1=50, d_chin_y2=75,
    d_eye_x=2, d_eye_y=50):

    """
    Rotate and zoom the face to create a full frame face. This is based on the
    fact that the nose is the highest point of the picture
    """

    # We apply surfature to calculate the first and second derivates
    K, H, Pmax, Pmin = surfature(face)

    # Remove all key points 
    face.key_points.clear()

    #
    # Nose
    #
    nose_x, nose_y = max_xy(face.Z)
    face.key_points["nose"] = (nose_x, nose_y)

    #
    # Nose left and right
    #
    nose_left = Pmin[(nose_y - d_nose_y):(nose_y + d_nose_y), (nose_x - d_nose_x1):(nose_x - d_nose_x2)]
    nose_right = Pmin[(nose_y - d_nose_y):(nose_y + d_nose_y), (nose_x + d_nose_x2):(nose_x + d_nose_x1)]

    nose_left_x, nose_left_y = min_xy(nose_left, offset_x=(nose_x - d_nose_x1), offset_y=(nose_y - d_nose_y))
    nose_right_x, nose_right_y = min_xy(nose_right, offset_x=(nose_x + d_nose_x2), offset_y=(nose_y - d_nose_y))

    face.key_points["nose_left"] = (nose_left_x, nose_left_y)
    face.key_points["nose_right"] = (nose_right_x, nose_right_y)

    # 
    # Upper, lower, left right lip
    #
    lip_y = numpy.nanargmax(Pmax[(nose_y + d_lip_y1):(nose_y + d_lip_y2), nose_x]) + (nose_y + d_lip_y1)
    lip_left = Pmax[(lip_y - d_lip_y3):(lip_y + d_lip_y3), (nose_x - d_lip_x1):nose_x]
    lip_right = Pmax[(lip_y - d_lip_y3):(lip_y + d_lip_y3), nose_x:(nose_x + d_lip_x1)]

    lip_left_x = find_peak_start(numpy.sum(lip_left, axis=0)) + (nose_x - d_lip_x1)
    lip_left_y = numpy.nanargmax(Pmax[(lip_y - d_lip_y3):(lip_y + d_lip_y3), lip_left_x]) + (lip_y - d_lip_y3)

    lip_right_x = find_peak_stop(numpy.sum(lip_right, axis=0)) + nose_x
    lip_right_y = numpy.nanargmax(Pmax[(lip_y - d_lip_y3):(lip_y + d_lip_y3), lip_right_x]) + (lip_y - d_lip_y3)

    face.key_points['lip'] = (nose_x, lip_y)
    face.key_points['lip_left'] = (lip_left_x, lip_left_y)
    face.key_points['lip_right'] = (lip_right_x, lip_right_y)

    #
    # Chin
    #
    chin = numpy.gradient(signal.bspline(face.Z[(lip_y + d_chin_y1):, nose_x], 25))
    chin_x, chin_y = nose_x, numpy.nanargmin(chin) + (lip_y + d_chin_y1)

    face.key_points["chin"] = (chin_x, chin_y)

    # 
    # Eyes
    #
    eye_left = Pmax[d_eye_y:nose_left_y - d_eye_y, nose_left_x - d_eye_x:nose_left_x + d_eye_x]
    eye_right = Pmax[d_eye_y:nose_right_y - d_eye_y, nose_right_x - d_eye_x:nose_right_x + d_eye_x]

    eye_left_x, eye_left_y = max_xy(eye_left, nose_left_x - d_eye_x, d_eye_y)
    eye_right_x, eye_right_y = max_xy(eye_right, nose_right_x - d_eye_x, d_eye_y)

    face.key_points["eye_left"] = (eye_left_x, eye_left_y)
    face.key_points["eye_right"] = (eye_right_x, eye_right_y)

    #
    # Nose face border
    #
    nose_line = numpy.gradient(face.Z[nose_y, :])
    border_nose_left_x, border_nose_left_y = numpy.nanargmax(nose_line[:lip_left_x - 10]), nose_y
    border_nose_right_x, border_nose_right_y = numpy.nanargmin(nose_line[lip_right_x + 10:]) + lip_right_x + 10, nose_y

    face.key_points["border_nose_left"] = (border_nose_left_x, border_nose_left_y)
    face.key_points["border_nose_right"] = (border_nose_right_x, border_nose_right_y)

    #
    # Lip face border
    #
    lip_line = numpy.gradient(face.Z[lip_y, :])
    border_lip_left_x, border_lip_left_y = numpy.nanargmax(lip_line[:lip_left_x - 10]), lip_y
    border_lip_right_x, border_lip_right_y = numpy.nanargmin(lip_line[lip_right_x + 10:]) + lip_right_x + 10, lip_y

    face.key_points["border_lip_left"] = (border_lip_left_x, border_lip_left_y)
    face.key_points["border_lip_right"] = (border_lip_right_x, border_lip_right_y)

    #
    # Forehead border
    #
    forehead_line = numpy.gradient(face.Z[nose_y - (chin_y - nose_y), :])
    border_forehead_left_x, border_forehead_left_y = numpy.nanargmax(forehead_line[:lip_left_x - 10]), nose_y - (chin_y - nose_y)
    border_forehead_right_x, border_forehead_right_y = numpy.nanargmin(forehead_line[lip_right_x + 10:]) + lip_right_x + 10, nose_y - (chin_y - nose_y)

    face.key_points["border_forehead_left"] = (border_forehead_left_x, border_forehead_left_y)
    face.key_points["border_forehead_right"] = (border_forehead_right_x, border_forehead_right_y)

def rotate(face):
    """ Rotate the face by taking the mean slope of the nose and lip """

    # Nose rotation
    d_nose_y = face.key_points["nose_left"][1] - face.key_points["nose_right"][1]
    d_nose_x = face.key_points["nose_right"][0] - face.key_points["nose_left"][0]
    degrees_nose = math.degrees(math.atan2(d_nose_y, d_nose_x))

    # Lip rotation
    d_lip_y = face.key_points["lip_left"][1] - face.key_points["lip_right"][1]
    d_lip_x = face.key_points["lip_right"][0] - face.key_points["lip_left"][0]
    degrees_lip = math.degrees(math.atan2(d_lip_y, d_lip_x))

    # Calculate average rotation and rotate
    degrees = (degrees_nose + degrees_lip) / 2
    face.abs_file.data['X'] = interpolation.rotate(face.abs_file.data['X'], degrees, mode='nearest', prefilter=False, reshape=False)
    face.abs_file.data['Y'] = interpolation.rotate(face.abs_file.data['Y'], degrees, mode='nearest', prefilter=False, reshape=False)
    face.abs_file.data['Z'] = interpolation.rotate(face.abs_file.data['Z'], degrees, mode='nearest', prefilter=False, reshape=False)

def zoom(face):
    """ Move everything such that nose is at depth 0 """

    # Correct the nose tip to be at 0
    point = max_xy(face.Z)

    face.abs_file.data['X'] = face.abs_file.data['X'] + abs(face.X[point])
    face.abs_file.data['Y'] = face.abs_file.data['Y'] + abs(face.Y[point])
    face.abs_file.data['Z'] = face.abs_file.data['Z'] + abs(face.Z[point])

def fit(face):
    """ Crops the image to face width and face height """

    chin_x, chin_y = face.key_points["chin"]
    nose_x, nose_y = face.key_points["nose"]

    border_lip_left_x, border_lip_left_y = face.key_points["border_lip_left"]
    border_lip_right_x, border_lip_right_y = face.key_points["border_lip_right"]

    border_forehead_left_x, border_forehead_left_y = face.key_points["border_forehead_left"]
    border_forehead_right_x, border_forehead_right_y = face.key_points["border_forehead_right"]

    golden_ratio = 1.61803399
    face_height = (chin_y - nose_y) + (chin_y - nose_y) * golden_ratio
    #face_width = stats.nanmean(numpy.array([border_forehead_right_x, border_lip_right_x])) - stats.nanmean(numpy.array([border_forehead_left_x + border_lip_left_x]))
    face_width = face_height / golden_ratio

    # Overscan
    face_height = face_height * 0.90
    face_width = face_width * 0.95

    # Fit region
    face.center_at(nose_x, chin_y - (face_height / 2.0), face_width / 2.0, face_height / 2.0)

def crop(face):
    """ 
    Crop the image to remove as much unneeded information as possible. This
    works by applying PCA to find the torso and then find the nose.

    The result is a view that is centered at the nose.
    """

    # Reset image first to make sure we take all of the image
    face.reset()

    # Calculate the position of the image
    masked_z = numpy.ma.masked_array(face.Z, numpy.isnan(face.Z))
    x, y, covariance = intertial_axis(masked_z)

    # Center at the point
    overscan_x = face.width * 0.25
    overscan_y = face.height * 0.25
    face.center_at(x, y, overscan_x, overscan_y)

    # Calculate max Z-value x and y
    x, y = max_xy(face.Z)

    # Set view to center of nose
    face.center_at(x, y, 240 / 2.0, 320 / 2.0)

def features_histogram(face, N=67, K=12):
    """
    From 'A 3D Face Recognition Algorithm Using Histogram-based Features'
    """

    # It only works with non-nan values
    masked_z = numpy.ma.masked_array(face.Z, numpy.isnan(face.Z))
    results = []

    # Split the complete Z matrix into N smaller Zi
    for i, Zi in zip(range(N), numpy.array_split(masked_z, N)):
        result, temp = numpy.histogram(Zi, bins=K, range=(Zi.min(), Zi.max()), density=False)
        results.append(result)

    # Convert back to array
    face.features = ("histogram", numpy.array(results).reshape(-1), (N, K))

def distance_histogram_city_block(face1, face2):
    """ 
    Calculate the City Block distance of two histogram feature vectors 
    """

    def _func(U, V, U_V):
        return numpy.sum([ numpy.abs(Ui - Vi) for Ui, Vi in U_V ])

    return distance_histogram(face1, face2, _func)

def distance_histogram_euclidean(face1, face2):
    """ 
    Calculate the Euclidean distance of two histogram feature vectors 
    """

    def _func(U, V, U_V):
        return numpy.sqrt(numpy.sum([ numpy.power(Ui - Vi, 2) for Ui, Vi in U_V ]))

    return distance_histogram(face1, face2, _func)

def distance_histogram_correlation(face1, face2):
    """
    Calculate the Sample Correlation Coefficient of two histogram feature vectors
    """

    def _func(U, V, U_V):
        Umean = U.mean()
        Vmean = V.mean()
        Ustd = U.std()
        Vstd = V.std()

        samples = [ (Ui - Umean)*(Vi - Vmean) for Ui, Vi in U_V ]
        return numpy.sum(samples) / ((len(samples) - 1) * Ustd * Vstd)

    return distance_histogram(face1, face2, _func)

def distance_histogram(face1, face2, func):
    """ Base method for distance funcions """

    U = face1.features[1]
    V = face2.features[1]

    # Make sure both are same size
    if U.shape != V.shape:
        raise Exception("Feature vectors do not match size")

    # Calculate the distance
    return func(U, V, zip(U, V))

def similarity_matrix(faces, methods=None, normalizers=None, limit=None):
    """ 
    Calculate the similarity matrix for given set of faces with a given 
    set of methods and normalizers. For each method, a seperate thread will
    be spawned
    """

    # Set default methods
    if not methods:
        methods = [distance_histogram_euclidean, distance_histogram_city_block, distance_histogram_correlation]

    # Set default normalizers
    if not normalizers:
        normalizers = [score_normalization_min_max, score_normalization_min_max, False]

    # Create output array
    output = numpy.zeros(shape=(len(methods), len(faces), len(faces)))
    output[:] = numpy.nan

    # Precalculations
    count_faces = len(faces)
    count_methods = len(methods)
    count_faces_limited = count_faces if not limit else limit
    threads = []

    # Iterate each face
    def _func(index, method, normalizer):
        # Create similarity matrix
        for i in range(count_faces_limited):
            if i % 25 == 0:
                print "Method %d: %d/%d" % (index, i, count_faces)

            for j in range(i, count_faces):
                output[(index, i, j)] = method(faces[i], faces[j])

        # Normalize matrix
        if normalizer: 
            normalizer(output[index])

        # Print some info
        print "Finished similarity matrix for method %d" % index

    # Spawn the threads
    for i in range(count_methods):
        thread = threading.Thread(target=_func, args=(i, methods[i], normalizers[i]))
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads: 
        thread.join()

    # Done
    return output

def score_normalization_min_max(matrix):
    """ In place normalization to min-max scores """

    # Calculate min/max
    dmin = numpy.nanmin(matrix)
    dmax = numpy.nanmax(matrix)

    # In-place transformation
    matrix -= dmin
    matrix /= (dmax - dmin)

    # Convert to score
    numpy.subtract(1, matrix, matrix)

def calculate_roc_eer(matrix, person_ids):
    """ Calculate the ROC curve and estimate the EER """

    methods, _, _ = matrix.shape
    count = len(person_ids)
    result = [False, False, False]
    threads = []

    # Calculate ROC curve and EER for each method
    def _func(index):
        targets = []
        outputs = []

        for i in range(count):
            if i % 25 == 0:
                print "Method %d: %d/%d" % (index, i, count)

            for j in range(i, count):
                if person_ids[i] == person_ids[j]:
                    targets.append(1)
                else:
                    targets.append(0)

                outputs.append(matrix[index][i][j])

        # Calculate ROC curve
        tpr, fpr, _ = metrics.roc_curve(targets, outputs)

        # Create three function for solving
        f = interpolate.interp1d(tpr, fpr, bounds_error=False)
        g = lambda x: 1 - x
        
        # Estimate the EER -- the intersection of f(x) and g(x)
        for x in numpy.linspace(0, 1, 1000):
            # Skip boundaries as they are invalid for the interpolator
            if x == 0.0 or x == 1.0: 
                continue

            # Check intersection point
            if f(x) >= g(x):
                eer = x
                break

        # Append data to result list
        result[index] = ((tpr, fpr), eer)

        # Print some info
        print "Finished ROC and EER for method %d" % index

    
    # Spawn the threads
    for i in range(methods):
        thread = threading.Thread(target=_func, args=(i,))
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads: 
        thread.join()
        
    # Done
    return result

def surfature(face):
    """ 
    Calculate the surfatures of a given face. Based on a Matlab implementation
    http://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    """

    # First derivatives
    Xu, Xv = numpy.gradient(face.X)
    Yu, Yv = numpy.gradient(face.Y)
    Zu, Zv = numpy.gradient(face.Z)

    # Second derivates
    Xuu, Xuv = numpy.gradient(Xu)
    Yuu, Yuv = numpy.gradient(Yu)
    Zuu, Zuv = numpy.gradient(Zu)

    Xuv, Xvv = numpy.gradient(Xv)
    Yuv, Yvv = numpy.gradient(Yv)
    Zuv, Zvv = numpy.gradient(Zv)

    # Reshape to vector
    Xu = Xu.reshape(-1, 1)
    Yu = Yu.reshape(-1, 1)
    Zu = Zu.reshape(-1, 1)

    Xv = Xv.reshape(-1, 1)
    Yv = Yv.reshape(-1, 1)
    Zv = Zv.reshape(-1, 1)

    Xuu = Xuu.reshape(-1, 1)
    Yuu = Yuu.reshape(-1, 1)
    Zuu = Zuu.reshape(-1, 1)

    Xuv = Xuv.reshape(-1, 1)
    Yuv = Yuv.reshape(-1, 1)
    Zuv = Zuv.reshape(-1, 1)

    Xvv = Xvv.reshape(-1, 1)
    Yvv = Yvv.reshape(-1, 1)
    Zvv = Zvv.reshape(-1, 1)

    # Reshape data
    XYZu = numpy.concatenate((Xu, Yu, Zu), 1)
    XYZv = numpy.concatenate((Xv, Yv, Zv), 1)
    XYZuu = numpy.concatenate((Xuu, Yuu, Zuu), 1)
    XYZuv = numpy.concatenate((Xuv, Yuv, Zuv), 1)
    XYZvv = numpy.concatenate((Xvv, Yvv, Zvv), 1)

    # First fundamental coefficients
    E = numpy.sum(XYZu * XYZu, 1)
    F = numpy.sum(XYZu * XYZv, 1)
    G = numpy.sum(XYZv * XYZv, 1)
    
    m = numpy.cross(XYZu, XYZv)
    p = numpy.sqrt(numpy.sum(m * m, 1))
    n = numpy.divide(m, numpy.array([p, p, p]).T)

    # Second fundamental coefficients
    L = numpy.sum(XYZuu * n, 1)
    M = numpy.sum(XYZuv * n, 1)
    N = numpy.sum(XYZvv * n, 1)

    # Retrieve size
    s, t = face.Z.shape

    # Gaussian curvature
    K1 = numpy.multiply(L, N) - numpy.power(M, 2)
    K2 = numpy.multiply(E, G) - numpy.power(F, 2)
    K = numpy.divide(K1, K2).reshape(s, t)

    # Mean curvature
    H1 = numpy.multiply(E, N) + numpy.multiply(G, L) - numpy.multiply(numpy.multiply(2, F), M)
    H2 = numpy.multiply(2, numpy.multiply(E, G) - numpy.power(F, 2))
    H = numpy.divide(H1, H2).reshape(s, t)

    # Determine min and max curvatures
    Pmax = H + numpy.sqrt(numpy.power(H, 2) - K)
    Pmin = H - numpy.sqrt(numpy.power(H, 2) - K)

    # Done
    return K, H, Pmax, Pmin