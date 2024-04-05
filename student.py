import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # print('Randomly setting matrix entries as a placeholder')
    # M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
    #               [0.6750, 0.3152, 0.1136, 0.0480],
    #               [0.1020, 0.1725, 0.7244, 0.9932]])
    N = points2d.shape[0]
    A = np.zeros([2*N,11])
    y = np.zeros([2*N,1])
    for i in range(0,N):
        u = points2d[i,0]
        v = points2d[i,1]

        X = points3d[i,0]
        Y = points3d[i,1]
        Z = points3d[i,2]

        A[2*i:2*(i+1),:] = [[X,Y,Z,1,0,0,0,0,-X*u,-Y*u,-Z*u],[0,0,0,0,X,Y,Z,1,-X*v,-Y*v,-Z*v]]
        y[2*i,:] = u
        y[2*(i+1)-1,:] = v

    M = np.linalg.lstsq(A, y, rcond=None)[0]
    #Remember to try SVD, pseudoinverse is GUD
    M = np.concatenate((M,np.array([1],ndmin=2)))
    M = M.reshape([3,4])

    return M


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    # T = np.eye(3)
    
    N = points.shape[0]
    c = np.sum(points,axis = 0)/N
    s = 1/np.std( points-c )

    scale = np.zeros((3,3))
    scale[0,0] = s 
    scale[1,1] = s
    scale [2,2] = 1

    offset = np.eye(3)
    offset[0,2] = -c[0]
    offset[1,2] = -c[1] 

    T = scale @ offset

    for i in range(0,N):
        dummy_pts = T @ np.concatenate((points[i,:],np.array([1])))
        points[i,:] = dummy_pts[0:2]

    return points, T


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################

    # This is an intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    N = points1.shape[0]
    # n_ptsA = points2
    # n_ptsB = points1
    n_ptsB, TB = normalize_coordinates(points2)
    n_ptsA, TA = normalize_coordinates(points1)

    A = np.zeros((N,9))
    for i in range(0,N):
        uA = n_ptsA[i,0]
        vA = n_ptsA[i,1]
        uB = n_ptsB[i,0]
        vB = n_ptsB[i,1]

        A[i,:] = [uA*uB, vA*uB, uB, uA*vB, vA*vB, vB,uA, vA, 1]
    
    #Using Berkeley slides pseudocodes
    #https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture9-4-computing-the-fundamental-matrix.pdf

    [u,s,vh] = np.linalg.svd(A)
    f = vh[-1].reshape((3,3))

    #enforce rank 2
    [u,s,vh] = np.linalg.svd(f)
    s[2] = 0
    F_matrix = u @ np.diag(s) @ vh    

    F_matrix = np.transpose(TB) @ F_matrix @ TA
    return F_matrix


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = matches1[0:29, :]
    inliers_b = matches2[0:29, :]

    #Using Berkeley slides pseudocodes
    #https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture9-4-computing-the-fundamental-matrix.pdf


    N = matches1.shape[0]
    confidence = 0
    for i in range(0,num_iters):
        sampling = np.random.randint(low = 0,high = N, size = 8)
        sampling_1 = matches1[sampling,:]
        sampling_2 = matches2[sampling,:]
        dummyF = estimate_fundamental_matrix(sampling_1,sampling_2)
        # dummyF, _ = cv2.findFundamentalMat(sampling_1, sampling_2, cv2.FM_8POINT, 1e10, 0, 1)
        while type(dummyF) == type(None):
            sampling = np.random.randint(low = 0,high = N, size = 8)
            sampling_1 = matches1[sampling,:]
            sampling_2 = matches2[sampling,:]
            # dummyF, _ = cv2.findFundamentalMat(sampling_1, sampling_2, cv2.FM_8POINT, 1e10, 0, 1)
            dummyF = estimate_fundamental_matrix(sampling_1,sampling_2)


        dummy1 = np.zeros((N,2))
        dummy2 = np.zeros((N,2))
        threshold = 0.0001

        for j in range(0,N):
            if np.abs( np.concatenate((matches2[j,:],[1])) @ dummyF @ np.transpose(np.concatenate((matches1[j,:],[1]))) ) < threshold:
                dummy1[j,:] = matches1[j,:]
                dummy2[j,:] = matches2[j,:]
        
        dummy1 = dummy1[~np.all(dummy1 == 0,axis = 1)]
        dummy2 = dummy2[~np.all(dummy2 == 0,axis = 1)]

        n = dummy1.shape[0]

        if n/N > confidence:
            confidence = n/N
            best_Fmatrix = dummyF
            inliers_a = dummy1
            inliers_b = dummy2
    return best_Fmatrix, inliers_a, inliers_b


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] list of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    
    ########################
    # TODO: Your code here #
    ########################
    N = points1.shape[0]
    points3d = np.zeros((N,3)) 
    M = np.zeros((4,3))
    for i in range(0,N):
        u1 = points1[i,0]
        v1 = points1[i,1]
        u2 = points2[i,0]
        v2 = points2[i,1]
        vec2d = np.array([u1-M1[0,3],v1-M1[1,3],u2-M2[0,3],v2-M2[1,3]])

        
        M[0,0] = M1[0,0] - M1[2,0]*u1
        M[0,1] = M1[0,1] - M1[2,1]*u1
        M[0,2] = M1[0,2] - M1[2,2]*u1
 

        M[1,0] = M1[1,0] - M1[2,0]*v1
        M[1,1] = M1[1,1] - M1[2,1]*v1
        M[1,2] = M1[1,2] - M1[2,2]*v1

        M[2,0] = M2[0,0] - M2[2,0]*u2
        M[2,1] = M2[0,1] - M2[2,1]*u2
        M[2,2] = M2[0,2] - M2[2,2]*u2


        M[3,0] = M2[1,0] - M2[2,0]*v2
        M[3,1] = M2[1,1] - M2[2,1]*v2
        M[3,2] = M2[1,2] - M2[2,2]*v2


        # vec3d = np.linalg.lstsq(M,vec2d,rcond=None)[0]
        vec3d = np.linalg.lstsq(M,vec2d)[0]
        print(vec3d)
        # if np.abs(vec3d[3]-1)<0.001:
        points3d[i,:] = vec3d[0:3]

    return points3d.tolist()
