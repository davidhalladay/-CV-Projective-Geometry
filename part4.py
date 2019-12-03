import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt
import os

"""
# reference here :
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
# Author :
    Wan-Cyuan Fan
"""
# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # A = np.zeros((2*N, 8))
	# if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    for i in range(0,N):
        tmp_1 = [u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0]*v[i][0], -u[i][1]*v[i][0], -v[i][0]]
        tmp_2 = [0, 0, 0, u[i][0], u[i][1], 1, -u[i][0]*v[i][1], -u[i][1]*v[i][1], -v[i][1]]
        A[2*i] = tmp_1
        A[2*i+1] = tmp_2
    U, sigma, Vh = np.linalg.svd(A.transpose() @ A, full_matrices=False)

    H = Vh[-1].reshape(3,3) # U[:, 8].reshape(3,3)
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    h_c, w_c, ch = canvas.shape
    # print(h_c, w_c) # 1080 1920
    # TODO: some magic
    img_corners = np.array([[0,0],[0,w],[h,0],[h,w]])
    H = solve_homography(img_corners, corners)
    X = np.array([[i, j, 1] for i in range(h) for j in range(w)])
    Y = X @ H.transpose()
    Y = (Y / Y[:,2].reshape(-1,1)).astype(np.int)
    Y_x = Y[:,0].reshape(h,w)
    Y_y = Y[:,1].reshape(h,w)

    if Y_x.max() > w_c or Y_y.max() > h_c or Y_x.min() < 0 or Y_y.min() < 0:
        print("Warning! some pixels are out of image. You may use auto_crop in main function.")
        for i_x in range(h):
            for i_y in range(w):
                if not (Y_x[i_x,i_y] >= w_c or Y_y[i_x,i_y] >= h_c):
                    canvas[[Y_y[i_x,i_y]], [Y_x[i_x,i_y]]] = img[i_x,i_y]

    else:
        canvas[Y_y, Y_x] = img

def img_matcher(queryImage, trainImage, analyze=False):
    """
    args
        queryImage: image you want to query
        trainImage: target image contain query image
    return
        corners: four corners of the query image in the trainImage
            -> [[lefttop],[righttop],[leftbottom],[rightbottom]]
    """
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainImage, None)
    if analyze:
        gray= cv2.cvtColor(queryImage,cv2.COLOR_BGR2GRAY)
        img=cv2.drawKeypoints(gray,kp1,gray)
        plt.imsave(os.path.join('output','kp_queryImage.png'),img)
        gray= cv2.cvtColor(trainImage,cv2.COLOR_BGR2GRAY)
        img=cv2.drawKeypoints(gray,kp2,gray)
        plt.imsave(os.path.join('output','kp_trainImage.png'),img)

    # FLANN ->  optimized for fast nearest neighbor search in large
    #           datasets and for high dimensional features
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if analyze:
        print('Total matched points : ',len(matches))
        print('Matched points after Lowe\'s ratio test: ',len(good))

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        if len(queryImage.shape) == 2:
            h, w = queryImage.shape
        if len(queryImage.shape) == 3:
            h, w, ch = queryImage.shape

        pts = np.float32([ [0,0],[0,w],[h,0],[h,w]]).reshape(-1,1,2)
        if analyze:
            pts = np.float32([ [0,0],[0,w],[h,w],[h,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        if analyze:
            trainImage = cv2.polylines(trainImage,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            plt.imsave(os.path.join('output','mask.png'),trainImage)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    if analyze:
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
        img3 = cv2.drawMatches(queryImage,kp1,trainImage,kp2,good,None,**draw_params)
        plt.imsave(os.path.join('output','drawMatches.png'), img3)
    dst_new = np.int32(dst).reshape(4,2)[[0,2,1,3]]
    if analyze:
        exit()
    return dst_new

def main(ref_image,template,video,auto_crop=False):
    scale_size = 1
    scale_size_ref = 3
    ref_image = cv2.imread(ref_image)  ## load gray if you need.
    template = cv2.imread(template)  ## load gray if you need.
    if auto_crop :
        template = template[35:375,35:375,:]
    ref_image = cv2.resize(ref_image,(int(ref_image.shape[1]/scale_size_ref),int(ref_image.shape[0]/scale_size_ref)))
    template = cv2.resize(template,(int(template.shape[1]/scale_size),int(template.shape[0]/scale_size)))

    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_h, film_w = int(film_h/scale_size), int(film_w/scale_size)
    film_fps = video.get(cv2.CAP_PROP_FPS)
    film_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))
    i = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}/{}'.format(i,film_num))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            ## TODO: homography transform, feature detection, ransanc, etc.
            frame = cv2.resize(frame,(int(frame.shape[1]/scale_size),int(frame.shape[0]/scale_size)))
            corners = img_matcher(template, frame)
            transform(ref_image, frame, corners)
            videowriter.write(frame)
            i += 1
        else:
            break

    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)
