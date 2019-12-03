import numpy as np
import cv2
import time


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
    # TODO: some magic
    img_corners = np.array([[0,0],[0,w],[h,0],[h,w]])
    H = solve_homography(img_corners, corners)
    X = np.array([[i, j, 1] for i in range(h) for j in range(w)])
    Y = X @ H.transpose()
    Y = (Y / Y[:,2].reshape(-1,1)).astype(np.int)
    Y_x = Y[:,0].reshape(h,w)
    Y_y = Y[:,1].reshape(h,w)
    canvas[Y_y, Y_x] = img

def transform_inv(img, canvas, img_corners):
    h, w, ch = canvas.shape
    # TODO: some magic
    canvas_corners = np.array([[0,0],[0,w],[h,0],[h,w]])
    H = solve_homography(img_corners, canvas_corners)
    H_i = np.linalg.inv(H)
    X = np.array([[i, j, 1] for i in range(h) for j in range(w)])
    Y = X @ H_i.transpose()
    Y = (Y / Y[:,2].reshape(-1,1)).astype(np.int)
    Y_x = Y[:,0].reshape(h,w)
    Y_y = Y[:,1].reshape(h,w)
    canvas[:] = img[Y_y, Y_x]

def main():
    # Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])

    # TODO: some magic
    transform(img1, canvas, canvas_corners1)
    transform(img2, canvas, canvas_corners2)
    transform(img3, canvas, canvas_corners3)
    transform(img4, canvas, canvas_corners4)
    transform(img5, canvas, canvas_corners5)

    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 2
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')
    # TODO: some magic
    img_corners = np.array([[1984,1244],[2040,1215],[2028,1396],[2081,1365]])
    canvas = np.zeros((480, 480, 3), dtype = np.int)
    transform_inv(img, canvas, img_corners)
    cv2.imwrite('part2.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    img_front_corners = np.array([[95,140],[630,140],[0,322],[725,322]])
    h, w, ch = img_front.shape
    canvas = np.zeros((h, w, 3), dtype = np.int)
    transform_inv(img_front, canvas, img_front_corners)
    cv2.imwrite('part3.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
