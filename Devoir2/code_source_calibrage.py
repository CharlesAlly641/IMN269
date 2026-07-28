# https://github.com/a-pijpaert/python_stereo_camera_calibrate/blob/main/MRE_calibration.py
import cv2
import numpy as np
import glob

rows = 6  # number of checkerboard rows.
columns = 9  # number of checkerboard columns.
world_scaling = 35.0  # change this to the real world square size. Or not.
_show = True


def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv2.imread(imname, 1)
        images.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv2 can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            if _show:
                # draw chessboard corners on frame
                cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)

                # resize frame to fit to screen
                res_frame = cv2.resize(frame, (1080, 720))
                cv2.imshow('img', res_frame)
                k = cv2.waitKey(100)

            # append corner locations to imgpoints
            objpoints.append(objp)
            imgpoints.append(corners)

    # perform camera calibration
    # ********* Ajout du flag pour forcer les coeffs de distorsion à 0 *********
    flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None, flags=flags)
    print(f'rmse single camera ({images_folder}): {ret}')

    return mtx, dist, rvecs, tvecs, ret


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_1, frames_2):
    # read the synched frames
    c1_images_names = glob.glob(frames_1)
    c2_images_names = glob.glob(frames_2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv2.imread(im1, 1)
        c1_images.append(_im)

        _im = cv2.imread(im2, 1)
        c2_images.append(_im)

    # criteria for stereo calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    count = 0
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            if count == 0:
                corner_point = [corners1[0], corners2[1]]

            if _show:
                cv2.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                res_frame = cv2.resize(frame1, (1080, 720))
                cv2.imshow('img', res_frame)

                cv2.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                res_frame = cv2.resize(frame2, (1080, 720))
                cv2.imshow('img2', res_frame)
                k = cv2.waitKey(100)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            count += 1

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

    # stereo calibrate system
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                  dist1,
                                                                  mtx2, dist2, (width, height), criteria=criteria,
                                                                  flags=stereocalibration_flags)

    print(f"rmse stereo: {ret}")

    print_stereo_results(mtx1, mtx2, R, T, E, F, ret)

    return R, T, corner_point, ret, E, F

# ********* Ajout du print des matrices *********
def print_stereo_results(mtx1, mtx2, R, T, E, F, ret_stereo):
    """Affiche les résultats du calibrage stéréo"""

    # Erreur de reprojection
    print(f"\nErreur de reprojection stéréo (RMS): {ret_stereo:.4f} pixels")

    # Matrices intrinsèques
    print("\nMatrice intrinsèque (Caméra gauche):")
    print(mtx1)
    print("\nMatrice intrinsèque (Caméra droite):")
    print(mtx2)

    # Matrice fondamentale
    print("\nMatrice fondamentale F:")
    print(F)

    # Matrice essentielle
    print("\nMatrice essentielle E:")
    print(E)

    # Matrice de rotation
    print("\nMatrice de rotation R (caméra droite par rapport à gauche):")
    print(R)

    # Vecteur de translation
    print("\nVecteur de translation T (caméra droite par rapport à gauche, en mm):")
    print(T.ravel())

if __name__ == "__main__":
    mtx1, dist1, rvecs1, tvecs1, ret1 = calibrate_camera(images_folder='calib_images/calib_images_vendredi_partiel/left_*.jpg')
    mtx2, dist2, rvecs2, tvecs2, ret2 = calibrate_camera(images_folder='calib_images/calib_images_vendredi_partiel/right_*.jpg')

    R, T, corner_point, ret_stereo, E, F = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'calib_images/calib_images_vendredi_partiel/left_*.jpg',
                                          'calib_images/calib_images_vendredi_partiel/right_*.jpg')

    #mtx1, dist1, rvecs1, tvecs1, ret1 = calibrate_camera(images_folder='calib_images/calib_images_udes/left_*.jpg')
    #mtx2, dist2, rvecs2, tvecs2, ret2 = calibrate_camera(images_folder='calib_images/calib_images_udes/right_*.jpg')

    #R, T, corner_point, ret_stereo, E, F = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'calib_images/calib_images_udes/left_*.jpg',
    #                                      'calib_images/calib_images_udes/right_*.jpg')

    transformation_matrix = np.empty((4, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T.T[0]
    transformation_matrix[3, :] = [0, 0, 0, 1]

    location_cam2 = np.dot(transformation_matrix, [[0], [0], [0], [1]])
