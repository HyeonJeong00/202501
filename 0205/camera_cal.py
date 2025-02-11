import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (4, 8)       # input image chessboard 내부 corner 수(row, col)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)      # 종료 조건(termination criteria)
# corner 위치를 정확히 추정하기 위한 종료 기준, (반복 횟수 최대 30) 또는 (정확도 0.001에 도달시 종료)

objpoints = []      # chessboard 3D world 좌표
imgpoints = []      # 3D image 좌표

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)              # Defining the world coordinates for 3D points
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)    # X, Y 좌표 생성, Z = 0으로.    다차원 격자(grid) 생성

images = glob.glob('./images/*.jpeg')


for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 흑백 image로 변환환
    ret, corners = cv2.findChessboardCorners(       # chessboard corner 감지 여부, corner 2D 좌표
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # 이미지의 밝기와 명암이 일정하지 않을 경우, Adaptive Thresholding를 적용하여 체스보드 패턴을 감지
        # 체스보드 감지 알고리즘의 속도를 향상시키기 위해 간단한 체크를 먼저 수행하여 체스보드가 존재하지 않는 이미지를 빠르게 걸러냄
        # 입력 이미지를 정규화(normalization)하여 명암 대비를 증가시킨 뒤, 코너 감지를 수행
    if ret:
        objpoints.append(objp)      # 3D word coordinate 좌표에 추가
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    # corner 좌표를 subpixel 수준으로 정밀화
        imgpoints.append(corners2)  # 정밀화된 2D pixel 좌표 추가
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        output_image = f'output/visualized_img_{idx}.jpg'
        cv2.imwrite(output_image, img)

        h, w = img.shape[:2]
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners2], gray.shape[::-1], None, None)  # Zhang’s Method

        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)              # rotation vector --> rotation matrix
        extrinsic_matrix = np.hstack((R, tvec)) # extrinsic matrix = [R | t]            가로 방향(horizontal)으로 연결(합치기)
        c = np.dot(R.T, tvec) * (-1)
        rotation_angle = np.linalg.norm(c)
        axis = c / rotation_angle

        output_file = f"output/calibration_result_{idx}.txt"
        with open(output_file, "w") as f:
            f.write(f"Results for image: {fname}\n\n")
            f.write("Camera matrix (Intrinsic Parameters):\n")
            f.write(f"{mtx}\n\n")
            f.write("Camera matrix (Extrinsic Parameters):\n")
            f.write(f"{extrinsic_matrix}\n\n")
        print(f"Processed and saved results for image: {fname}")