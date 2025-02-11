import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Step 1: 캘리브레이션 이미지 로드
image_paths = glob.glob("calibration_images/*.jpg")  # 이미지 경로를 glob으로 가져옴
image_paths = image_paths[:5]  # 처음 5개의 이미지만 사용

# Step 2: 체스보드 패턴 감지
checkerboard_size = (7, 7)  # 체스보드 패턴의 내부 코너 개수 (열, 행)
square_size = 25  # 체스보드 한 칸의 크기 (밀리미터)

# 월드 좌표 생성
world_points = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
world_points[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# 이미지 좌표와 월드 좌표 저장용 리스트
object_points = []  # 월드 좌표
image_points = []  # 이미지 좌표

# 이미지에서 체스보드 코너 감지
for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 체스보드 코너 감지
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret:
        object_points.append(world_points)
        # 서브픽셀 정확도로 코너 개선
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        image_points.append(corners_subpix)
        
        # 코너 시각화
        cv2.drawChessboardCorners(img, checkerboard_size, corners_subpix, ret)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

# Step 3: 카메라 캘리브레이션
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)

# Step 4: Extrinsic Parameters 시각화
# 첫 번째 이미지의 외부 파라미터로 예제 시각화
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
translation_vector = tvecs[0]

print("\nRotation Matrix (First Image):")
print(rotation_matrix)
print("\nTranslation Vector (First Image):")
print(translation_vector)

# Step 5: 결과 저장 및 시각화
# 예제 이미지에서 왜곡 제거
for image_path in image_paths:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    plt.show()
