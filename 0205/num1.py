import cv2
import numpy as np

# 이미지 로드
img1 = cv2.imread('/images/img_1.jpeg')
img2 = cv2.imread('/images/resized_img_2.jpg')

if img1 is None or img2 is None:
    raise ValueError("Error loading images. Check file paths.")

# 카메라 내부 파라미터 (예제 값)
K = np.array([[ 4.30355142e+02,  0,  3.01782522e+02],
              [ 0,  1.75925864e+03, -2.64554427e+02],
              [ 0,  0, 1]])

# ORB 특징점 검출기
orb = cv2.ORB_create(nfeatures=5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 특징점 검출 및 기술자 계산
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 특징점 매칭
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭된 특징점 개수 확인
if len(matches) < 5:
    raise ValueError("Not enough matches found! Need at least 5 matches.")

# 2D 특징점 좌표 추출
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

# Essential Matrix 계산 (검사 후 실행)
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# 포즈 복원 (R, t 추출)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

# 카메라 투영 행렬 설정
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

# 투영 행렬을 카메라 좌표계로 변환
P1 = K @ P1
P2 = K @ P2

# 3D 포인트 복원 (Triangulation)
points_4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3D = points_4D / points_4D[3]  # Homogeneous -> Cartesian 변환
points_3D = points_3D[:3, :].T

# 3D 시각화
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 점 플로팅
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o', s=5, c='r', alpha=0.5)

# 축 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
