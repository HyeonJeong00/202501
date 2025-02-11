import numpy as np
import os
b = np.array([ 3.65053670e-05,  4.47678215e-06,  1.53516101e-07, -6.05825631e-03, -8.00147462e-04,  1.00000000e+00])

B11, B12, B22, B13, B23, B33 = b

# ✅ `det_B` 검증 (수치 오류 방지)
det_B = B11 * B22 - B12**2
if abs(det_B) < 1e-6:
    raise ValueError("Singular matrix detected: det(B) is too small. Check your Homographies.")

# ✅ 내부 파라미터 복원 (Zhang's method 공식)
v0 = (B12 * B13 - B11 * B23) / det_B
lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

if lambda_ <= 0:
    raise ValueError("Invalid lambda_: negative value encountered. Check Homographies.")

# ✅ 초점 거리 계산 (수치 오류 방지)
alpha = np.sqrt(lambda_ / np.abs(B11))  
beta  = np.sqrt(lambda_ * np.abs(B11) / np.abs(det_B))  
gamma = -B12 * alpha**2 * beta / lambda_
u0    = gamma * v0 / beta - B13 * alpha**2 / lambda_

# ✅ 내재 행렬(K) 생성
K_est = np.array([
    [alpha, gamma, u0],
    [0,     beta,  v0],
    [0,     0,     1]
])

print("\nEstimated camera intrinsic matrix K:")
print(K_est)



H1 = np.array([
 [0.3925646,  0.22040174, 0.14753924],
 [0.36214627, 0.72694601, 0.34030648],
 [0.00331412, 0.00268002, 0.00302485]
])
H2 = np.array([
    [0.63293221, 0.2398352,  0.25079213],
 [0.43410278, 0.4567363,  0.28607245],
 [0.00469417, 0.00149894, 0.00308952]
])
H3 = np.array([
   [0.57204421, 0.2566701,  0.24731568],
 [0.44825061, 0.50121322, 0.30582624],
 [0.00318327, 0.00144083, 0.00217854]
])
H4 = np.array([
    [8.00989155e-01, 1.17157987e-01, 3.08964444e-01],
 [3.60639602e-01, 2.93676643e-01, 1.81349029e-01],
 [3.59995978e-03, 5.30441647e-04, 1.89876456e-03]
])

