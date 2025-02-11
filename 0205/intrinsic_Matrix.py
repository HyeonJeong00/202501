import numpy as np
import os


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

def scale_H(H):
    """
    H 행렬의 (2,2) 원소(H[2,2])가 1이 되도록 스케일링
    """
    factor = H[2, 2]
    if factor != 0:
        return H / factor
    return H


H1s = scale_H(H1)
H2s = scale_H(H2)
H3s = scale_H(H3)
H4s = scale_H(H4)
# [[129.77985685  72.86369241  48.77572111]
#  [119.72371192 240.3246475  112.50358861]
#  [  1.09563119   0.88600096   1.        ]]
# [[204.86425399  77.6286284   81.17511134]
#  [140.50816308 147.83406484  92.59446451]
#  [  1.51938489   0.48516922   1.        ]]
# [[262.58145822 117.81748327 113.52358919]
#  [205.75734666 230.06840361 140.38128288]
#  [  1.46119419   0.66137413   1.        ]]


# --- 2. v_ij 계산 함수 ---
def compute_v_ij(H, i, j):
    v = np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])
    return v

def compute_constraints(H):
    v12 = compute_v_ij(H, 0, 1)
    v11 = compute_v_ij(H, 0, 0)
    v22 = compute_v_ij(H, 1, 1)
    return v12, (v11 - v22)

v12_1, diff_1 = compute_constraints(H1s)
v12_2, diff_2 = compute_constraints(H2s)
v12_3, diff_3 = compute_constraints(H3s)
v12_4, diff_4 = compute_constraints(H4s)

# 각 H마다 2개의 식이 있으므로 전체 V 행렬은 (6 x 6)
# V = np.vstack([v12_1, diff_1, v12_2, diff_2, v12_3, diff_3, v12_4, diff_4])
V = np.vstack([v12_1, diff_1, v12_2, diff_2, v12_3, diff_3])
print("Constraint matrix V:")
print(V)

# --- 4. Vb = 0 문제를 SVD를 통해 풀어서 b를 구함 ---
U, S, VT = np.linalg.svd(V)
print("\nS:")
print(S)
print("\nVT:")
print(VT)
b = VT[-1, :]
print("\nSolution vector b (up to scale):")
print(b)                # why B33의 값들이 다 1보다 작지.... 이상하다...
b /= b[-1]

B11, B12, B22, B13, B23, B33 = b        
print(b)
# --- 5. 내부 파라미터 복원 (Zhang's method 공식) ---
v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

# det_B = B11 * B22 - B12**2
# print(det_B)
# if abs(det_B) < 1e-6:
#     raise ValueError("Singular matrix detected: det(B) is too small")

print(lambda_)          # lambda_ 값은 0보다 커야함함
alpha = np.sqrt(lambda_ / B11)
beta  = np.sqrt(lambda_ * B11 / abs(B11 * B22 - B12**2))
# print(beta)
# gamma = -B12 * alpha**2 * beta / lambda_
gamma = 0           
u0    = gamma * v0 / beta - B13 * alpha**2 / lambda_

K_est = np.array([
    [alpha, gamma, u0],
    [0,     beta,  v0],
    [0,     0,     1]
])

print("\nEstimated camera intrinsic matrix K:")
print(K_est)


# Estimated camera intrinsic matrix K:
# [[   8.87328501    0.          165.95522268]
#  [   0.           85.25039568 -144.63689184]
#  [   0.            0.            1.        ]]



P1 = np.array([
    [3.92564596e-01,  2.20401738e-01, -1.05856839e-02,  1.47539240e-01],
   [3.62146269e-01,  7.26946005e-01,  0.00000000e+00,  3.40306477e-01],
   [3.31412273e-03,  2.68002332e-03,  0.00000000e+00,  3.02484591e-03]
])
P2 = np.array([
    [ 6.32932214e-01,  2.39835195e-01, -7.39263984e-03,  2.50792132e-01],
   [4.34102780e-01,  4.56736304e-01,  0.00000000e+00,  2.86072447e-01],
   [4.69416896e-03,  1.49894170e-03,  0.00000000e+00,  3.08952446e-03]
])
P3 = np.array([
    [ 5.72044214e-01,  2.56670099e-01, -5.55800868e-03,  2.47315684e-01],
   [4.48250607e-01,  5.01213217e-01,  0.00000000e+00,  3.05826238e-01],
   [3.18327099e-03, 1.44083367e-03,  0.00000000e+00,  2.17854251e-03]
])
extrin_matrix1 = K_est.T @ P1 
print("\nextrinsic_matrix_1:")
print(extrin_matrix1)
extrin_matrix2 = K_est.T @ P2 
print("\nextrinsic_matrix_2:")
print(extrin_matrix2)
extrin_matrix3 = K_est.T @ P3 
print("\nextrinsic_matrix_3:")
print(extrin_matrix3)

# output_file = 'output_matrix/intrinsic_matrix.txt'
# output_dir = os.path.dirname(output_file)
# # output_dir이 비어있지 않은 경우에 한해서 디렉터리 생성
# if output_dir and not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# with open(output_file, "w") as f:
#     f.write("Your calibration results here.")
# output_file = f"output_matrix/intrinsic_matrix.txt"
# with open(output_file, "w") as f:
#     f.write(f"S: {S0}\n\n")
#     f.write("Solution vector b (up to scale):\n")
#     f.write(f"{b}\n\n")
#     f.write("Estimated camera intrinsic matrix K:\n")
#     f.write(f"{K_est}\n\n")
