import numpy as np
import os

worldPoints0 = np.array([        # 한 장의 사진에서 구한 것임.
    [0,   0,  0],
    [25,  0,  0],
    # [50,  0,  0],
    # [75,  0,  0],
    # [0, 25,  0],
    [25, 25,  0]
], dtype=float)

imagePoints0 = np.array([        # object point를 받아와서 world coordinate에 각각 쌍을 이룸

#  [439.25455,  142.12595 ],

#  [431.24683,  188.29335 ],

#  [423.92456,  230.4077  ],

#  [395.46658,   91.431816],

#  [389.3845,   142.22966 ]
[81.174995,  92.59433],
[133.4573,    92.480034],
[139.75656,  142.84091]
], dtype=float)

A_rows1 = []
for (x, y, z), (u, v) in zip(worldPoints0, imagePoints0):
    row1 = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
    row2 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
    A_rows1.append(row1)
    A_rows1.append(row2)

A1 = np.array(A_rows1)
U1, S1, Vt1 = np.linalg.svd(A1)     # svd를 사용해서, projection matrix를 구함
print(A1)
print(S1)                        # s는 singular value인데, 0인 경우가 정확한 값이 나온다는 뜻임, 하지만 실제로 찍은 사진(noise와 왜곡..)으로 이를 구하기에 0을 사용하면 안된다...
# 모든 값이 다 0이 아닌 수임!
# print(Vt1)                       # 실제로 값을 출력하면 맨 마지막 행은 0과 -1로만 이루어져 있다. 뒤에서 두번째 행을 사용하자.
print(Vt1[-1])

arr1 = np.array(Vt1[-1])

# 배열을 3행 4열의 행렬로 재구성
mat1 = arr1.reshape(3, 4)
print("3x4 행렬:")
print(mat1)

# 3번째 열(인덱스 2)을 제거하여 3x3 행렬로 만듦
mat_3x3_1 = np.delete(mat1, 2, axis=1)
print("\n3번째 열 제거 후 3x3 행렬:")
print(mat_3x3_1)

output_file = 'maybe_output/calibration_result_1.txt'
output_dir = os.path.dirname(output_file)
# output_dir이 비어있지 않은 경우에 한해서 디렉터리 생성
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "w") as f:
    f.write("Your calibration results here.")
output_file = f"maybe_output/calibration_result_1.txt"
with open(output_file, "w") as f:
    f.write(f"S: {S1}\n\n")
    f.write("Vt:\n")
    f.write(f"{Vt1}\n\n")
    f.write("H:\n")
    f.write(f"{mat_3x3_1}\n\n")

outputs_file = 'maybe_output/homography.txt'
outputs_dir = os.path.dirname(outputs_file)
if outputs_dir and not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# 'a' 모드: 파일이 존재하면 내용 뒤에 추가하고, 없으면 새 파일 생성
with open(outputs_file, 'a') as f:
    f.write("H1:\n")
    f.write(f"{mat_3x3_1}\n\n")