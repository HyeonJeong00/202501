import numpy as np
import os

worldPoints0 = np.array([        # 한 장의 사진에서 구한 것임.
    # [0,   0,  0],
    # [25,  0,  0],
    # [50,  0,  0],
    # [75,  0,  0],
    # [0, 25,  0],
    # [25, 25,  0]
    [0,0,0],
    [1,0,0],
    # [2,0,0],
    # [3,0,0],
    # [0,1,0],
    [1,1,0]
], dtype=float)

imagePoints0 = np.array([        # object point를 받아와서 world coordinate에 각각 쌍을 이룸
#     [382.48492,   96.906456],
#  [380.36853,  159.60272 ],
#  [378.03198,  221.97307 ],
#  [375.5168,   284.0935  ],
#  [320.55902,   99.69174 ],
#  [318.3758,   159.4883  ]
#  [48, 112],
#  [85,  110],
#  [124,  109],
#  [166,   106],
#  [48, 158],
#  [84,  158]
 [48.775787, 112.50374],
 [85.20374,  110.81499],
#  [124.25527,  109.05714],
#  [166.7278,   106.81466],
#  [48.173367, 158.28143],
 [84.32268,  158.48764]

], dtype=float)

A_rows0 = []
for (x, y, z), (u, v) in zip(worldPoints0, imagePoints0):
    row1 = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
    row2 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
    A_rows0.append(row1)
    A_rows0.append(row2)

A0 = np.array(A_rows0)
U0, S0, Vt0 = np.linalg.svd(A0)     # svd를 사용해서, projection matrix를 구함
print(A0)
print(S0)                        # s는 singular value인데, 0인 경우가 정확한 값이 나온다는 뜻임, 하지만 실제로 찍은 사진(noise와 왜곡..)으로 이를 구하기에 0을 사용하면 안된다...

# print(Vt0)                       # 실제로 값을 출력하면 맨 마지막 행은 0과 -1로만 이루어져 있다. 뒤에서 두번째 행을 사용하자.
print(Vt0[-1])

arr = np.array(Vt0[-1])

# 배열을 3행 4열의 행렬로 재구성
mat = arr.reshape(3, 4)
print("3x4 행렬:")
print(mat)

# 3번째 열(인덱스 2)을 제거하여 3x3 행렬로 만듦
mat_3x3 = np.delete(mat, 2, axis=1)
print("\n3번째 열 제거 후 3x3 행렬:")
print(mat_3x3)

output_file = 'maybe_output/calibration_result_0.txt'
output_dir = os.path.dirname(output_file)
# output_dir이 비어있지 않은 경우에 한해서 디렉터리 생성
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "w") as f:
    f.write("Your calibration results here.")
output_file = f"maybe_output/calibration_result_0.txt"
with open(output_file, "w") as f:
    f.write(f"S: {S0}\n\n")
    f.write("Vt:\n")
    f.write(f"{Vt0}\n\n")
    f.write("H:\n")
    f.write(f"{mat_3x3}\n\n")

outputs_file = 'maybe_output/homography.txt'
outputs_dir = os.path.dirname(outputs_file)
if outputs_dir and not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# 'a' 모드: 파일이 존재하면 내용 뒤에 추가하고, 없으면 새 파일 생성
with open(outputs_file, 'a') as f:
    f.write("H0:\n")
    f.write(f"{mat_3x3}\n\n")