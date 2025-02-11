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
    # [553.14276, 144.66359],[552.5641,  206.06647],[551.86957, 267.23932],[551.1604,  328.14932],[492.18906, 144.09941],[491.7666,  205.7083 ]
    [113.52346, 140.38112],
    [177.93988, 140.80292],
    [178.00108, 204.12791]


], dtype=float)

A_rows2 = []
for (x, y, z), (u, v) in zip(worldPoints0, imagePoints0):
    row1 = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
    row2 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
    A_rows2.append(row1)
    A_rows2.append(row2)

A2 = np.array(A_rows2)
U2, S2, Vt2 = np.linalg.svd(A2)     # svd를 사용해서, projection matrix를 구함
print(A2)
print(S2)                        # s는 singular value인데, 0인 경우가 정확한 값이 나온다는 뜻임, 하지만 실제로 찍은 사진(noise와 왜곡..)으로 이를 구하기에 0을 사용하면 안된다...
# 마지막 값이 0이 나옴... Vt의 맨 뒤에서 두번째 값을 사용하자.
# print(Vt2)                       # 실제로 값을 출력하면 맨 마지막 행은 0과 -1로만 이루어져 있다. 뒤에서 두번째 행을 사용하자.
print(Vt2[-1])

arr2 = np.array(Vt2[-1])

# 배열을 3행 4열의 행렬로 재구성
mat2 = arr2.reshape(3, 4)
print("3x4 행렬:")
print(mat2)

# 3번째 열(인덱스 2)을 제거하여 3x3 행렬로 만듦
mat_3x3_2 = np.delete(mat2, 2, axis=1)
print("\n3번째 열 제거 후 3x3 행렬:")
print(mat_3x3_2)

output_file = 'maybe_output/calibration_result_2.txt'
output_dir = os.path.dirname(output_file)
# output_dir이 비어있지 않은 경우에 한해서 디렉터리 생성
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "w") as f:
    f.write("Your calibration results here.")
output_file = f"maybe_output/calibration_result_2.txt"
with open(output_file, "w") as f:
    f.write(f"S: {S2}\n\n")
    f.write("Vt:\n")
    f.write(f"{Vt2}\n\n")
    f.write("H:\n")
    f.write(f"{mat_3x3_2}\n\n")

outputs_file = 'maybe_output/homography.txt'
outputs_dir = os.path.dirname(outputs_file)
if outputs_dir and not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# 'a' 모드: 파일이 존재하면 내용 뒤에 추가하고, 없으면 새 파일 생성
with open(outputs_file, 'a') as f:
    f.write("H02:\n")
    f.write(f"{mat_3x3_2}\n\n")