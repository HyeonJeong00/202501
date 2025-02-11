import numpy as np
import os

worldPoints3 = np.array([        # 한 장의 사진에서 구한 것임.
    [0,   0,  0],
    [25,  0,  0],
    [25, 25,  0]
], dtype=float)

imagePoints3 = np.array([        # object point를 받아와서 world coordinate에 각각 쌍을 이룸
    # [113.52346, 140.38112],
    # [177.93988, 140.80292],
    # [178.00108, 204.12791]
    [162.71867,   95.50896],
    [221.2643,   100.0823],
    [221.21442,  157.27885]


], dtype=float)

A_rows3 = []
for (x, y, z), (u, v) in zip(worldPoints3, imagePoints3):
    row1 = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u]
    row2 = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]
    A_rows3.append(row1)
    A_rows3.append(row2)

A3 = np.array(A_rows3)
U3, S3, Vt3 = np.linalg.svd(A3)     # svd를 사용해서, projection matrix를 구함
print(A3)
print(S3)                        # s는 singular value인데, 0인 경우가 정확한 값이 나온다는 뜻임, 하지만 실제로 찍은 사진(noise와 왜곡..)으로 이를 구하기에 0을 사용하면 안된다...
# 마지막 값이 0이 나옴... Vt의 맨 뒤에서 두번째 값을 사용하자.
# print(Vt2)                       # 실제로 값을 출력하면 맨 마지막 행은 0과 -1로만 이루어져 있다. 뒤에서 두번째 행을 사용하자.
print(Vt3[-1])

arr3 = np.array(Vt3[-1])

# 배열을 3행 4열의 행렬로 재구성
mat3 = arr3.reshape(3, 4)
print("3x4 행렬:")
print(mat3)

# 3번째 열(인덱스 2)을 제거하여 3x3 행렬로 만듦
mat_3x3_3 = np.delete(mat3, 2, axis=1)
print("\n3번째 열 제거 후 3x3 행렬:")
print(mat_3x3_3)

output_file = 'maybe_output/calibration_result_3.txt'
output_dir = os.path.dirname(output_file)
# output_dir이 비어있지 않은 경우에 한해서 디렉터리 생성
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "w") as f:
    f.write("Your calibration results here.")
output_file = f"maybe_output/calibration_result_3.txt"
with open(output_file, "w") as f:
    f.write(f"S: {S3}\n\n")
    f.write("Vt:\n")
    f.write(f"{Vt3}\n\n")
    f.write("H:\n")
    f.write(f"{mat_3x3_3}\n\n")

outputs_file = 'maybe_output/homography.txt'
outputs_dir = os.path.dirname(outputs_file)
if outputs_dir and not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# 'a' 모드: 파일이 존재하면 내용 뒤에 추가하고, 없으면 새 파일 생성
with open(outputs_file, 'a') as f:
    f.write("H03:\n")
    f.write(f"{mat_3x3_3}\n\n")