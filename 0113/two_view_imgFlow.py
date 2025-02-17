import numpy as np
import cv2
from scene3D import SceneReconstruction3D

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    # K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
    #                1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
    # d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    K = np.array([[527.94400783,    0,   293.30808222,
                   0,   545.31328121,   226.73218949,
                   0,   0,  1]]).reshape(3,3)
    d = np.array([ 2.33178373e-01, -8.58219361e-01, -1.19475591e-04,  8.24016121e-03, 1.15078478e+00]).reshape(1,5)
    scene = SceneReconstruction3D(K, d)

    old_img = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img1 = cv2.imread('resized_img_R.jpg', cv2.CV_8UC3)
    # img1 = cv2.pyrUp(img1)
    # img1 = cv2.pyrDown(img1)
    img2 = cv2.imread('resized_img_L.jpg', cv2.CV_8UC3)
    # img2 = cv2.pyrUp(img2)
    # img2 = cv2.pyrDown(img2)

    # cv2.imshow("Image Right", img1)
    # cv2.imshow("Image Left", img2)
    cv2.waitKey()

    scene.load_image_pair(img1, img2)

    opt_flow_img = scene.plot_optic_flow()

    # cv2.imshow("imgFlow", opt_flow_img)
    cv2.imwrite(f'output/result_imgFlow.jpg', opt_flow_img)
    cv2.waitKey()

    Xs, Zs, Ys = scene.plot_point_cloud()

    ax.scatter(Xs, Ys, Zs, c=Ys, cmap=cm.hsv, marker='o')

    # 그래프 저장
    plt.savefig(f"output/scatter_plot.png", dpi=300)  # dpi는 해상도를 설정
    plt.show()


if __name__ == '__main__':
    main()