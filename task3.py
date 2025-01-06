import matplotlib.pyplot as plt
import cv2
import numpy as np

corner_threshold = 1e6 
show_corners = True
scatterplot = False
print_evs = False

# load I1 saved as I1.jpeg
I1 = cv2.imread('images/leftang.jpeg', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('images/centang.jpeg', cv2.IMREAD_GRAYSCALE)  

def get_eigenvalues(img):
    blur = cv2.GaussianBlur(img, (5,5), 7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    Ix2_blur = cv2.GaussianBlur(Ix2, (25, 25), 35 * 5)
    Iy2_blur = cv2.GaussianBlur(Iy2, (25, 25), 35 * 5)
    IxIy_blur = cv2.GaussianBlur(IxIy, (25, 25), 35 * 5)

    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur

    eigenv1 = (trace + np.sqrt(trace**2 - 4*det))/2
    eigenv2 = (trace - np.sqrt(trace**2 - 4*det))/2
    
    print("the shape of eigenvalues: ", eigenv1.shape)

    return eigenv1, eigenv2

lambda1_I1, lambda2_I1 = get_eigenvalues(I1)
lambda1_I2, lambda2_I2 = get_eigenvalues(I2)

if print_evs:
    print("Eigenvalues for I1:")
    print("λ1 (I1):", lambda1_I1)
    print("λ2 (I1):", lambda2_I1)

    print("\nEigenvalues for I2:")
    print("λ1 (I2):", lambda1_I2)
    print("λ2 (I2):", lambda2_I2)


# Filter where lambda1 > lambda2
mask_I1 = lambda1_I1 > lambda2_I1
mask_I2 = lambda1_I2 > lambda2_I2


if scatterplot:
    # Scatter plot for I1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(lambda1_I1[mask_I1], lambda2_I1[mask_I1], s=1, alpha=0.5)
    plt.xlabel("λ1 (I1)")
    plt.ylabel("λ2 (I1)")
    plt.title("Scatter plot of Eigenvalues for I1")

    # Scatter plot for I2
    plt.subplot(1, 2, 2)
    plt.scatter(lambda1_I2[mask_I2], lambda2_I2[mask_I2], s=1, alpha=0.5)
    plt.xlabel("λ1 (I2)")
    plt.ylabel("λ2 (I2)")
    plt.title("Scatter plot of Eigenvalues for I2")

    plt.savefig("./images/scatterplot.png")
    plt.show()
    
if show_corners:
    # smaller threshold for more corners

    # Detect corners for I1
    corners_I1 = (lambda1_I1 > corner_threshold) & (lambda2_I1 > corner_threshold)
    I1_corners = np.zeros_like(I1)
    I1_corners[corners_I1] = 255  # Mark corners in a binary image

    # Detect corners for I2
    corners_I2 = (lambda1_I2 > corner_threshold) & (lambda2_I2 > corner_threshold)
    I2_corners = np.zeros_like(I2)
    I2_corners[corners_I2] = 255  # Mark corners in a binary image

    # Visualize detected corners on the original images
    plt.figure(figsize=(12, 6))

    # Plot original image with corners for I1
    plt.subplot(1, 2, 1)
    plt.imshow(I1, cmap='gray')
    plt.imshow(I1_corners, alpha=0.5)  # Overlay corners
    plt.title("Detected Corners in I1")

    # Plot original image with corners for I2
    plt.subplot(1, 2, 2)
    plt.imshow(I2, cmap='gray')
    plt.imshow(I2_corners, alpha=0.5)  # Overlay corners
    plt.title("Detected Corners in I2")
    plt.savefig("./images/corners.png")
    plt.show()
