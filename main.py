import cv2
import numpy as np
import matplotlib.pyplot as plt



def preprocess_and_extract_features(image_path,
                                    target_size=(256, 256),
                                    n_features=500):
    """
    img (np.ndarray): the resized color image
    keypoints (list of cv2.KeyPoint), 
    descriptors (np.ndarray of shape [k, 32])
    """

    # load & resize 
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # grayscale & equalize 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ORB feature detection 
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return img, keypoints, descriptors

def visualize_keypoints(img,
                        keypoints,
                        figsize=(8, 8),
                        show=True,
                        save_path=None):
    """
    Draws ORB keypoints on the image using matplotlib
    """
    # Draw rich keypoints
    img_kp = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # Convert BGR to RGB for matplotlib
    img_kp = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(img_kp)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    img, keypoints, descriptors = preprocess_and_extract_features(path)
    print(f"Extracted {len(keypoints)} keypoints, descriptors shape: {descriptors.shape}")
    visualize_keypoints(img, keypoints)

