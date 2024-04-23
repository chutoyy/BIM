import cv2


def transform_images(path, new_path):
    img = cv2.imread(path)

    # centre de image
    center = (img.shape[1] / 2, img.shape[0] / 2)

    M = cv2.getRotationMatrix2D(center, angle=6, scale=1.0)
    new_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    #rendre flou
    inversed_img = cv2.GaussianBlur(new_img, (7, 7), 0)

    image = cv2.cvtColor(inversed_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(new_path, image)
