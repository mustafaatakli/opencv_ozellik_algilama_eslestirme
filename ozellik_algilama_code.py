import cv2
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bfm.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1_gray, keypoints1, img2_gray, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
scale_factor = 0.5
height, width = img_matches.shape[:2]
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
resized_img_matches = cv2.resize(img_matches, (new_width, new_height))

cv2.imshow("cikti_goruntusu", resized_img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

