# Name: Guilherme Mafra da Costa
# USP Number: 11272015
# Course Code: SCC0251
# Year/Semester: 2024/01
# Title: Enhancement and Superresolution

from lib import *

low_img_name = input().rstrip()
high_img_name = input().rstrip()
enhancement_method = int(input().rstrip())
gamma = float(input().rstrip())

low_img, high_img = get_input(low_img_name, high_img_name)

if enhancement_method == 0:
    img = superresolution(low_img)
elif enhancement_method == 1:
    img = single_image_cumulative_histogram(low_img)
elif enhancement_method == 2:
    img = joint_cumulative_histogram(low_img)
else:
    img = gamma_correction(low_img, gamma)
 
rmse(high_img, img)