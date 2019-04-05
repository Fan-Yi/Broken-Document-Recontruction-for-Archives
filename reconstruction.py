import imutils
import cv2
import numpy as np
import math
import os
import shutil
import datetime
import sys


alpha = 10
beta = 5
overlap_tolerance = 0.05
match_len_tolerance = 0.05
mini_angle = 5
center_dist_bound = 10
closeness_bound_for_center_dist = 1
closeness_bound_for_perp_dist = 0.1
FLIP_ALLOWED_MODE = False
BACKGROUND_IS_BLACK = True
BACKGROUND_IS_WHITE = False
SINGLE_FRAGMENT_NAME_PREFIX = "transparent_backgound_fragment"

# for simple shapes
# input_fragment_file_name = "shapes_and_colors.jpg"
# interesting_area_lower_bound = 2500
# interesting_area_upper_bound = 20000000
# dp_approx_precision = 0.01

# for letter
input_fragment_file_name = "letter.tif"
interesting_area_lower_bound = 100000
interesting_area_upper_bound = 20000000
dp_approx_precision = 0.005

# for map
# input_fragment_file_name = "map_2_fragment.tif"
# interesting_area_lower_bound = 25000
# interesting_area_upper_bound = 20000000
# dp_approx_precision = 0.002

cnts = list()
approx_cnts = list()
flipped_approx_cnts = list()
flipped_cnts = list()

compatible_match_list = list()

partial_picture = list()
partial_picture_for_movement = list()

first_placed_fragment_id = -1

matched_fragments = list()
current_matched_fragment_pile = list()

fragment_flipped = list()

map_to_perimeter = dict()
perimeter_list = list()
barycenter_and_x_to_angle_dict = dict()

fragment_barycenter_list = list()
flipped_fragment_barycenter_list = list()

tested_case_set = set()


def enlarged_by_a_factor(binary, mul):
    return int(binary[0] * mul), int(binary[1] * mul)


def initialize_approx_cnts(img):
    if FLIP_ALLOWED_MODE:
        image_containing_flipped_shape = np.zeros(enlarged_by_a_factor(img.shape[0:2], 1))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]  # for book, for toy example

    # inverted_thresh = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)[1]  # for map, statement 1
    # thresh = cv2.subtract(255, inverted_thresh)  # for map, statement 2

    # find contours in the thresholded image
    temp_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # for book, for toy example

    # temp_cnts = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,
    #                              cv2.CHAIN_APPROX_NONE)  # for map, choose either statement

    # temp_cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # for map

    temp_cnts = imutils.grab_contours(temp_cnts)

    for i in range(len(temp_cnts)):
        if interesting_area_lower_bound <= cv2.contourArea(temp_cnts[i]) < interesting_area_upper_bound:  # for map
            cnts.append(temp_cnts[i])

    print("cnt_num: %d" % len(cnts))

    for i in range(len(cnts)):

        c = cnts[i]

        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("barycenter of the original curve: ", cX, cY)
            fragment_barycenter_list.append((cX, cY))
        else:
            fragment_barycenter_list.append((None, None))

        if FLIP_ALLOWED_MODE:
            flipped_c = np.array([[[-px + 2 * cX, py]] for [[px, py]] in c])
            flipped_cnts.append(flipped_c)

            M = cv2.moments(flipped_c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("barycenter of the original curve: ", cX, cY)
                flipped_fragment_barycenter_list.append((cX, cY))
            else:
                flipped_fragment_barycenter_list.append((None, None))

        dp_epsilon = dp_approx_precision * cv2.arcLength(c, True)  # for book, for letter

        approx_c = cv2.approxPolyDP(c, dp_epsilon, True)  # apply DP Algorithm to cluster the boundary points

        approx_cnts.append(approx_c)  # index starts from zero

        # for testing copying
        print("start to test copying...")

        mask = np.zeros(img.shape[:2], np.uint8)
        print("having init a mask")

        cv2.drawContours(mask, [approx_c], -1, 255, -1)
        print("having drawn contours")

        x, y, w, h = cv2.boundingRect(approx_c)

        bit_dealt_img = cv2.bitwise_and(img, img, mask=mask)
        print("bitwise_and succeed")

        crop_img = bit_dealt_img[y:y + h, x:x + w]

        trans_bg_img_rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGBA)

        print("check the channel below")
        print(trans_bg_img_rgba.shape)

        if BACKGROUND_IS_BLACK:
            bg_color = [0, 0, 0, 255]
        elif BACKGROUND_IS_WHITE:
            bg_color = [255, 255, 255, 255]
        print("check whether the last element is 0 or 255")
        print(bg_color)

        mask = np.all(trans_bg_img_rgba == bg_color, axis=2)
        # print("check whether there are true and false, they refer to black/white or not")
        # print(mask)

        trans_bg_img_rgba[mask] = bg_color[:3] + [0]
        print("having applied mask and replaced those completely black pixel with transparent ones")

        cv2.imwrite(SINGLE_FRAGMENT_NAME_PREFIX + "_" + str(i) + ".png", trans_bg_img_rgba)

        print("having completed the copy of Fragment %d" % i)

        # end testing copying

        M = cv2.moments(approx_c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("barycenter of the original approx curve: ", cX, cY)

        cv2.drawContours(image, [approx_c], 0, (255, 0, 255), 2)

        if FLIP_ALLOWED_MODE:
            cv2.drawContours(image_containing_flipped_shape, [flipped_c], 0, (255, 0, 255), 2)

        print("having drawn contour %d" % i)

        if M["m00"] != 0:
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, str(i), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if FLIP_ALLOWED_MODE:
            flipped_approx_c = np.array([[[-px + 2 * cX, py]] for [[px, py]] in approx_c])
            # print("compare the org and the flipped curve:")
            # print(approx_c)
            # print(flipped_approx_c)

            M = cv2.moments(flipped_approx_c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("barycenter of the flipped approx curve: ", cX, cY)

            flipped_approx_cnts.append(flipped_approx_c)  # index starts from zero

    # show the image

    obtain_perimeter_and_original_and_flipped_barycenter_x_to_angle_info(approx_cnts)

    print("perimeter list:\n", perimeter_list)

    perimeter_list.sort(key=lambda tup: (tup[3], tup[0]))

    print("perimeter list:\n", perimeter_list)

    print("barycenter_and_x_to_angle_dict:\n", barycenter_and_x_to_angle_dict)

    # print("about to show and write image")
    cv2.imshow("Approx Image", image)
    cv2.imwrite("Approx input_image.jpg", image)

    if FLIP_ALLOWED_MODE:
        cv2.imshow("Flipped Image", image_containing_flipped_shape)
        cv2.imwrite("Flipped input_image.jpg", image_containing_flipped_shape)

    cv2.waitKey()


def obtain_perimeter_and_original_and_flipped_barycenter_x_to_angle_info(cnt_list):
    # just the properties of a single segment cluster
    if id(cnt_list) == id(approx_cnts):
        tag_str = "approx_cnts"
        flipped_tag_str = "flipped_approx_cnts"
    elif id(cnt_list) == id(flipped_approx_cnts):
        print("no operations needed because having done")
        return
    else:
        print("inappropriate input")
        sys.exit()

    for i in range(len(cnt_list)):
        for k in range(len(cnt_list[i])):
            if k < len(cnt_list[i]) - 1:
                curve_segment_ik = [cnt_list[i][k], cnt_list[i][k + 1]]
            else:
                curve_segment_ik = [cnt_list[i][k], cnt_list[i][0]]

            x1 = curve_segment_ik[0][0][0]
            y1 = curve_segment_ik[0][0][1]
            x2 = curve_segment_ik[1][0][0]
            y2 = curve_segment_ik[1][0][1]

            perimeter = euclidean_distance(x1, y1, x2, y2)
            perimeter_list.append((tag_str, i, k, perimeter))
            if FLIP_ALLOWED_MODE:
                perimeter_list.append((flipped_tag_str, i, k, perimeter))

            map_to_perimeter[(i, k)] = perimeter

            original_barycenter = ((x1 + x2) / 2, (y1 + y2) / 2)

            if FLIP_ALLOWED_MODE:
                flipped_x1 = -x1 + 2 * fragment_barycenter_list[i][0]
                flipped_x2 = -x2 + 2 * fragment_barycenter_list[i][0]
                flipped_barycenter = ((flipped_x1 + flipped_x2) / 2, (y1 + y2) / 2)

            if abs(x1 - x2) < 1:
                original_x_to_angle = 90
            else:
                original_x_to_angle = int(math.atan((y1 - y2) / (x1 - x2)) * 180 / math.pi)

            if FLIP_ALLOWED_MODE:
                flipped_x_to_angle = 180 - original_x_to_angle

            barycenter_and_x_to_angle_dict[(tag_str, i, k)] = (original_barycenter, original_x_to_angle)
            if FLIP_ALLOWED_MODE:
                barycenter_and_x_to_angle_dict[(flipped_tag_str, i, k)] = (flipped_barycenter, flipped_x_to_angle)


def len_close(l1, l2, error_rate):
    if l1 > l2:
        return (l1 - l2) / l2 < error_rate
    else:
        return (l2 - l1) / l1 < error_rate


def segment_close(curve_segment_1, curve_segment_2, closeness_bound):
    x11 = curve_segment_1[0][0][0]
    y11 = curve_segment_1[0][0][1]
    x12 = curve_segment_1[1][0][0]
    y12 = curve_segment_1[1][0][1]

    x21 = curve_segment_2[0][0][0]
    y21 = curve_segment_2[0][0][1]
    x22 = curve_segment_2[1][0][0]
    y22 = curve_segment_2[1][0][1]

    dist_1 = euclidean_distance(x11, y11, x12, y12)
    # print("compute dist_1 from: ", x11, y11, x12, y12)
    dist_2 = euclidean_distance(x21, y21, x22, y22)
    # print("compute dist_2 from: ", x21, y21, x22, y22)
    print("dist_1: ", round(dist_1))
    print("dist_2: ", round(dist_2))
    center_1 = ((x11 + x12) / 2, (y11 + y12) / 2)
    center_2 = ((x21 + x22) / 2, (y21 + y22) / 2)
    center_dist = euclidean_distance(center_1[0], center_1[1], center_2[0], center_2[1])
    print("center_dist: ", round(center_dist))

    if center_dist < closeness_bound_for_center_dist * dist_1 / 2 and center_dist < closeness_bound_for_center_dist * dist_2 / 2:
        # if center_dist < center_dist_bound:
        return True

    print("perpendicular distances:")
    print(round(perpendicular_distance_between_point_to_two_point_line(center_1, (x21, y21), (x22, y22))))
    print(round(perpendicular_distance_between_point_to_two_point_line(center_2, (x11, y11), (x12, y12))))

    if perpendicular_distance_between_point_to_two_point_line(
            center_1, (x21, y21), (x22, y22)) < closeness_bound_for_perp_dist * dist_1 / 2:
        return True

    if perpendicular_distance_between_point_to_two_point_line(
            center_2, (x11, y11), (x12, y12)) < closeness_bound_for_perp_dist * dist_2 / 2:
        return True

    return False


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def inner_angle_between_lines(x11, y11, x12, y12, x21, y21, x22, y22):
    if abs(x11 - x12) < 1:
        angle_1 = 90
    else:
        angle_1 = int(math.atan((y11 - y12) / (x11 - x12)) * 180 / math.pi)

    if abs(x21 - x22) < 1:
        angle_2 = 90
    else:
        angle_2 = int(math.atan((y21 - y22) / (x21 - x22)) * 180 / math.pi)

    return abs(angle_1 - angle_2)


def contour_area_overlapped(cnt1, cnt2, img, area_overlap_tolerance):
    # cnt1 and cnt2 are two contours
    # img is the image where they are located in
    # epsilon is the tolerance

    blank = np.zeros(img.shape[0:2])
    print("to create 2 images for computing overlap")

    # copy each of the contours (assuming there's just two) to its own image.
    # Just fill with a '1'.
    # Then fill in the area sorrounded by the contours
    # img1 = cv2.drawContours(blank.copy(), [approx_cnts[j]], 0, 1)
    img1 = cv2.drawContours(blank.copy(), [cnt1], 0, 1, thickness=-1)
    # cv2.fillPoly(img1, pts=[approx_cnts[j]], color=(255, 255, 255))
    # img2 = cv2.drawContours(blank.copy(), [rotated_approx_c_1], 0, 1)
    img2 = cv2.drawContours(blank.copy(), [cnt2], 0, 1, thickness=-1)
    # cv2.fillPoly(img2, pts=[rotated_approx_c_1], color=(255, 255, 255))

    # we could just add img1 to img2 and pick all points that sum to 2 (1+1=2):
    intersection = (img1 + img2) == 2

    intersection_area = 0
    for i in range(0, len(intersection)):
        for j in range(0, len(intersection[i])):
            if intersection[i][j]:
                intersection_area += 1

    contour_area_1 = cv2.contourArea(cnt1)
    contour_area_2 = cv2.contourArea(cnt2)
    print("contour_area_1: ", contour_area_1)
    print("contour_area_2: ", contour_area_2)

    if contour_area_1 < contour_area_2:
        if intersection_area / contour_area_1 > area_overlap_tolerance:
            return True
    else:
        if intersection_area / contour_area_2 > area_overlap_tolerance:
            return True

    return False


def vector_inner_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot_product = x1 * x2 + y1 * y2
    v1_len = math.sqrt(x1 ** 2 + y1 ** 2)
    v2_len = math.sqrt(x2 ** 2 + y2 ** 2)
    theta = int(math.acos(dot_product / (v1_len * v2_len)) * 180 / math.pi)
    return theta


def coefficients_of_a_line_passing_thru_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    return a, b, c


def perpendicular_distance_between_point_line(p, line):
    x0, y0 = p
    a, b, c = line
    d = abs((a * x0 + b * y0 + c)) / (math.sqrt(a * a + b * b))
    return d


def perpendicular_distance_between_point_to_two_point_line(p, p1, p2):
    a, b, c = coefficients_of_a_line_passing_thru_two_points(p1, p2)
    d = perpendicular_distance_between_point_line(p, (a, b, c))
    return d


def fitness_of_a_match(cnt1, cnt2, tag_str_i, tag_str_j, i, j):
    len_of_match_seg = 0
    match_segment_set_ik = set()
    match_segment_set_jl = set()
    match_angle_list_ik = list()
    match_angle_list_jl = list()
    map_to_vector = dict()

    print("size of the two matching curves")
    print(len(cnt1))
    print(len(cnt2))

    for k in range(len(cnt1)):
        if k < len(cnt1) - 1:
            curve_segment_ik = (cnt1[k], cnt1[k + 1])
        else:
            curve_segment_ik = (cnt1[k], cnt1[0])

        for l in range(len(cnt2)):
            if l < len(cnt2) - 1:
                curve_segment_jl = (cnt2[l], cnt2[l + 1])
            else:
                curve_segment_jl = (cnt2[l], cnt2[0])

            print("for k = %d and l = %d" % (k, l))

            close_to_each_other = segment_close(curve_segment_ik, curve_segment_jl, closeness_bound_for_center_dist)

            x11 = curve_segment_ik[0][0][0]
            y11 = curve_segment_ik[0][0][1]
            x12 = curve_segment_ik[1][0][0]
            y12 = curve_segment_ik[1][0][1]

            x21 = curve_segment_jl[0][0][0]
            y21 = curve_segment_jl[0][0][1]
            x22 = curve_segment_jl[1][0][0]
            y22 = curve_segment_jl[1][0][1]

            # cannot use the two lines below, because at least one curve has been rotated
            # bary_center_ik, x_to_ang_ik = barycenter_and_x_to_angle_dict[(tag_str_i, i, k)]
            # bary_center_jl, x_to_ang_jl = barycenter_and_x_to_angle_dict[(tag_str_j, j, l)]

            inner_ang = inner_angle_between_lines(x11, y11, x12, y12, x21, y21, x22, y22)
            # inner_ang = abs(x_to_ang_ik - x_to_ang_jl)
            print("inner_ang for checking closeness: ", inner_ang)

            if not close_to_each_other:  # segment close to each other
                continue
            else:
                None

            if mini_angle <= inner_ang <= 180 - mini_angle:  # angle difference tolerance < 5 degrees
                continue

            match_segment_set_ik.add(k)
            match_segment_set_jl.add(l)
            map_to_vector[(i, k)] = curve_segment_ik
            map_to_vector[(j, l)] = curve_segment_jl
            print("having collected a map from (%d, %d) to " % (i, k), curve_segment_ik)
            print("having collected a map from (%d, %d) to " % (j, l), curve_segment_jl)
            # cv2.waitKey(0)

            perimeter_ik = euclidean_distance(x11, y11, x12, y12)
            perimeter_jl = euclidean_distance(x21, y21, x22, y22)

            print("perimeter_ik and perimeter_jl: ", round(perimeter_ik, 2), round(perimeter_jl, 2))

    print("two matched segment sets:")
    print(match_segment_set_ik)
    print(match_segment_set_jl)

    num_of_match_seg = len(match_segment_set_ik) + len(match_segment_set_jl)

    for e in match_segment_set_ik:
            len_of_match_seg += map_to_perimeter[(i, e)]

    for e in match_segment_set_jl:
            len_of_match_seg += map_to_perimeter[(j, e)]

    if len(match_segment_set_ik) >= 2:
        match_segment_list_ik = sorted(match_segment_set_ik)
        match_segment_list_ik.append(match_segment_list_ik[0])
        print("having constructed a list:\n", match_segment_list_ik)
        for p in range(len(match_segment_list_ik) - 1):
            index_p = match_segment_list_ik[p]
            index_p_ = match_segment_list_ik[p+1]
            curve_segment_ik = map_to_vector[(i, index_p)]
            # len_of_match_seg += map_to_perimeter[(i, index_p)]
            curve_segment_ik_ = map_to_vector[(i, index_p_)]
            print("two vectors: ", curve_segment_ik, curve_segment_ik_)
            v = (curve_segment_ik[1][0][0] - curve_segment_ik[0][0][0],
                 curve_segment_ik[1][0][1] - curve_segment_ik[0][0][1])
            v_ = (curve_segment_ik_[1][0][0] - curve_segment_ik_[0][0][0],
                  curve_segment_ik_[1][0][1] - curve_segment_ik_[0][0][1])
            print("two vectors: ", v, v_)
            theta = vector_inner_angle(v, v_)
            print("vector inner angle: ", theta)
            match_angle_list_ik.append(theta)
            # cv2.waitKey(0)

    if len(match_segment_set_jl) >= 2:
        match_segment_list_jl = sorted(match_segment_set_jl)
        match_segment_list_jl.append(match_segment_list_jl[0])
        print("having constructed a list:\n", match_segment_list_jl)
        for p in range(len(match_segment_list_jl) - 1):
            index_p = match_segment_list_jl[p]
            index_p_ = match_segment_list_jl[p+1]
            curve_segment_jl = map_to_vector[(j, index_p)]
            # len_of_match_seg += map_to_perimeter[(j, index_p)]
            curve_segment_jl_ = map_to_vector[(j, index_p_)]
            print("two vectors: ", curve_segment_jl, curve_segment_jl_)
            v = (curve_segment_jl[1][0][0] - curve_segment_jl[0][0][0],
                 curve_segment_jl[1][0][1] - curve_segment_jl[0][0][1])
            v_ = (curve_segment_jl_[1][0][0] - curve_segment_jl_[0][0][0],
                  curve_segment_jl_[1][0][1] - curve_segment_jl_[0][0][1])
            print("two vectors: ", v, v_)
            theta = vector_inner_angle(v, v_)
            print("vector inner angle: ", theta)
            match_angle_list_jl.append(theta)
            # cv2.waitKey(0)

    print("having collected vector inner angles as follows")
    print(match_angle_list_ik)
    print(match_angle_list_jl)

    if not match_angle_list_ik:
        match_angle_list_ik = [0]
    if not match_angle_list_jl:
        match_angle_list_jl= [0]
    rotate_extend = round(sum(match_angle_list_ik) + sum(match_angle_list_jl))
    print("num_of_match_seg_part: %d" % (alpha * num_of_match_seg))
    print("rotate_extend_part: %d" % (beta * rotate_extend))
    print("len_of_match_seg_part: %d" % round(len_of_match_seg))
    # cv2.waitKey(0)
    match_fitness = num_of_match_seg * alpha + round(len_of_match_seg) + beta * rotate_extend
    print("total fitness: %d" % match_fitness)
    # cv2.waitKey(0)

    return match_fitness


def create_or_clear_one_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Directory ", folder_name, " Created ")
    else:
        print("Directory ", folder_name, " already exists")
        for the_file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def geometric_relation_info(m, n):
    print("perimeter info:")
    print(perimeter_list[m])
    print(perimeter_list[n])

    if perimeter_list[m][1] <= perimeter_list[n][1]:
        tag_str_i, i, k, perimeter_ik = perimeter_list[m]
        tag_str_j, j, l, perimeter_jl = perimeter_list[n]
    else:
        tag_str_i, i, k, perimeter_ik = perimeter_list[n]
        tag_str_j, j, l, perimeter_jl = perimeter_list[m]

    original_barycenter_ik, x_to_angle_i = barycenter_and_x_to_angle_dict[(tag_str_i, i, k)]
    original_barycenter_jl, x_to_angle_j = barycenter_and_x_to_angle_dict[(tag_str_j, j, l)]

    print("curve segments: ")

    print("for curves %d and %d" % (i, j))
    print("\tclusters %d and %d may be matched" % (k, l))

    print("center_point_ik: ")
    print(original_barycenter_ik)
    print("center_point_jl: ")
    print(original_barycenter_jl)

    to_ang = x_to_angle_i - x_to_angle_j
    print("former to latter, to angle is %d" % (to_ang))

    x_shift = original_barycenter_jl[0] - original_barycenter_ik[0]
    y_shift = original_barycenter_jl[1] - original_barycenter_ik[1]

    return x_shift, y_shift, to_ang, original_barycenter_ik, original_barycenter_jl, perimeter_ik, perimeter_jl


def str_list_name(lst_name):
    if id(lst_name) == id(approx_cnts):
        return "approx_cnts"
    elif id(lst_name) == id(flipped_approx_cnts):
        return "flipped_approx_cnts"


def flipped_str_list_name(lst_name):
    if id(lst_name) == id(approx_cnts):
        return "flipped_approx_cnts"
    elif id(lst_name) == id(flipped_approx_cnts):
        return "approx_cnts"


def list_name_from_str(str_name):
    if str_name == "approx_cnts":
        return approx_cnts
    elif str_name == "flipped_approx_cnts":
        return flipped_approx_cnts


def cut_tail(s):
    return s[:s.rfind(".")]


def compute_compatible_matches(m, n, overlap_tolerance, img, compatibleDirName):
    print("dealing with the %d-th and the %d-th tuples of %d tuples" % (m, n, len(perimeter_list)))

    # if perimeter_list[m][1] == perimeter_list[n][1]:
    #     print("they belong to the same fragment %d" % perimeter_list[n][1])
    #     return

    if perimeter_list[m][1] < perimeter_list[n][1]:
        tag_str_i, i, k, perimeter_ik = perimeter_list[m]
        tag_str_j, j, l, perimeter_jl = perimeter_list[n]
    else:
        tag_str_i, i, k, perimeter_ik = perimeter_list[n]
        tag_str_j, j, l, perimeter_jl = perimeter_list[m]

    if perimeter_ik == 0 or perimeter_jl == 0:
        print("perimeter is equal to 0, no operations needed.")
        return

    if i >= j:
        print("the value of i should be smaller than the value of j")
        sys.exit()

    if FLIP_ALLOWED_MODE:
        if (tag_str_i, tag_str_j, i, j, k, l) in tested_case_set:
            print((tag_str_i, tag_str_j, i, j, k, l), end=" ")
            print("has been tested. No operations needed.")
            return

    print("for contours %d and %d," % (i, j))
    print("\tfor clusters %d and %d" % (k, l))

    x_shift, y_shift, to_ang, original_barycenter_ik, original_barycenter_jl, perimeter_ik, perimeter_jl \
        = geometric_relation_info(m, n)

    print("\t\tperimeters: %.2f and %.2f" % (perimeter_ik, perimeter_jl))

    print("\t\tlen close to each other")

    lst_name_i = list_name_from_str(tag_str_i)
    lst_name_j = list_name_from_str(tag_str_j)

    blank_image_1 = np.zeros(enlarged_by_a_factor(img.shape[0:2], 2))
    blank_image_2 = np.zeros(enlarged_by_a_factor(img.shape[0:2], 2))

    # translation
    translate_M = np.matrix([[1, 0, x_shift], [0, 1, y_shift]])

    inverse_translate_M = np.matrix([[1, 0, -x_shift], [0, 1, -y_shift]])

    translated_approx_c = np.array(approx_cnts[i])  # bad implementation

    cv2.transform(lst_name_i[i], translate_M, translated_approx_c)

    # cv2.drawContours(blank_image_1, [lst_name_i[i]], 0, (255, 0, 255),
    #                  1)  # mind the color, I don't know how to make it colorful
    # cv2.drawContours(blank_image_2, [lst_name_i[i]], 0, (255, 0, 255),
    #                  1)  # mind the color, I don't know how to make it colorful
    #
    # cv2.drawContours(blank_image_1, [translated_approx_c], 0, (255, 0, 255),
    #                  1)  # mind the color, I don't know how to make it colorful
    # cv2.drawContours(blank_image_2, [translated_approx_c], 0, (255, 0, 255),
    #                  1)  # mind the color, I don't know how to make it colorful

    # rotation
    rotate_M_1 = cv2.getRotationMatrix2D((original_barycenter_jl[0], original_barycenter_jl[1]), to_ang, 1)
    rotate_M_2 = cv2.getRotationMatrix2D((original_barycenter_jl[0], original_barycenter_jl[1]), to_ang + 180, 1)

    inverse_rotate_M_1 = cv2.getRotationMatrix2D((original_barycenter_ik[0], original_barycenter_ik[1]), -to_ang, 1)
    inverse_rotate_M_2 = cv2.getRotationMatrix2D((original_barycenter_ik[0], original_barycenter_ik[1]), -to_ang - 180,
                                                 1)

    rotated_approx_c_11 = np.array(approx_cnts[i])  # bad implementation
    rotated_approx_c_12 = np.array(approx_cnts[i])  # bad implementation

    cv2.transform(translated_approx_c, rotate_M_1, rotated_approx_c_11)
    cv2.transform(translated_approx_c, rotate_M_2, rotated_approx_c_12)

    cv2.drawContours(blank_image_1, [rotated_approx_c_11], 0, (255, 0, 255),
                     1)  # mind the color, I don't know how to make it colorful
    cv2.drawContours(blank_image_2, [rotated_approx_c_12], 0, (255, 0, 255),
                     1)  # mind the color, I don't know how to make it colorful

    # draw the other contour
    cv2.drawContours(blank_image_1, [lst_name_j[j]], 0, (255, 0, 255), 1)  #
    cv2.drawContours(blank_image_2, [lst_name_j[j]], 0, (255, 0, 255), 1)  #

    # display_window_name = "may overlap: " + "Segment " + str(-k) + " in Fragment " + str(i) + " and " + " Segment " + str(l) + " in Fragment " + str(j)
    output_file_name_1 = str(i) + "-" + str(k) + "-" + str(j) + "-" + str(l) + "_1" + ".jpg"
    output_file_name_2 = str(i) + "-" + str(k) + "-" + str(j) + "-" + str(l) + "_2" + ".jpg"

    # for flipping the two fragments together
    if FLIP_ALLOWED_MODE:
        flipped_barycenter_ik, flipped_x_to_angle_ik = barycenter_and_x_to_angle_dict[
            (flipped_str_list_name(lst_name_i), i, k)]
        flipped_barycenter_jl, flipped_x_to_angle_jl = barycenter_and_x_to_angle_dict[
            (flipped_str_list_name(lst_name_j), j, l)]
        flipped_x_shift = flipped_barycenter_jl[0] - flipped_barycenter_ik[0]
        flipped_y_shift = flipped_barycenter_jl[1] - flipped_barycenter_ik[1]

        flipped_translate_M = np.matrix([[1, 0, flipped_x_shift], [0, 1, flipped_y_shift]])

        flipped_inverse_translate_M = np.matrix([[1, 0, -flipped_x_shift], [0, 1, -flipped_y_shift]])

        flipped_to_ang = flipped_x_to_angle_jl - flipped_x_to_angle_ik

    # test overlap
    print("about to check translated and rotated curve 1")
    if contour_area_overlapped(lst_name_j[j], rotated_approx_c_11, image, overlap_tolerance):
        print("***curve 1 fail in the overlap test***")
    else:
        match_fitness_1 = fitness_of_a_match(rotated_approx_c_11, lst_name_j[j], tag_str_i, tag_str_j, i, j)
        now = datetime.datetime.now()
        print("Current date and time using str method of datetime object:")
        print(cut_tail(str(now)))
        print("***curve 1 pass intersection test***")
        print(str_list_name(lst_name_i), str_list_name(lst_name_j))

        compatible_match_list[i][j].append(
            (str_list_name(lst_name_i), str_list_name(lst_name_j), k, l, translate_M, rotate_M_1,
             x_shift, y_shift, to_ang, original_barycenter_jl[0], original_barycenter_jl[1], match_fitness_1))

        # below is the twin match
        compatible_match_list[j][i].append(
            (str_list_name(lst_name_j), str_list_name(lst_name_i), l, k, inverse_translate_M, inverse_rotate_M_1,
             -x_shift, -y_shift, -to_ang, original_barycenter_ik[0], original_barycenter_ik[1], match_fitness_1))

        # flip the two fragments together
        if FLIP_ALLOWED_MODE:
            flipped_rotate_M_1 = cv2.getRotationMatrix2D((flipped_barycenter_jl[0], flipped_barycenter_jl[1]),
                                                         flipped_to_ang, 1)
            flipped_inverse_rotate_M_1 = cv2.getRotationMatrix2D((flipped_barycenter_ik[0], flipped_barycenter_ik[1]),
                                                                 -flipped_to_ang, 1)

            compatible_match_list[i][j].append(
                (flipped_str_list_name(lst_name_i), flipped_str_list_name(lst_name_j), k, l, flipped_translate_M,
                 flipped_rotate_M_1,
                 flipped_x_shift, flipped_y_shift, flipped_to_ang, flipped_barycenter_jl[0], flipped_barycenter_jl[1],
                 match_fitness_1))

            # below is the twin match
            compatible_match_list[j][i].append(
                (flipped_str_list_name(lst_name_j), flipped_str_list_name(lst_name_i), l, k, flipped_inverse_translate_M,
                 flipped_inverse_rotate_M_1,
                 -flipped_x_shift, -flipped_y_shift, -flipped_to_ang, flipped_barycenter_ik[0], flipped_barycenter_ik[1],
                 match_fitness_1))

        # print("compatible match list for <%d, %d> and the twin match list:\n" % (i, j))
        # print(compatible_match_list[i][j])
        # print(compatible_match_list[j][i])

        display_window_name_1 = str_list_name(lst_name_i) + "-" + str(i) + "-" + str(k) + "-" + str_list_name(
            lst_name_j) + "-" + str(j) + "-" + str(l) + "-sum_fitness-" + str(
            match_fitness_1) + "_1"
        output_file_name_1 = str_list_name(lst_name_i) + "-" + str(i) + "-" + str(k) + "-" + str_list_name(
            lst_name_j) + "-" + str(j) + "-" + str(l) + "-sum_fitness-" + str(
            match_fitness_1) + "_1.jpg"

        # if k == 7 and l == 2:
        #     cv2.imshow(display_window_name_1, blank_image_1)
        #     cv2.waitKey(0)

        cv2.imwrite("./" + compatibleDirName + "/" + output_file_name_1, blank_image_1)
        # cv2.waitKey(0)
        # cv2.destroyWindow(display_window_name_1)

    # test overlap
    print("about to check translated and rotated curve 2")
    if contour_area_overlapped(lst_name_j[j], rotated_approx_c_12, image, overlap_tolerance):
        print("***curve 2 fail in the overlap test***")
    else:
        match_fitness_2 = fitness_of_a_match(rotated_approx_c_12, lst_name_j[j], tag_str_i, tag_str_j, i, j)
        now = datetime.datetime.now()
        print("Current date and time using str method of datetime object:")
        print(cut_tail(str(now)))
        print("***curve 2 pass intersection test***")
        print(str_list_name(lst_name_i), str_list_name(lst_name_j))

        compatible_match_list[i][j].append(
            (str_list_name(lst_name_i), str_list_name(lst_name_j), k, l, translate_M, rotate_M_2,
             x_shift, y_shift, to_ang + 180, original_barycenter_jl[0], original_barycenter_jl[1], match_fitness_2))

        # below is the twin match
        compatible_match_list[j][i].append(
            (str_list_name(lst_name_j), str_list_name(lst_name_i), l, k, inverse_translate_M, inverse_rotate_M_2,
             -x_shift, -y_shift, -to_ang - 180, original_barycenter_ik[0], original_barycenter_ik[1], match_fitness_2))

        # flip the two fragments together
        if FLIP_ALLOWED_MODE:
            flipped_rotate_M_2 = cv2.getRotationMatrix2D((flipped_barycenter_jl[0], flipped_barycenter_jl[1]),
                                                         flipped_to_ang + 180, 1)
            flipped_inverse_rotate_M_2 = cv2.getRotationMatrix2D((flipped_barycenter_ik[0], flipped_barycenter_ik[1]),
                                                                 -flipped_to_ang - 180, 1)

            # flip the two fragments together
            compatible_match_list[i][j].append(
                (flipped_str_list_name(lst_name_i), flipped_str_list_name(lst_name_j), k, l, flipped_translate_M,
                 flipped_rotate_M_2,
                 flipped_x_shift, flipped_y_shift, flipped_to_ang + 180, flipped_barycenter_jl[0], flipped_barycenter_jl[1],
                 match_fitness_2))

            # below is the twin match
            compatible_match_list[j][i].append(
                (flipped_str_list_name(lst_name_j), flipped_str_list_name(lst_name_i), l, k, flipped_inverse_translate_M,
                 flipped_inverse_rotate_M_2,
                 -flipped_x_shift, -flipped_y_shift, -flipped_to_ang - 180, flipped_barycenter_ik[0],
                 flipped_barycenter_ik[1], match_fitness_2))

        # print("compatible match list for <%d, %d> and the twin match list:\n" % (i, j))
        # print(compatible_match_list[i][j])
        # print(compatible_match_list[j][i])

        display_window_name_2 = str_list_name(lst_name_i) + "-" + str(i) + "-" + str(k) + "-" + str_list_name(
            lst_name_j) + "-" + str(j) + "-" + str(l) + \
                                "-sum_fitness-" + str(match_fitness_2) + "_2"
        output_file_name_2 = str_list_name(lst_name_i) + "-" + str(i) + "-" + str(k) + "-" + str_list_name(
            lst_name_j) + "-" + str(j) + "-" + str(l) + \
                             "-sum_fitness-" + str(match_fitness_2) + "_2.jpg"

        # cv2.imshow(display_window_name_2, blank_image_2)
        cv2.imwrite("./" + compatibleDirName + "/" + output_file_name_2, blank_image_2)
        # cv2.waitKey(0)
        # cv2.destroyWindow(display_window_name_2)

        # tested_case_set.add((str_list_name(lst_name_i), str_list_name(lst_name_j), i, j, k, l))
        # no need to prevent this case, because it won't occur any more
    if FLIP_ALLOWED_MODE:
        tested_case_set.add((flipped_str_list_name(lst_name_i), flipped_str_list_name(lst_name_j), i, j, k, l))

    # print("inner angle: ", ang)
    print("to angle: %d and %d" % (to_ang, to_ang + 180))
    # cv2.waitKey(0)
    # cv2.destroyWindow(display_window_name)


def match_clusters(overlap_tolerance, img):
    compatibleDirName = "compatible-matches"
    create_or_clear_one_folder(compatibleDirName)

    for i in range(len(approx_cnts)):
        compatible_match_list.append([])
        for j in range(len(approx_cnts)):
            compatible_match_list[i].append([])

    for m in range(len(perimeter_list)):
        perimeter_i = perimeter_list[m][-1]
        for n in range(m + 1, len(perimeter_list)):
            perimeter_j = perimeter_list[n][-1]
            print("*******to deal with %d-th and %d-th of the total %d tuples*******" % (m, n, len(perimeter_list)))
            print("for " + str(perimeter_list[m]) + " and " + str(perimeter_list[n]) + ",")
            if (perimeter_j - perimeter_i) / perimeter_i > match_len_tolerance:
                print("\tlens: %.2f and %.2f, the diff is too large, the first index (%d) should be increased" % (
                    perimeter_i, perimeter_j, m))
                break
            if perimeter_list[m][1] == perimeter_list[n][1]:
                print("\tthey belong to the same fragment %d, impossible to be matched" % perimeter_list[m][1])
                continue
            print("\ttheir lens are close")
            compute_compatible_matches(m, n, overlap_tolerance, img, compatibleDirName)


def remove_twin_match(i, j, match, match_list):
    for m in range(len(match_list[j][i])):
        match_ = match_list[j][i][m]
        if match[0] == match_[1] and match[1] == match_[0] and \
                match[2] == match_[3] and match[3] == match_[2] and \
                match[6] == -match_[6] and match[7] == -match_[7] and match[8] == -match_[8]:
            match_list[j][i].remove(match_list[j][i][m])
            print("having removed the twin match successfully")
            break


def select_a_match_with_the_greatest_fitness(match_list, matched_fragment_list):
    bst_match_fitness = -1
    bst_match = []
    bst_i = -1
    bst_j = -1

    for i in range(len(approx_cnts)):
        if i in matched_fragment_list:
            continue

        for j in range(i + 1, len(approx_cnts)):
            if j in matched_fragment_list:
                continue

            for m in range(len(match_list[i][j])):
                if match_list[i][j][m][0] != "approx_cnts" and match_list[i][j][m][1] != "approx_cnts":  # prefer to use original copies
                    continue

                match_fitness = match_list[i][j][m][-1]
                if match_fitness > bst_match_fitness:
                    bst_match = match_list[i][j][m]
                    bst_match_fitness = match_fitness
                    bst_i = i
                    bst_j = j

    return bst_i, bst_j, bst_match


def select_a_match_involving_some_fragments_with_the_greatest_fitness(match_list, matched_fragment_list):
    bst_match_fitness = -1
    bst_match = []
    bst_i = -1
    bst_j = -1

    print("in select involving function, matched_fragment_list: \n", matched_fragment_list)

    list_of_indexes_where_item_to_remove = []

    for j in range(len(matched_fragment_list)):
        matched_fragment_id = matched_fragment_list[j]
        print("1. matched_fragment_id: ", matched_fragment_id)
        print("its flip state is ", fragment_flipped[matched_fragment_id])
        for i in range(matched_fragment_id):
            if i in matched_fragment_list:
                print("nothing to be done because %d has been matched" % i)
                continue
            print("to deal with %d" % i)
            for m in range(len(compatible_match_list[i][matched_fragment_id])):

                match = compatible_match_list[i][matched_fragment_id][m]

                print("considering ", compatible_match_list[i][matched_fragment_id][m])

                # should be compatible with previous selections
                if match_list[i][matched_fragment_id][m][1] == "flipped_approx_cnts" and fragment_flipped[
                    matched_fragment_id] == False:  # the flip state of a matched fragment cannot be changed
                    print("1. the latter %d cannot be flipped, so this match causes conflicts" % matched_fragment_id)

                    # compatible_match_list[i][matched_fragment_id].remove(compatible_match_list[i][matched_fragment_id][m])
                    list_of_indexes_where_item_to_remove.append(m)
                    print("having collected this match successfully")

                    remove_twin_match(i, matched_fragment_id, match, match_list)

                    # cv2.waitKey(0)
                    continue

                if match_list[i][matched_fragment_id][m][1] == "approx_cnts" and fragment_flipped[
                    matched_fragment_id] == True:  # cannot change the flip state of a matched fragment
                    print("1. the latter %d has been flipped, so this match causes conflicts" % matched_fragment_id)

                    # compatible_match_list[i][matched_fragment_id].remove(compatible_match_list[i][matched_fragment_id][m])
                    list_of_indexes_where_item_to_remove.append(m)
                    print("having collected this match successfully")
                    remove_twin_match(i, matched_fragment_id, match, match_list)

                    # cv2.waitKey(0)
                    continue

                print("pass flip tests")
                match_fitness = match_list[i][matched_fragment_id][m][-1]
                print("1. for match <%d, %d>, match:\n" % (i, matched_fragment_id),
                      match_list[i][matched_fragment_id][m])
                if match_fitness > bst_match_fitness:
                    bst_match = match_list[i][matched_fragment_id][m]
                    bst_match_fitness = match_fitness
                    bst_i = i
                    bst_j = matched_fragment_id

            # delete those collected elements
            print("1. to delete from\n", match_list[i][matched_fragment_id])
            print("to delete elements at indexes \n", sorted(list_of_indexes_where_item_to_remove, reverse=True))
            for ind in sorted(list_of_indexes_where_item_to_remove, reverse=True):
                print("1. to delete the element at index %d" % ind)
                print("to delete\n", match_list[i][matched_fragment_id][ind])
                del match_list[i][matched_fragment_id][ind]
                print("1. having just deleted it because of incompatible flip info:\n")
            list_of_indexes_where_item_to_remove = []

    for i in range(len(matched_fragment_list)):
        matched_fragment_id = matched_fragment_list[i]
        print("2. matched_fragment_id: ", matched_fragment_id)
        print("its flip state is ", fragment_flipped[matched_fragment_id])
        for j in range(matched_fragment_id + 1, len(approx_cnts)):
            if j in matched_fragment_list:
                print("nothing to be done because %d has been matched" % j)
                continue
            print("to deal with %d" % j)
            for m in range(len(compatible_match_list[matched_fragment_id][j])):

                match = compatible_match_list[matched_fragment_id][j][m]

                print("considering ", compatible_match_list[matched_fragment_id][j][m])

                # should be compatible with previous selections
                if match_list[matched_fragment_id][j][m][0] == "flipped_approx_cnts" and fragment_flipped[
                    matched_fragment_id] == False:
                    print("2. the former %d cannot be flipped, so this match causes conflicts" % matched_fragment_id)

                    # compatible_match_list[matched_fragment_id][j].remove(compatible_match_list[matched_fragment_id][j][m])
                    list_of_indexes_where_item_to_remove.append(m)
                    print("having collected this match successfully")
                    remove_twin_match(matched_fragment_id, j, match, match_list)

                    # cv2.waitKey(0)
                    continue

                if match_list[matched_fragment_id][j][m][0] == "approx_cnts" and fragment_flipped[
                    matched_fragment_id] == True:
                    print("2. the former %d has been flipped, so this match causes conflicts" % matched_fragment_id)

                    # compatible_match_list[matched_fragment_id][j].remove(compatible_match_list[matched_fragment_id][j][m])
                    list_of_indexes_where_item_to_remove.append(m)
                    print("having collected this match successfully")
                    remove_twin_match(matched_fragment_id, j, match, match_list)

                    # cv2.waitKey(0)
                    continue

                print("pass flip tests")
                print("2. for match <%d, %d>, match:\n" % (matched_fragment_id, j),
                      match_list[matched_fragment_id][j][m])
                match_fitness = match_list[matched_fragment_id][j][m][-1]
                if match_fitness > bst_match_fitness:
                    bst_match = match_list[matched_fragment_id][j][m]
                    bst_match_fitness = match_fitness
                    bst_i = matched_fragment_id
                    bst_j = j

            # delete those collected elements
            print("2. to delete from\n", match_list[matched_fragment_id][j])
            print("to delete elements at indexes \n", sorted(list_of_indexes_where_item_to_remove, reverse=True))
            for ind in sorted(list_of_indexes_where_item_to_remove, reverse=True):
                print("2. to delete the element at index %d" % ind)
                del match_list[matched_fragment_id][j][ind]
                print("2. having just deleted because of incompatible flip info:\n")
            list_of_indexes_where_item_to_remove = []

    now = datetime.datetime.now()
    print("Current date and time using str method of datetime object:")
    print(str(now))
    print("Finally, in greedy selection, having obtained <%d, %d>:\n" % (bst_i, bst_j), bst_match)
    return bst_i, bst_j, bst_match


def extend_partial_picture_with_one_match(i, j, match, img, epsilon):
    # precondition: i < j

    if i == -1 or j == -1:
        return False

    if i not in matched_fragments:
        print("1. for fragment pair <%d, %d>, match:\n" % (i, j), match)
        _, _, _, _, T_M, R_M, x_shift, y_shift, to_angle, x_r_center, y_r_center, _ = match

        temp_list = partial_picture[i]
        temp_list_for_movement = partial_picture_for_movement[i]

        partial_picture[i] = [T_M, R_M] + partial_picture[j]
        partial_picture_for_movement[i] = [(x_shift, y_shift, to_angle, x_r_center, y_r_center)] + \
                                          partial_picture_for_movement[j]

        print("about to call match-causes-overlap")
        print("currently matched_fragments:\n", matched_fragments)

        # if match_causes_overlap(i, matched_fragments, img, epsilon):
        print("current_matched_fragment_pile:\n", current_matched_fragment_pile)
        if match_causes_overlap(i, current_matched_fragment_pile, img, epsilon):
            partial_picture[i] = temp_list
            partial_picture_for_movement[i] = temp_list_for_movement
            print("fragment %d cannot be placed" % i)
            return False
        current_matched_fragment_pile.append(i)
        matched_fragments.append(i)

    if j not in matched_fragments:
        print("2. for fragment pair <%d, %d>, match:\n" % (i, j), match)
        print(compatible_match_list[j][i])
        for m in range(len(compatible_match_list[j][i])):
            str_lst_name1, str_lst_name2, l, k, inverse_T_M, inverse_R_M, x_shift, y_shift, to_angle, x_r_center, y_r_center, _ \
                = compatible_match_list[j][i][m]
            if str_lst_name2 == match[0] and str_lst_name1 == match[1] \
                    and k == match[2] and l == match[3] \
                    and x_shift == -match[6] and y_shift == -match[7] and to_angle == -match[8]:
                print("having found inverse transformation for moving fragments")
                break

        # print("2. for fragment pair <%d, %d>, match:\n" % (j, i), compatible_match_list[j][i])

        temp_list = partial_picture[j]
        temp_list_for_movement = partial_picture_for_movement[j]

        partial_picture[j] = [inverse_T_M, inverse_R_M] + partial_picture[i]
        partial_picture_for_movement[j] = [(x_shift, y_shift, to_angle, x_r_center, y_r_center)] + \
                                          partial_picture_for_movement[i]

        print("about to call match-causes-overlap")
        print("currently matched_fragments:\n", matched_fragments)

        # if match_causes_overlap(j, matched_fragments, img, epsilon):
        print("current_matched_fragment_pile:\n", current_matched_fragment_pile)
        if match_causes_overlap(j, current_matched_fragment_pile, img, epsilon):
            partial_picture[j] = temp_list
            partial_picture_for_movement[j] = temp_list_for_movement
            print("fragment %d cannot be placed" % j)
            return False
        current_matched_fragment_pile.append(j)
        matched_fragments.append(j)

    print("fragment %d placed, partial picture:" % i)
    print(partial_picture)
    print("fragment %d placed, partial picture for movement:" % i)
    print(partial_picture_for_movement)
    return True


def contour_in_partial_picture(index, flipped):
    transformation_list = partial_picture[index]

    if not flipped[index]:
        c = np.array(approx_cnts[index])  # bad implementation
    else:
        c = np.array(flipped_approx_cnts[index])

    c1 = np.array(approx_cnts[index])  # bad implementation

    # apply transformations
    for t in range(len(transformation_list)):
        cv2.transform(c, transformation_list[t], c1)
        c = c1

    return c


def match_causes_overlap(index, matched_fragment_list, img, epsilon):
    print("checking overlap for %d..." % index)
    print("matched_fragment_list: \n", matched_fragment_list)
    current_fragment_contour = contour_in_partial_picture(index, fragment_flipped)
    for i in range(len(matched_fragment_list)):
        matched_fragment_id = matched_fragment_list[i]
        matched_fragment_contour = contour_in_partial_picture(matched_fragment_id, fragment_flipped)
        print("current_fragment_contour (index %d):" % index)
        # print(current_fragment_contour)
        print("matched_fragment_contour (matched_fragment_id %d):" % matched_fragment_id)
        # print(matched_fragment_contour)
        print("their overlap will be checked")
        if contour_area_overlapped(current_fragment_contour, matched_fragment_contour, img, epsilon):
            print("match causes overlap: %d and %d" % (matched_fragment_id, index))
            return True
        print("no overlap detected between %d and %d" % (matched_fragment_id, index))
    print("the new match is compatible with existing matches")
    return False


def draw_final_image_list(p_picture, final_image_lst, flipped, img, pileDirName):
    global cnts

    print("just enterd draw_final_image_list, cnt_num: ", len(cnts))
    print("in draw_final_image_list, p_picture: ")

    print(p_picture)

    print("to draw the matched fragment piles")

    # iterate curves
    print("iterating the curves")

    for k in range(len(final_image_lst)):

        print("the %d-th round" % (k + 1))

        to_draw_list = final_image_lst[k]
        print("to-draw-list:\n", to_draw_list)

        blank_image = np.zeros(enlarged_by_a_factor(img.shape[0:2], 2))

        print("*************")
        for i in range(len(approx_cnts)):

            if i not in to_draw_list:
                continue

            print("for fragment: ", i)

            print("p_picture[%d]: " % i)
            print(p_picture[i])
            print("below is the transformation. \n[(x_shift, y_shift, rotate_angle, x_r_center, y_r_center)]:")
            print(partial_picture_for_movement[i])

            transformation_list = p_picture[i]
            # print("transformation list: ")
            # print(transformation_list)

            # c = cv2.approxPolyDP(approx_cnts[i], 0.1, True)  # bad implementation
            if not flipped[i]:

                # print("cnt_num: ", len(cnts))
                # print("approx_cnt_num: ", len(approx_cnts))

                # c = np.array(approx_cnts[i])  # bad implementation
                c = np.array(cnts[i])
                print("start from the original copy")
                print("barycenter: ", fragment_barycenter_list[i])
                print("it is displayed in " + SINGLE_FRAGMENT_NAME_PREFIX + "_" + str(i) + ".png")
            else:
                # c = np.array(flipped_approx_cnts[i])
                c = np.array(flipped_cnts[i])
                print("start from the flipped copy")
                print("barycenter: ", flipped_fragment_barycenter_list[i])

            c1 = np.array(cnts[i])  # bad implementation

            # apply transformations
            for t in range(len(transformation_list)):
                cv2.transform(c, transformation_list[t], c1)
                c = c1

            # translate_matrix = np.matrix([[1, 0, img.shape[0:2][0]], [0, 1, img.shape[0:2][1]]])
            # cv2.transform(c, translate_matrix, c1)
            # c = c1

            if not fragment_flipped[i]:
                cv2.drawContours(blank_image, [c], 0, (255, 0, 255), 1)
            else:
                cv2.drawContours(blank_image, [c], 0, (127, 255, 63), 1)

        print("having iterated the curves")
        display_window_name = "matched_fragment_pile_" + str(k + 1)
        # cv2.imshow(display_window_name, blank_image)
        # cv2.imwrite(display_window_name + ".jpg", blank_image)
        cv2.imwrite(pileDirName + "/" + display_window_name + ".jpg", blank_image)
        # cv2.waitKey(0)


def element_in_two_level_list(e, two_level_lst):
    for i in range(len(two_level_lst)):
        if e in two_level_lst[i]:
            return True
    return False


if __name__ == "__main__":

    image = cv2.imread(input_fragment_file_name)

    print("having just read an image")

    initialize_approx_cnts(image)

    print("in main function, just after initialization, cnt_num: ", len(cnts))

    cv2.waitKey(0)

    match_clusters(overlap_tolerance, image)

    for i in range(len(approx_cnts)):
        partial_picture.append([])
        partial_picture_for_movement.append([])

    best_i, best_j, best_match \
        = select_a_match_with_the_greatest_fitness(compatible_match_list, matched_fragments)  # starting

    fragment_flipped = [False] * len(approx_cnts)

    print("having selected %d and %d as the first match" % (best_i, best_j))

    _, _, _, _, Translate_M, rotate_M, x_shift, y_shift, to_angle, x_r_center, y_r_center, _ = best_match

    first_placed_fragment_id = best_j

    partial_picture[best_i].extend([Translate_M, rotate_M])
    partial_picture_for_movement[best_i].append((x_shift, y_shift, to_angle, x_r_center, y_r_center))

    print("partial picture:\n", partial_picture)
    print("partial picture for movement:\n", partial_picture_for_movement)

    print("initial_best_match: <%d, %d>\n" % (best_i, best_j), best_match)

    current_matched_fragment_pile = [best_i, best_j]
    matched_fragments.extend([best_i, best_j])

    if FLIP_ALLOWED_MODE:
        if best_match[0] == "flipped_approx_cnts":
            fragment_flipped[best_i] = True
        if best_match[1] == "flipped_approx_cnts":
            fragment_flipped[best_j] = True

    compatible_match_list[best_i][best_j] = []
    compatible_match_list[best_j][best_i] = []

    print("current_matched_fragment_pile:\n", current_matched_fragment_pile)

    best_i, best_j, best_match = \
        select_a_match_involving_some_fragments_with_the_greatest_fitness(compatible_match_list,
                                                                        current_matched_fragment_pile)

    print("best_match_for_growing: <%d, %d>\n" % (best_i, best_j), best_match)

    cv2.waitKey(0)

    final_image_list = []

    print("in main function, before entering the big loop, cnt_num: ", len(cnts))

    while True:

        # keep selecting a match with the greatest fitness until extension is done, or no compatible matches exist
        # during this procedure, unsuitable matches will be deleted:
        while not extend_partial_picture_with_one_match(best_i, best_j, best_match, image, overlap_tolerance):

            if best_i != -1 and best_j != -1:

                print("incompatible for <%d, %d>, should select another match" % (best_i, best_j))

                # remove unsuitable matches
                print("to remove\n", best_match)
                print("from\n", compatible_match_list[best_i][best_j])

                compatible_match_list[best_i][best_j].remove(best_match)
                print("match_num: ", len(compatible_match_list[best_j][best_i]))
                print(compatible_match_list[best_j][best_i])
                # cv2.waitKey(0)
                for m in range(len(compatible_match_list[best_j][best_i])):
                    print("m: ", m)
                    print("in the for loop, match_num: ", len(compatible_match_list[best_j][best_i]))
                    print(compatible_match_list[best_j][best_i][m])
                    lst_name2, lst_name1, l, k, _, _, x_shift, y_shift, to_angle, _, _, _ = \
                        compatible_match_list[best_j][best_i][m]
                    if lst_name1 == best_match[0] and lst_name2 == best_match[1] \
                            and k == best_match[2] and l == best_match[3] \
                            and x_shift == -best_match[6] and y_shift == -best_match[7] and to_angle == -best_match[8]:
                        print("having found inverse transformation and will remove")
                        compatible_match_list[best_j][best_i].remove(compatible_match_list[best_j][best_i][m])
                        break
                # cv2.waitKey(0)
            # select another greedy match
            best_i, best_j, best_match = select_a_match_involving_some_fragments_with_the_greatest_fitness(
                compatible_match_list, current_matched_fragment_pile)  # growing
            # cv2.waitKey(0)
            print("having obtained %d and %d for attempting" % (best_i, best_j), best_match)
            # cv2.waitKey(0)

            # if fail to select such a match
            if best_i == -1:
                print("no matches selected")
                break
            print("having obtained a valid pair, <%d, %d>" % (best_i, best_j))

        if best_i != -1:  # succeed in extending current pile with a piece
            if FLIP_ALLOWED_MODE:
                if best_match[0] == "flipped_approx_cnts":
                    fragment_flipped[best_i] = True
                if best_match[1] == "flipped_approx_cnts":
                    fragment_flipped[best_j] = True

            compatible_match_list[best_i][best_j] = []
            compatible_match_list[best_j][best_i] = []
            print("having deleted the qualified pair <%d, %d> from cand match database" % (best_i, best_j), best_match)
            # cv2.waitKey(0)
            best_i, best_j, best_match = select_a_match_involving_some_fragments_with_the_greatest_fitness(
                compatible_match_list, current_matched_fragment_pile)

        # while fail to grow:
        while best_i == -1:

            final_image_list.append(current_matched_fragment_pile)  # put away the current pile

            print("final_image_list: \n", final_image_list)

            print("no adjacent matches selected")

            # select another starting match
            best_i, best_j, best_match = select_a_match_with_the_greatest_fitness(compatible_match_list,
                                                                                matched_fragments)  # starting

            print("have selecting a new starting point, the match is: <%d, %d>\n" % (best_i, best_j), best_match)

            if best_i == -1:  # no starting matches available, the construction procedure should be ended
                break

            # a new pile of fragments
            current_matched_fragment_pile = [best_i, best_j]

            _, _, _, _, Translate_M, rotate_M, x_shift, y_shift, to_angle, x_r_center, y_r_center, _ = best_match

            partial_picture[best_i].extend([Translate_M, rotate_M])
            partial_picture_for_movement[best_i].append((x_shift, y_shift, to_angle, x_r_center, y_r_center))

            matched_fragments.extend([best_i, best_j])

            print("after initializing a new starting point, match_fragments: \n", matched_fragments)

            if FLIP_ALLOWED_MODE:
                if best_match[0] == "flipped_approx_cnts":
                    fragment_flipped[best_i] = True
                if best_match[1] == "flipped_approx_cnts":
                    fragment_flipped[best_j] = True

            compatible_match_list[best_i][best_j] = []
            compatible_match_list[best_j][best_i] = []

            best_i, best_j, best_match = select_a_match_involving_some_fragments_with_the_greatest_fitness(
                compatible_match_list, current_matched_fragment_pile)

        if best_i == -1:  # no starting matches available
            print("Puzzle ended!")
            break

        # draw_partial_picture_in_one_image(partial_picture, image)

    # fill in remaining elements
    for i in range(len(approx_cnts)):
        if element_in_two_level_list(i, final_image_list):
            continue
        final_image_list.append([i])

    print("end of the program, final_image_list: \n", final_image_list)
    # draw_partial_picture_in_one_image(partial_picture, image)
    # print("matched_fragment_list:\n", matched_fragments)
    # cv2.waitKey(0)
    pileDirName = "pile-matches"
    create_or_clear_one_folder(pileDirName)

    draw_final_image_list(partial_picture, final_image_list, fragment_flipped, image, pileDirName)
    print("final-image-list:\n", final_image_list)
    print("fragment flipped:\n", fragment_flipped)



