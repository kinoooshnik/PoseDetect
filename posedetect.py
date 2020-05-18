import sys
import pickle
from collections import defaultdict

import subprocess
import datetime
import os
import numpy as np

sys.path.append('detectron2/projects/DensePose')
from densepose.data.structures import DensePoseResult


def get_n(data):
    return int(sum([1 if x >= 0.95 else 0 for x in data['scores']]))


def get_info(data, N):
    bbox_xyxy = []
    result_encoded = []
    iuv_arr = []
    for n in range(N):
        bbox_xyxy.append(data['pred_boxes_XYXY'][n])
        result_encoded.append(data['pred_densepose'].results[n])
        iuv_arr.append(DensePoseResult.decode_png_data(*result_encoded[n]))
        iuv_arr[len(iuv_arr) - 1] = iuv_arr[len(iuv_arr) - 1].astype(np.int16, copy=False)
    return bbox_xyxy, result_encoded, iuv_arr


def get_parts():
    center_back_dots = [(130, x) for x in range(10, 250, 25)]
    left_back_dots = [(65, x) for x in range(10, 250, 25)]
    right_back_dots = [(195, x) for x in range(10, 250, 25)]

    center_front_dots = [(130, x) for x in range(10, 250, 25)]
    left_front_dots = [(65, x) for x in range(10, 250, 25)]
    right_front_dots = [(195, x) for x in range(10, 250, 25)]

    return [(1, left_back_dots), (1, center_back_dots), (1, right_back_dots),
            (2, left_front_dots), (2, center_front_dots), (2, right_front_dots)], ['r', 'b', 'g', 'c', 'm', 'y']


def calculate_dots(parts, N, iuv_arr, bbox_xyxy):
    u = []
    v = []
    for n in range(N):
        i = 0
        for part, area_dots_list in enumerate(parts):
            u.append(defaultdict(dict))
            v.append(defaultdict(dict))
            #             print(area_dots_list)
            area, dots_list = area_dots_list
            #             for area, dots_list in area_dots_list:
            for i, dot in enumerate(dots_list):
                is_area = (iuv_arr[n][0, :, :] == area)
                length = abs((iuv_arr[n][2, :, :] - dot[0]) * is_area) + abs((iuv_arr[n][1, :, :] - dot[1]) * is_area)
                xy = np.where(np.logical_and(length < 10, length != 0))
                xy = (xy[0] + float(bbox_xyxy[n][1]), xy[1] + float(bbox_xyxy[n][0]))
                if len(xy[0]) != 0:
                    u[n][part][i] = sum(xy[0]) / len(xy[0])
                    v[n][part][i] = sum(xy[1]) / len(xy[1])
                else:
                    u[n][part][i], v[n][part][i] = 0, 0
    return u, v


def calculate_angle(p1, p2, p3):
    """Считает угол между 3 точками.

    Строит 2 вектора (p1, p2) и (p3, p2) и считает между ними угол. Если одна из точек (0, 0), то возвращает 0.

    Args:
        p1 list: массив из 2 элементов x и y координат точки.
        p2 list: массив из 2 элементов x и y координат точки.
        p3 list: массив из 2 элементов x и y координат точки.

    Returns:
        float: угол между точками в градусах
   """
    if (0, 0) in [p1, p2, p3]:
        return 0
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = p2 - p1
    v2 = p3 - p2
    cos = (v1.dot(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos))


def calculate_person_angles(dots, N):
    angles = np.zeros(shape=(N, 8), dtype=float)
    for n in range(N):
        for i in range(len(dots[n]) - 2):
            angles[n][i] = calculate_angle(dots[n][i], dots[n][i + 1], dots[n][i + 2])
    return angles


def check_angles(angles, N):
    IDEAL_ANGLES_STRAIGHT = [12.45692969, 14.59143796, 16.21804211, 1.03094231, 11.7492887, 4.5544555, 6.01517189,
                             4.16746659]
    IDEAL_ANGLES_CURVED = [9.789623, 22.60836205, 12.63835315, 7.11686044, 1.29802535, 3.04290574, 7.19902816,
                           12.48225848]

    is_ok = []
    for n in range(N):
        if sum(angles[n]) <= 30:
            is_ok.append(True)
        elif sum(abs(angles[n] - IDEAL_ANGLES_STRAIGHT)) <= 30:
            is_ok.append(True)
        elif sum(abs(angles[n] - IDEAL_ANGLES_CURVED)) <= 30:
            is_ok.append(True)
        else:
            is_ok.append(False)
    return is_ok


def choose_best_dots(u, v, N, center_ids):
    person_dots = []
    for n in range(N):
        parts_len = {}
        for part, coord_u in u[n].items():
            parts_len[part] = np.count_nonzero(list(coord_u.values()))
            if part in center_ids:
                parts_len[part] += 2
        max_part = max(parts_len, key=parts_len.get)
        person_dots.append({x: (u[n][max_part][x], v[n][max_part][x]) for x in range(len(u[n][max_part]))})
    return person_dots


def generate_dump(densepose_path, img_path):
    timeshtamp = int(datetime.datetime.now().timestamp() * 10000)
    dump_file = 'result_{}.pkl'.format(timeshtamp)
    args = {'densepose_path': densepose_path, 'img_path': img_path, 'dump_file': dump_file}

    bash_command = 'python3 {densepose_path}apply_net.py dump {densepose_path}configs/densepose_rcnn_R_50_FPN_WC1_s1x.yaml {densepose_path}model_final_289019.pkl {img_path} --output {dump_file} -v --opts MODEL.DEVICE cpu'
    bash_command = bash_command.format(**args)

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    print(output.decode("utf-8"))

    if process.returncode != 0:
        raise Exception(output.decode("utf-8"))

    with open(dump_file, 'rb') as file:
        dump = pickle.load(file)
    os.remove(dump_file)

    return dump

def check_pose(img_path):
    DENSEPOSE_FOLDER = 'detectron2/projects/DensePose/'
    # сгенерировать и загрузить результат работы DensePose
    data = generate_dump(DENSEPOSE_FOLDER, img_path)
    for i, img_data in enumerate(data):
        # получить количество распознанных людей с вероятностью >= 0.95
        N = get_n(img_data)

        # получить bounding box и координаты точек модели человека на фото
        bbox_xyxy, _, iuv_arr = get_info(img_data, N)

        # получить координаты точек на модели, которые нужно найти на фото
        parts, colors = get_parts()

        # получить координаты найденных точек на фото
        u, v = calculate_dots(parts, N, iuv_arr, bbox_xyxy)

        # выбирает "лучшие" точки для распознования (отдает приоритет центральным)
        dots = choose_best_dots(u, v, N, [1, 4])

        # считает углы между точками
        angles = calculate_person_angles(dots, N)

        # проверяет правильность углов между точками
        check = check_angles(angles, N)

        return dots, check
