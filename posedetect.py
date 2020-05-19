import sys
import pickle
from collections import defaultdict

import math
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


def rotate(origin, xs, ys, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin

    qx, qy = np.zeros_like(xs), np.zeros_like(ys)
    for i in range(len(xs)):
        qx[i] = ox + math.cos(angle) * (xs[i] - ox) - math.sin(angle) * (ys[i] - oy)
        qy[i] = oy + math.sin(angle) * (xs[i] - ox) + math.cos(angle) * (ys[i] - oy)
    return qx, qy


def check_dots(dots, N):
    is_ok = []
    for n in range(N):
        # разделяем точки на 2 массива
        x = np.array([x[0] for x in dots[n].values() if x[0] != 0])
        y = np.array([x[1] for x in dots[n].values() if x[1] != 0])

        # центрируем точки
        x -= (x.max() - x.min()) / 2 + x.min()
        y -= (y.max() - y.min()) / 2 + y.min()

        # аппроксимируем точки прямой
        m, c = np.polyfit(x, y, 1)

        # находим поворот этой прямой относительно оси ox
        angle = np.arctan(m)

        # поворачиваем точки на найденный угол
        x, y = rotate((0, c), x, y, -angle)

        # масштабирем так, чтобы точки лежали на интервале от -1 до 1
        divider = abs(max(x.min(), x.max(), key=abs))
        x /= divider
        y /= divider

        # опять центрируем их
        x -= (x.max() - x.min()) / 2 + x.min()

        # аппроксимируем полиномом 3-й степени
        m3, m2, m, c = np.polyfit(x, y, 3)

        # создаем объект полинома для расчетов
        p = np.poly1d([m3, m2, m, 0])

        # находим конри полинома
        roots = p.r.tolist()
        roots.extend([1.0, -1.0])
        roots = sorted([x for x in roots if -1 <= x <= 1])

        # находим интеграл
        integ = p.integ()

        # считаем площадь под кривой от -1 до 1
        S = 0
        for i in range(len(roots) - 1):
            S += abs(integ(roots[i + 1]) - integ(roots[i]))

        # добавляем результат в массив
        is_ok.append(True if S <= 0.16 else False)

    return is_ok


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

        # проверяет точки на правильность позы
        check = check_dots(dots, N)

        return dots, check
