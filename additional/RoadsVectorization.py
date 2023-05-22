import cv2
import os
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import numpy as np
from matplotlib.pylab import plt
from . import sknw
from itertools import tee
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
import pickle
from shapely.wkt import loads
from PIL import Image
from shapely.geometry import mapping
from shapely.wkt import loads
import fiona
import shapely

linestring = "LINESTRING {}"


class RoadsVectorization:
    def vectorise(self, full_name_img, dir_image, dir_to_save, fn_wkt='wkt.dat', fn_nodes='nodes.txt',
                  fn_foo='graph-foo.png', threshold=.3, pixel_objects=500, pixel_holes=50):
        name_img = os.path.splitext(full_name_img)[0]
        add_small = True
        ske = self.__make_skeleton(dir_image, threshold, pixel_objects, pixel_holes)

        im = Image.fromarray(ske * 255).convert("L")
        im.save("skeletonized.jpg")

        G = sknw.build_sknw(ske, multi=True)

        self.__remove_small_terminal(G)
        node_lines = self.__graph2lines(G)

        node = G.nodes
        deg = G.degree()
        wkt = []
        terminal_points = [i for i, d in deg if d == 1]

        terminal_lines = {}
        vertices = []
        for w in node_lines:
            coord_list = []
            additional_paths = []
            for s, e in self.__pairwise(w):  # start - end, new_start(=end) - new_end,...
                vals = self.__flatten([[v] for v in G[s][e].values()])
                for ix, val in enumerate(vals):  # ix - количество линий - 1, val - координаты точек линии

                    s_coord, e_coord = node[s]['o'], node[e]['o']

                    pts = val.get('pts', [])  # points
                    if s in terminal_points:  # if start in terminal
                        terminal_lines[s] = (s_coord, e_coord)  # start note - (start-coord, end-coord)
                    if e in terminal_points:  # if end in terminal
                        terminal_lines[e] = (e_coord, s_coord)  # end note - (end-coord, start-coord)

                    # pts - смежные точки, s - id стартовой, e - id конечной, s/e_coord - координаты
                    ps = self.__add_direction_change_nodes(pts, s, e, s_coord, e_coord)  # аппроксимация

                    if len(ps.shape) < 2 or len(ps) < 2:  # если одна точка то продолжить
                        continue

                    if len(ps) == 2 and np.all(ps[0] == ps[1]):  # если 2 точки то продолжить
                        continue

                    line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in
                                    ps]  # преобразование массива координат в строку
                    if ix == 0:
                        coord_list.extend(line_strings)  # добавить координаты узлов линии в массив
                    else:
                        additional_paths.append(line_strings)  # если это добавочный путь, то в другой

                    vertices.append(ps)

            if not len(coord_list):
                continue
            segments = self.__remove_duplicate_segments(coord_list)  # удаление повторяющихся фрагментов линий
            # преобразование сегментов в формат Well-known text WKT
            for coord_list in segments:
                if len(coord_list) > 1:
                    line = '(' + ", ".join(coord_list) + ')'
                    wkt.append(linestring.format(line))
            for line_strings in additional_paths:
                line = ", ".join(line_strings)
                line_rev = ", ".join(reversed(line_strings))
                for s in wkt:
                    if line in s or line_rev in s:
                        break
                else:
                    wkt.append(linestring.format('(' + line + ')'))
        if add_small and len(terminal_points) > 1:
            wkt.extend(self.__add_small_segments(G, terminal_points, terminal_lines))  # соединение близлежащих точек
        # вывод и сохранение
        plt.rcParams["figure.dpi"] = 100
        plt.imshow(ske, cmap='gray')

        lines = [loads(l) for l in wkt]
        for ls in lines:
            x, y = ls.coords.xy
            plt.plot(x, y, 'green')

        path_to_wkt = os.path.join(dir_to_save, name_img)
        if not os.path.exists(path_to_wkt):
            os.makedirs(path_to_wkt)
        path_to_nodes = os.path.join(dir_to_save, name_img)
        if not os.path.exists(path_to_nodes):
            os.makedirs(path_to_nodes)
        path_to_foo = os.path.join(dir_to_save, name_img)
        if not os.path.exists(path_to_foo):
            os.makedirs(path_to_foo)

        with open(os.path.join(path_to_wkt, fn_wkt), 'wb') as f:
            pickle.dump(wkt, f)

        ps = np.array([node[i]['o'] for i in G.nodes])
        np.savetxt(os.path.join(path_to_nodes, fn_nodes), ps)
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

        # title and show
        plt.title('Roads Graph')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(os.path.join(path_to_foo, fn_foo), bbox_inches='tight', dpi=200)
        plt.show()

        return_images = []
        new_name = "skeletonized_" + full_name_img
        new_path = os.path.join(dir_to_save, name_img, new_name)
        im = Image.fromarray(ske * 255).convert("L")
        im.save(new_path)
        return_images.append((new_name, new_path))
        return_images.append((name_img + '_' + fn_foo, os.path.join(path_to_foo, fn_foo)))
        return return_images

    def save_shp(self, path_to_rects, path_to_save, offset=None, filename='shpfile.shp', fn_rects='wkt.dat'):
        with open(os.path.join(path_to_rects, fn_rects), 'rb') as fp:
            wkt = pickle.load(fp)

        wkt_to_shp(wkt, path_to_save, offset)

    def __make_skeleton(self, dir_image, thresh, pixel_objects, pixel_holes):
        "open and skeletonize"
        img = cv2.imread(dir_image, cv2.IMREAD_GRAYSCALE)

        img = self.__preprocess(img, thresh, pixel_objects, pixel_holes)
        if not np.any(img):
            return None, None
        ske = skeletonize(img).astype(np.uint16)
        return ske

    def __pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def __remove_sequential_duplicates(self, seq):
        'Удаление повторяющихся точек в списке координат'
        res = [seq[0]]
        for elem in seq[1:]:
            if elem == res[-1]:
                continue
            res.append(elem)
        return res

    def __remove_duplicate_segments(self, seq):
        'Метод удаления повторяющихся фрагментов линий'
        seq = self.__remove_sequential_duplicates(seq)  # удаление повторяющихся точек
        segments = set()
        split_seg = []
        res = []
        for idx, (s, e) in enumerate(self.__pairwise(seq)):  # idx - номер отрезка линии
            if (s, e) not in segments and (
            e, s) not in segments:  # если линия не в сегментах то добавить края в сегменты
                segments.add((s, e))
                segments.add((e, s))
            else:
                split_seg.append(idx + 1)  # иначе добавить индикатор отрезка для сплита
        for idx, v in enumerate(split_seg):  # idx - номер повт сегмента, v - номер повторяющегося отрезка
            if idx == 0:  # сначала
                res.append(seq[:v])  # добавляем в результат до номера повторяющегося отрезка
            if idx == len(split_seg) - 1:  # если повторяющийся отрезок последний
                res.append(seq[v:])  # добавить в результат точки после отрезка
            else:  # иначе
                s = seq[split_seg[idx - 1]:v]  # добавить в результат отрезки с предыдущего повт. сегмента до текущего
                if len(s) > 1:  # если это отрезок, а не точка
                    res.append(s)  # добавить в результат
        if not len(split_seg):  # если нет повт. сегментов
            res.append(seq)  # добавить в результат
        return res

    def __flatten(self, l):
        return [item for sublist in l for item in sublist]

    def __get_angle(self, p0, p1=np.array([0, 0]), p2=None):
        """ compute angle (in degrees) for p0p1p2 corner
        Inputs:
            p0,p1,p2 - points in the form of [x,y]
        """
        if p2 is None:
            p2 = p1 + np.array([1, 0])
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

    def __preprocess(self, img, thresh, pixels_objects=500, pixels_holes=150, remove_holes: bool = True):
        img = (img > (255 * thresh)).astype(bool)  # 1.23.1

        im = Image.fromarray(img)
        im.save("threshold.jpg")

        remove_small_objects(img, pixels_objects)
        if (remove_holes): remove_small_holes(img, pixels_holes)

        im = Image.fromarray(img)
        im.save("removed_holes.jpg")

        # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))
        return img

    def __graph2lines(self, G):
        node_lines = []
        edges = list(G.edges())  # считываем ребра
        if len(edges) < 1:
            return []
        prev_e = edges[0][1]  # предыдущая конечная точка = конечной первого ребра
        current_line = list(edges[0])  # текущее первое ребро в рассматриваемые
        added_edges = {edges[0]}  # и в рассмотренные
        for s, e in edges[1:]:  # далее рассматриваем после 0 ребра
            if (s, e) in added_edges:  # если ребро рассмотрено смотрим следующее
                continue
            if s == prev_e:  # если это продолжение предыдущего
                current_line.append(e)  # добавляем точку в линию
            else:
                node_lines.append(current_line)  # добавляем линию в массив
                current_line = [s, e]  # меняем текущую
            added_edges.add((s, e))  # добавляем ребро в рассмотренные
            prev_e = e  # смещаем точку
        if current_line:  # если остались линии
            node_lines.append(current_line)  # добавляем в результат
        return node_lines  # возвращаем

    def __line_points_dist(self, line1, pts):
        return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

    def __remove_small_terminal(self, G):
        deg = G.degree()
        terminal_points = [i for i, d in deg if d == 1]
        edges = list(G.edges())
        for s, e in edges:
            if s == e:
                sum_len = 0
                vals = self.__flatten([[v] for v in G[s][s].values()])
                for ix, val in enumerate(vals):
                    sum_len += len(val['pts'])
                if sum_len < 3:
                    G.remove_edge(s, e)
                    continue
            vals = self.__flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):
                if s in terminal_points and val.get('weight', 0) < 10:
                    G.remove_node(s)
                if e in terminal_points and val.get('weight', 0) < 10:
                    G.remove_node(e)
        return

    def __add_small_segments(self, G, terminal_points, terminal_lines):
        'Проверка близлежащих точек на возможное соединение линиями'
        node = G.nodes
        term = [node[t]['o'] for t in terminal_points]
        dists = squareform(pdist(term))  # получение квадратной матрицы расстояний от точек к точкам
        possible = np.argwhere((dists > 0) & (
                    dists < 20))  # выделить возможными для соединения где расстояние между точками в диапазоне (0,20)
        good_pairs = []
        for s, e in possible:
            if s > e:  # соблюдение очередности стартовая - конечная
                continue
            s, e = terminal_points[s], terminal_points[e]  # берем координаты

            if G.has_edge(s, e):  # если уже есть грань пропустить
                continue
            good_pairs.append((s, e))  # добавить в результат

        possible2 = np.argwhere((dists > 20) & (dists < 100))  # проверяем в диапазоне (20,100)
        for s, e in possible2:
            if s > e:
                continue
            s, e = terminal_points[s], terminal_points[e]
            if G.has_edge(s, e):
                continue
            l1 = terminal_lines[s]  # берем связную линию стартовой
            l2 = terminal_lines[e]  # берем связную линию конечной
            d = self.__line_points_dist(l1, l2[0])  # высчитываем расстояние между линиями

            if abs(d) > 20:  # если больше 20 пропускаем
                continue
            angle = self.__get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])  # высчитываем угол между линиями
            if -20 < angle < 20 or angle < -160 or angle > 160:  # если угол линиями меньше 20 градусов
                good_pairs.append((s, e))  # добавить в результат
        # высчитываем расстояния между возможных пар
        dists = {}
        for s, e in good_pairs:
            s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
            dists[(s, e)] = np.linalg.norm(s_d - e_d)
        # сортируем
        dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))
        # добавляем
        wkt = []
        added = set()
        for s, e in dists.keys():
            if s not in added and e not in added:
                added.add(s)
                added.add(e)
                s_d, e_d = G.nodes[s]['o'], G.nodes[e]['o']
                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
                line = '(' + ", ".join(line_strings) + ')'
                wkt.append(linestring.format(line))
        return wkt

    def __add_direction_change_nodes(self, pts, s, e, s_coord, e_coord):
        'Аппроксимация методом Дугласа-Пекера'
        if len(pts) > 3:
            ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)  # преобразование для метода аппроксимации
            approx = 2
            ps = cv2.approxPolyDP(ps, approx, False)
            ps = np.squeeze(ps, 1)
            # ps[] - s_coord - расстояние между точками в координатах
            st_dist = np.linalg.norm(ps[
                                         0] - s_coord)  # величина вектора между первой точкой результата апп. и исходной стартовой координатой
            en_dist = np.linalg.norm(ps[-1] - s_coord)  # между конечной апп и стартовой

            if st_dist > en_dist:  # если конечная точка апп ближе
                s, e = e, s  # меняем стартовую с конечной
                s_coord, e_coord = e_coord, s_coord  # включая координаты
            ps[0] = s_coord  # уточняем координаты крайних точек
            ps[-1] = e_coord
        else:
            ps = np.array([s_coord, e_coord], dtype=np.int32)
        return ps


def wkt_to_shp(wkt_list, shp_file, offset=None):
    '''Take output of build_graph_wkt() and render the list of linestrings
    into a shapefile
    # https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    '''

    # Define a linestring feature geometry with one attribute
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
    }

    # Write a new shapefile
    with fiona.open(shp_file, 'w', 'ESRI Shapefile', schema) as c:
        lines = [loads(l) for l in wkt_list]
        # draw edges by pts
        for ls in lines:
            x, y = ls.coords.xy
            line_strings = []
            for i, temp in enumerate(x):
                line_strings.append("{0} {1}".format(float(x[i] + offset[0]), float(-y[i] + offset[1])))
            line = '(' + ", ".join(line_strings) + ')'
            # print(ls)
            print()
            # for i, line in enumerate(wkt_list):
            shape = shapely.wkt.loads(linestring.format(line))
            # print(type(line))
            c.write({
                'geometry': mapping(shape),
                'properties': {'id': i},
            })
    return


if __name__ == "__main__":
    # rw = RoadsWorker()
    # rw.vectorize_roads_from_segmented_image('24478825_15-segmented.tiff','E:/Projects/VKR/temp/results/24478825_15-segmented.tiff', 'E:/Projects/VKR/temp/results/24478825_15-segmented')

    with open(os.path.join('E:/Projects/VKR/temp/results/23429080_15-segmented', 'wkt.dat'), 'rb') as fp:
        wkt = pickle.load(fp)
    wkt_to_shp(wkt, 'file.shp')

