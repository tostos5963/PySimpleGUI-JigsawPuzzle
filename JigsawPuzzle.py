#!/usr/bin/env python3
import os
import base64
import random
import numpy as np
import cv2
import PySimpleGUI as sg

def cvimg2pngimg(cv_img):
    _, encoded = cv2.imencode(".png", cv_img)
    return base64.b64encode(encoded).decode('ascii')

class MaskImageInfo():
    def __init__(self, img, contours, w, h) -> None:
        self.img = img
        self.contours = contours
        self.w = w
        self.h = h

class MaskImage():
    def __init__(self, resize_x, resize_y, scale_x, scale_y) -> None:
        # F: Flat,  C: conCave 凹  V: conVex  凸
        # Up-Right-Down-Left
        # 例) FVCF          F
        #            +--------------+
        #            |              |
        #            |              +----+
        #          F |                   | V
        #            |    +----+    +----+
        #            |    |    |    |
        #            +----+    +----+
        #                   C
        self.keylist = ['CCCC', 'CCCF', 'CCCV', 'CCFC', 'CCFF', 'CCFV', 'CCVC', 'CCVF',
                        'CCVV', 'CFCC', 'CFCV', 'CFFC', 'CFFV', 'CFVC', 'CFVV', 'CVCC',
                        'CVCF', 'CVCV', 'CVFC', 'CVFF', 'CVFV', 'CVVC', 'CVVF', 'CVVV',
                        'FCCC', 'FCCF', 'FCCV', 'FCVC', 'FCVF', 'FCVV', 'FFCC', 'FFCV',
                        'FFVC', 'FFVV', 'FVCC', 'FVCF', 'FVCV', 'FVVC', 'FVVF', 'FVVV',
                        'VCCC', 'VCCF', 'VCCV', 'VCFC', 'VCFF', 'VCFV', 'VCVC', 'VCVF',
                        'VCVV', 'VFCC', 'VFCV', 'VFFC', 'VFFV', 'VFVC', 'VFVV', 'VVCC',
                        'VVCF', 'VVCV', 'VVFC', 'VVFF', 'VVFV', 'VVVC', 'VVVF', 'VVVV']
        self.mask_info_tbl = {}

        for key in self.keylist:
            p_fname = f'images/jigsaw/{key}.png'
            img_mask = cv2.imread(p_fname, cv2.IMREAD_COLOR)
            sx = int(img_mask.shape[1] * resize_x * scale_x)
            sy = int(img_mask.shape[0] * resize_y * scale_y)
            img_mask_small = cv2.resize(img_mask, dsize=(sx, sy))
            mask_h, mask_w, ch = img_mask_small.shape

            mask_gray = cv2.cvtColor(img_mask_small, cv2.COLOR_BGR2GRAY)
            mask_mono = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
            mask_bin = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)[1]

            contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

            img_mask_mono = cv2.merge((mask_mono, mask_mono, mask_mono, mask_mono))
            img_mask_mono_not = cv2.bitwise_not(img_mask_mono)
            self.mask_info_tbl[key] = MaskImageInfo(img_mask_mono_not, contours, mask_w, mask_h)

class PieceObject():
    def is_hit(self, pos): pass
    def img_id(self): pass
    def has_imgid(self, imgid): pass
    def move(self, draw, dx, dy): pass
    def to_front(self, draw): pass
    def num_pieces(self) -> int: pass

WITHIN_TOLERANCE = 20

class PieceNode(PieceObject):
    def __init__(self, img_piece, img_id, x, y, row, col):
        self.img_piece = img_piece
        self._img_id = img_id
        self.last_x = x
        self.last_y = y
        self.x = x
        self.y = y
        self.row = row
        self.col = col

    def is_hit(self, pos):
        x, y = pos
        if self.x <= x < (self.x + self.img_piece.shape[1]):
            if self.y <= y < (self.y + self.img_piece.shape[0]):
                color = self.img_piece[(y - self.y, x - self.x)]
                if color[3] == 255:
                    return True
        return False

    @property
    def img_id(self):
        return self._img_id

    def has_imgid(self, imgid):
        if self.img_id == imgid:
            return True
        else:
            return False

    def move(self, draw, dx, dy):
        self.x += dx
        self.y += dy
        draw.MoveFigure(self.img_id, dx, dy)

    def to_front(self, draw):
        draw.BringFigureToFront(self.img_id)

    def adjacent_piece(self, piece):
        if self.row == piece.row:
            if  (piece.y < (self.y - WITHIN_TOLERANCE)) or ((self.y + WITHIN_TOLERANCE) < piece.y):
                return (False, 0, 0)

            adj_y = self.y - (piece.last_y - self.last_y)

            if (self.col - 1) == piece.col:
                # [piece][self]
                adj_x = self.x - (self.last_x - piece.last_x)

                if (piece.x < (adj_x - WITHIN_TOLERANCE)) or ((adj_x + WITHIN_TOLERANCE) < piece.x):
                    return (False, 0, 0)
                else:
                    return (True, adj_x - piece.x, adj_y - piece.y)

            if (self.col + 1) == piece.col:
                # [self][piece]
                adj_x = self.x + (piece.last_x - self.last_x)

                if (piece.x < (adj_x - WITHIN_TOLERANCE)) or ((adj_x + WITHIN_TOLERANCE) < piece.x):
                    return (False, 0, 0)
                else:
                    return (True, adj_x - piece.x, adj_y - piece.y)

        if self.col == piece.col:
            if  (piece.x < (self.x - WITHIN_TOLERANCE)) or ((self.x + WITHIN_TOLERANCE) < piece.x):
                return (False, 0, 0)

            adj_x = self.x - (piece.last_x - self.last_x)

            if (self.row - 1) == piece.row:
                # [piece]
                # [self]
                adj_y = self.y - (self.last_y - piece.last_y)

                if (piece.y < (adj_y - WITHIN_TOLERANCE)) or ((adj_y + WITHIN_TOLERANCE) < piece.y):
                    return (False, 0, 0)
                else:
                    return (True, adj_x - piece.x, adj_y - piece.y)

            if (self.row + 1) == piece.row:
                # [self]
                # [piece]
                adj_y = self.y + (piece.last_y - self.last_y)

                if (piece.y < (adj_y - WITHIN_TOLERANCE)) or ((adj_y + WITHIN_TOLERANCE) < piece.y):
                    return (False, 0, 0)
                else:
                    return (True, adj_x - piece.x, adj_y - piece.y)

        return (False, 0, 0)

    def num_pieces(self) -> int:
        return 1


class PieceComposite(PieceObject):
    def __init__(self, piece):
        self.pieces = []
        self.pieces.append(piece)

    @property
    def img_id(self):
        return self.pieces[0].img_id

    def is_hit(self, pos):
        hit = False
        for piece in self.pieces:
            if piece.is_hit(pos):
                hit = True
                break
        return hit

    def has_imgid(self, imgid):
        for piece in self.pieces:
            if piece.has_imgid(imgid):
                return True
        return False

    def add(self, piece):
        self.pieces.append(piece)

    def remove(self, piece):
        self.pieces.remove(piece)

    def move(self, draw, dx, dy):
        for piece in self.pieces:
            piece.move(draw, dx, dy)

    def to_front(self, draw):
        for piece in self.pieces:
            piece.to_front(draw)

    def connect(self, draw, pieces2) -> bool:
        can_connect, dx, dy = (False, 0, 0)
        for p1 in self.pieces:
            for p2 in pieces2.pieces:
                can_connect, dx, dy = p1.adjacent_piece(p2)
                if can_connect:
                    break
            if can_connect:
                break

        if can_connect:
            for p2 in pieces2.pieces:
                p2.move(draw, dx, dy)
                self.add(p2)
            return True

        return False

    def num_pieces(self) -> int:
        return len(self.pieces)

class Jigsaw():
    def __init__(self, draw) -> None:
        self.piece_tbl = {}
        self.first_click = True
        self.drag_img_id = 0
        self.drag_img_ids = []
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.draw = draw

    def add(self, img_id, piece):
        self.piece_tbl[img_id] = piece

    def click_down(self, pos):
        self.x1, self.y1 = pos
        if self.first_click:
            self.first_click = False
            self.x0, self.y0 = (self.x1, self.y1)
            ids = self.draw.GetFiguresAtLocation((self.x1, self.y1))
            is_hit = False
            if len(ids) >= 1:
                for imgid in reversed(ids):
                    is_hit = False
                    for id in self.piece_tbl:
                        if self.piece_tbl[id].has_imgid(imgid):
                            if self.piece_tbl[id].is_hit((self.x1, self.y1)):
                                self.drag_img_id = id
                                try:
                                    self.drag_img_ids.remove(id)
                                except Exception:
                                    pass
                                self.drag_img_ids.append(id)
                                self.piece_tbl[id].to_front(self.draw)
                                is_hit = True
                                break

                    if is_hit:
                        break
            if is_hit == False:
                self.drag_img_id = 0
        else:
            if self.drag_img_id in self.piece_tbl:
                dx, dy = self.x1 - self.x0, self.y1 - self.y0
                self.x0, self.y0 = (self.x1, self.y1)
                self.piece_tbl[self.drag_img_id].move(self.draw, dx, dy)

    def click_up(self):
        self.first_click = True
        if len(self.drag_img_ids) <= 0:
            return

        if len(self.drag_img_ids) >= 3:
            self.drag_img_ids.pop(0)

        if len(self.drag_img_ids) <= 1:
            img_id_prev = 0
            img_id_now = self.drag_img_ids[0]
        else:
            img_id_prev = self.drag_img_ids[0]
            img_id_now = self.drag_img_ids[1]

        if img_id_prev != 0 and img_id_prev != img_id_now:
            if self.piece_tbl[img_id_prev].connect(self.draw, self.piece_tbl[img_id_now]):
                del self.piece_tbl[img_id_now]
                self.drag_img_ids.remove(img_id_now)
                return

        max_pieces_img_id = 0
        num_max = 0
        for img_id in self.piece_tbl:
            if num_max < self.piece_tbl[img_id].num_pieces():
                max_pieces_img_id = img_id
                num_max = self.piece_tbl[img_id].num_pieces()

        if max_pieces_img_id != 0 and max_pieces_img_id != img_id_now:
            if self.piece_tbl[max_pieces_img_id].connect(self.draw, self.piece_tbl[img_id_now]):
                del self.piece_tbl[img_id_now]
                self.drag_img_ids.remove(img_id_now)

    def num_blocks(self) -> int:
        return len(self.piece_tbl)

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

class Board():
    def __init__(self, disp_image_width, disp_image_height) -> None:
        random.seed()

        self.is_complte = False
        self.event = None
        self.values = None

        self.disp_image_width = disp_image_width
        self.disp_image_height = disp_image_height

        self.graph_w = self.disp_image_width + 100
        self.graph_h = self.disp_image_height + 100

        img_dir = os.path.dirname(__file__) + "/images"
        self.layout = [ [sg.Text("Image file"), sg.InputText(size=(80,1), key='img_ffile_name'),
                         sg.FileBrowse(key="browse_img_file", initial_folder=img_dir, file_types=(("image", ".png"),("image", ".jpeg"),("image", ".jpg"),)) ],
                        [sg.Text("Number of pieces"), sg.Combo(values=[24, 60, 144, 240], default_value=24, key="num_pieces"), sg.Button('Start')],
                    [sg.Graph(canvas_size=(self.graph_w, self.graph_h),
                    graph_bottom_left=(0, self.graph_h), key='Graph', pad=(0,0), enable_events=True,
                    graph_top_right=(self.graph_w, 0), drag_submits=True)]]
        self.window = sg.Window('JigsawPuzzle', layout=self.layout, finalize=True)

        self.jigsaw = None

    def close(self):
        self.window.close()

    def click_down(self, pos):
        if self.jigsaw != None:
            self.jigsaw.click_down(pos)

    def click_up(self):
        if self.jigsaw != None:
            self.jigsaw.click_up()
            if self.is_complte != True:
                if self.jigsaw.num_blocks() <= 1:
                    self.is_complte = True
                    sg.popup('complete!', keep_on_top=True)

    def onclick_start(self):
        img_file = self.values['img_ffile_name']
        num_pieces = int(self.values['num_pieces'])

        try:
            org_img = cv2.imread(img_file)
            img_pic_h, img_pic_w, _  = org_img.shape

            sx = float(self.disp_image_width / img_pic_w)
            sy = float(self.disp_image_height / img_pic_h)
            if sx < sy:
                self.img_pic = cv2.resize(org_img, dsize=None, fx=sx, fy=sx)
            else:
                self.img_pic = cv2.resize(org_img, dsize=None, fx=sy, fy=sy)

            img_pic_h, img_pic_w, _  = self.img_pic.shape

            resize_x = img_pic_w / float(IMAGE_WIDTH)
            resize_y = img_pic_h / float(IMAGE_HEIGHT)
        except Exception as err:
            print(err)
            return

        if num_pieces == 24:
            num_cols = 6
            num_rows = 4

        elif num_pieces == 60:
            num_cols = 10
            num_rows = 6

        elif num_pieces == 144:
            num_cols = 16
            num_rows = 9

        elif num_pieces == 240:
            num_cols = 20
            num_rows = 12

        else:
            return

        x_step = int(IMAGE_WIDTH / num_cols)
        x_1st = int(x_step * 0.722)
        x_list = [0, x_1st]
        for x in range(2, num_cols):
            x_list.append(x_1st + x_step * (x - 1))

        y_step = int(IMAGE_HEIGHT / num_rows)
        y_1st = int(y_step * 0.722)
        y_list = [0, y_1st]
        for y in range(2, num_rows):
            y_list.append(y_1st + y_step * (y - 1))

        self.window['img_ffile_name'].update(disabled=True)
        self.window['browse_img_file'].update(disabled=True)
        self.window['Start'].update(disabled=True)
        self.window['num_pieces'].update(disabled=True)

        self.draw = self.window['Graph']
        self.jigsaw = Jigsaw(self.draw)
        if self.img_pic.ndim == 3:
            self.img_pic = cv2.cvtColor(self.img_pic, cv2.COLOR_RGB2RGBA)

        self.mask = MaskImage(resize_x, resize_y, 6.0 / num_cols, 4.0 / num_rows)
        self.piece_tbl = {}

        u_list = {}     # UP
        r_list = {}     # RIGHT
        d_list = {}     # DOWN
        l_list = {}     # LEFT
        for row in range(num_rows):
            for col in range(num_cols):
                # F: Flat,  C: conCave 凹  V: conVex  凸
                rval = random.randint(1, 100) % 2
                if rval == 0:
                    d_list[(row, col)] = 'C'
                else:
                    d_list[(row, col)] = 'V'

                if row == 0:
                    u_list[(row, col)] = 'F'
                else:
                    if d_list[(row - 1, col)] == 'C':
                        u_list[(row, col)] = 'V'
                    else:
                        u_list[(row, col)] = 'C'
                    if row == (num_rows - 1):
                        d_list[(row, col)] = 'F'

                rval = random.randint(1, 100) % 2
                if rval == 0:
                    r_list[(row, col)] = 'C'
                else:
                    r_list[(row, col)] = 'V'

                if col == 0:
                    l_list[(row, col)] = 'F'
                else:
                    if r_list[(row, col - 1)] == 'C':
                        l_list[(row, col)] = 'V'
                    else:
                        l_list[(row, col)] = 'C'
                    if col == (num_cols - 1):
                        r_list[(row, col)] = 'F'

                mask_key = u_list[(row, col)] + r_list[(row, col)] + d_list[(row, col)] + l_list[(row, col)]
                mask_info = self.mask.mask_info_tbl[mask_key]

                img_mask_mono_not = mask_info.img
                mask_w = mask_info.w
                mask_h = mask_info.h

                x = int(x_list[col] * resize_x)
                y = int(y_list[row] * resize_y)
                img_crop = self.img_pic[y:y+mask_h, x:x+mask_w]
                img_contours = img_crop.copy()
                cv2.drawContours(img_contours, mask_info.contours, -1, color=(255, 255, 255, 255), thickness=1)

                img_crop_mask = cv2.bitwise_and(img_contours, img_mask_mono_not)

                id_img = self.draw.DrawImage(data=cvimg2pngimg(img_crop_mask), location=(x, y))
                self.jigsaw.add(id_img, PieceComposite(PieceNode(img_crop_mask, id_img, x, y, row, col)))

                move_x = random.randint(10, self.graph_w - int(mask_w * 1.2))
                move_y = random.randint(10, self.graph_h - int(mask_h * 1.2))
                self.jigsaw.piece_tbl[id_img].move(self.draw, move_x - x, move_y - y)

    def main_loop(self):
        while True:
            self.event, self.values = self.window.read()
            if self.event == sg.WIN_CLOSED:
                self.close()
                break

            if self.event == 'Start':
                self.onclick_start()

            if self.event == 'Graph':
                board.click_down(self.values['Graph'])

            elif self.event == 'Graph+UP':
                board.click_up()

screen_width, screen_height = sg.Window.get_screen_size()
if screen_width >= 2060 and screen_height >= 1300:
    disp_image_width = 1920
    disp_image_height = 1080
elif screen_width >= 1820 and screen_height >= 1160:
    disp_image_width = 1680
    disp_image_height = 945
elif screen_width >= 1100 or screen_height >= 800:
    disp_image_width = 960
    disp_image_height = 540
elif screen_width >= 800 or screen_height >= 600:
    disp_image_width = 640
    disp_image_height = 360
else:
    sg.popup('ERROR: display size error')
    disp_image_width = 640
    disp_image_height = 360

board = Board(disp_image_width, disp_image_height)
board.main_loop()

# vim:set ts=4 sts=4 sw=4 tw=0 ft=python:
