# -*- coding: utf-8 -*-
# 获取屏幕信息，并通过视觉方法标定手牌与按钮位置，仿真鼠标点击操作输出
import os
import random
import time
from functools import wraps
from typing import List, Tuple

import cv2
import numpy as np
import pyautogui
from loguru import logger as logging

from .classifier import Classify
from ..sdk import Operation

pyautogui.PAUSE = 0  # 函数执行后暂停时间
pyautogui.FAILSAFE = False  # 开启鼠标移动到左上角自动退出


class ActionFailed(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def auto_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        waitPos = getattr(args[0], "waitPos", np.int32([0, 0]))

        err = None

        for i in range(10):
            try:
                return func(*args, **kwargs)
            except ActionFailed as e:
                err = e
                logging.warning(f"{func.__name__} failed ({e}), "
                                f"retrying... {i+1}/10")

                pyautogui.moveTo(waitPos[0], waitPos[1])
                time.sleep(0.3)

        raise err
    return wrapper


DEBUG = False  # 是否显示检测中间结果


def PosTransfer(pos, M: np.ndarray) -> np.ndarray:
    assert (len(pos) == 2)
    return cv2.perspectiveTransform(np.float32([[pos]]), M)[0][0]


def Similarity(img1: np.ndarray, img2: np.ndarray):
    assert (len(img1.shape) == len(img2.shape) == 3)
    if img1.shape[0] < img2.shape[0]:
        img1, img2 = img2, img1
    n, m, c = img2.shape
    img1 = cv2.resize(img1, (m, n))
    if DEBUG:
        cv2.imshow('diff', np.uint8(
            np.abs(np.float32(img1) - np.float32(img2))))
        cv2.waitKey(1)
    ksize = max(1, min(n, m) // 50)
    img1 = cv2.blur(img1, ksize=(ksize, ksize))
    img2 = cv2.blur(img2, ksize=(ksize, ksize))
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    if DEBUG:
        cv2.imshow('bit', np.uint8(
            (np.abs(img1 - img2) < 30).sum(2) == 3) * 255)
        cv2.waitKey(1)
    return ((np.abs(img1 - img2) < 30).sum(2) == 3).sum() / (n * m)


def ObjectLocalization(objImg: np.ndarray, targetImg: np.ndarray) -> np.ndarray:
    """
    https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    Feature based object detection
    return: Homography matrix M (objImg->targetImg), if not found return None
    """
    img1 = objImg
    img2 = targetImg
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=5000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # store all the good matches as per Lowe's ratio test.
    good = []
    for i, j in enumerate(matches):
        if len(j) == 2:
            m, n = j
            if m.distance < 0.7 * n.distance:
                good.append(m)
                matchesMask[i] = [1, 0]
    logging.debug(f'  Number of good matches: {len(good)}')
    if DEBUG:
        # draw
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(
            img1, kp1, img2, kp2, matches, None, **draw_params)
        img3 = cv2.pyrDown(img3)
        cv2.imshow('ORB match', img3)
        cv2.waitKey(1)
    # Homography
    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if DEBUG:
            # draw
            matchesMask = mask.ravel().tolist()
            h, w, d = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, (0, 0, 255), 10, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                                   good, None, **draw_params)
            img3 = cv2.pyrDown(img3)
            cv2.imshow('Homography match', img3)
            cv2.waitKey(1)
    else:
        logging.debug(
            "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        M = None
    assert (type(M) == type(None) or (
            type(M) == np.ndarray and M.shape == (3, 3)))
    return M


def getHomographyMatrix(img1, img2, threshold=0.0):
    # if similarity>threshold return M
    # else return None
    M = ObjectLocalization(img1, img2)
    if type(M) != type(None):
        logging.debug(f'  Homography Matrix: {M}', )
        n, m, c = img1.shape
        x0, y0 = np.int32(PosTransfer([0, 0], M))
        x1, y1 = np.int32(PosTransfer([m, n], M))
        sub_img = img2[y0:y1, x0:x1, :]
        S = Similarity(img1, sub_img)
        logging.debug(f'Similarity: {S}', )
        if S > threshold:
            return M
    return None


def screenShot():
    img = np.asarray(pyautogui.screenshot())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if DEBUG:
        from PIL import Image
        Image.fromarray(img).save("screen.jpeg")

    return img


class Layout:
    size = (1920, 1080)  # 界面长宽
    duanWeiChang = (1348, 321)  # 段位场按钮
    menuButtons = [(1382, 406), (1382, 573), (1382, 740),
                   (1383, 885), (1393, 813)]  # 铜/银/金之间按钮
    tileSize = (95, 152)  # 自己牌的大小


class GUIInterface:

    def __init__(self):
        self.M = None  # Homography matrix from (1920,1080) to real window
        # load template imgs
        join = os.path.join
        root = os.path.dirname(__file__)

        def load(name):
            return cv2.imread(join(root, 'template', name))

        self.menuImg = load('menu.png')  # 初始菜单界面
        if (type(self.menuImg) == type(None)):
            raise FileNotFoundError(
                "menu.png not found, please check the Chinese path")
        assert (self.menuImg.shape == (1080, 1920, 3))
        self.chiImg = load('chi.png')
        self.pengImg = load('peng.png')
        self.gangImg = load('gang.png')
        self.huImg = load('hu.png')
        self.zimoImg = load('zimo.png')
        self.liujuImg = load('liuju.png')
        self.tiaoguoImg = load('tiaoguo.png')
        self.liqiImg = load('liqi.png')
        # load classify model
        self.classify = Classify()

    def _click_area(self, x, y, m, n, offset_range=0.75):
        x1 = x + (m-x)*(0.5 + (random.random()-0.5)*offset_range)
        y1 = y + (n-y)*(0.5 + (random.random()-0.5)*offset_range)

        logging.debug(f"click ({x1}, {y1})  (area: ({x}, {y}) ({m}, {n}))")

        pyautogui.moveTo(x1, y1)
        time.sleep(0.2 + 0.1 * random.random())
        pyautogui.click(x=x1, y=y1, button='left')
        time.sleep(0.2 + 0.1 * random.random())
        pyautogui.moveTo(self.waitPos[0], self.waitPos[1])

    def forceTiaoGuo(self):
        # 如果跳过按钮在屏幕上则强制点跳过，否则NoEffect
        try:
            self.clickButton(self.tiaoguoImg, similarityThreshold=0.7)
        except:
            pass

    @auto_retry
    def actionDiscardTile(self, tile: str):
        hand = self._getHandTiles()
        for t, (x, y, m, n) in reversed(hand):  # tsumogiri if possible
            if t == tile:
                self._click_area(x, y, m, n)

                new_hand = self._getHandTiles()
                if len(hand) - len(new_hand) == 1:
                    return
                else:
                    raise ActionFailed(
                        'tile found but failed to discard.  hand:', hand, 'tile:', tile)
        raise ActionFailed('tile not found. hand:', hand, 'tile:', tile)

    def actionChiPengGang(self, type_: Operation):
        if type_ == Operation.NoEffect:
            self.clickButton(self.tiaoguoImg)
        elif type_ == Operation.Chi:
            self.clickButton(self.chiImg)
        elif type_ == Operation.Peng:
            self.clickButton(self.pengImg)
        elif type_ in (Operation.MingGang, Operation.JiaGang):
            self.clickButton(self.gangImg)

    def actionLiqi(self, tile: str):
        self.clickButton(self.liqiImg)
        time.sleep(0.5)
        self.actionDiscardTile(tile)

    def actionHu(self):
        self.clickButton(self.huImg)

    def actionZimo(self):
        self.clickButton(self.zimoImg)

    def actionLiuju(self):
        self.clickButton(self.liujuImg)

    def calibrateMenu(self):
        # if the browser is on the initial menu, set self.M and return to True
        # if not return False
        self.M = getHomographyMatrix(self.menuImg, screenShot(), threshold=0.7)
        result = type(self.M) != type(None)
        if result:
            self.waitPos = np.int32(PosTransfer([100, 100], self.M))
        return result

    def _getHandTiles(self) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        # return a list of my tiles' position
        result = []
        assert (type(self.M) != type(None))
        # screen_img1 = screenShot()
        # time.sleep(0.5)
        # screen_img2 = screenShot()
        # screen_img = np.minimum(screen_img1, screen_img2)  # 消除高光动画
        screen_img = screenShot()  # 失败直接重试
        img = screen_img.copy()  # for calculation
        start = np.int32(PosTransfer([235, 1002], self.M))
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 110
        tileThreshold = np.int32(
            0.7 * (PosTransfer(Layout.tileSize, self.M) - O))
        fail = 0
        maxFail = np.int32(PosTransfer([100, 0], self.M) - O)[0]
        i = 0
        while fail < maxFail:
            x, y = start[0]+i, start[1]
            if all(img[y, x, :] > colorThreshold):
                fail = 0
                img[y, x, :] = colorThreshold
                retval, image, mask, rect = cv2.floodFill(
                    image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                    loDiff=(0, 0, 0), upDiff=tuple([255-colorThreshold]*3), flags=cv2.FLOODFILL_FIXED_RANGE)
                x, y, dx, dy = rect
                if dx > tileThreshold[0] and dy > tileThreshold[1]:
                    tile_img = screen_img[y:y+dy, x:x+dx, :]
                    tileStr = self.classify(tile_img)
                    result.append((tileStr, (x, y, x+dx, y+dy)))
                    i = x+dx-start[0]
            else:
                fail += 1
            i += 1
        return result

    @auto_retry
    def clickButton(self, buttonImg, similarityThreshold=0.6):
        # 点击吃碰杠胡立直自摸
        x0, y0 = np.int32(PosTransfer([0, 0], self.M))
        x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
        zoom = (x1 - x0) / Layout.size[0]
        n, m, _ = buttonImg.shape
        n = int(n * zoom)
        m = int(m * zoom)
        templ = cv2.resize(buttonImg, (m, n))
        x0, y0 = np.int32(PosTransfer([595, 557], self.M))
        x1, y1 = np.int32(PosTransfer([1508, 912], self.M))

        screen = screenShot()
        img = screen[y0:y1, x0:x1, :]
        T = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF, mask=templ.copy())
        _, _, (x, y), _ = cv2.minMaxLoc(T)
        if DEBUG:
            T = np.exp((1 - T / T.max()) * 10)
            T = T / T.max()
            cv2.imshow('T', T)
            cv2.waitKey(0)
        dst = img[y:y + n, x:x + m].copy()
        dst[templ == 0] = 0

        sim = Similarity(templ, dst)
        if sim >= similarityThreshold:
            self._click_area(x+x0, y+y0, x+x0+m, y+y0+n, 0.5)

            # 等待按钮消失
            for _ in range(5):
                screen = screenShot()
                dst = screen[y0:y1, x0:x1, :][y:y + n, x:x + m].copy()
                dst[templ == 0] = 0

                now_sim = Similarity(templ, dst)
                if now_sim < similarityThreshold:
                    return

                time.sleep(0.2)
            else:
                raise ActionFailed('button found but failed to click')
        else:
            raise ActionFailed('button not found')

    @auto_retry
    def clickCandidateMeld(self, tiles: List[str]):
        # 有多种不同的吃碰方法，二次点击选择
        assert (len(tiles) == 2)
        # find all combination tiles
        result = []
        assert (type(self.M) != type(None))
        screen_img = screenShot()
        img = screen_img.copy()  # for calculation
        start = np.int32(PosTransfer([960, 753], self.M))
        leftBound = rightBound = start[0]
        O = PosTransfer([0, 0], self.M)
        colorThreshold = 200
        tileThreshold = np.int32(0.7 * (PosTransfer((78, 106), self.M) - O))
        maxFail = np.int32(PosTransfer([60, 0], self.M) - O)[0]
        for offset in [-1, 1]:
            # 从中间向左右两个方向扫描
            i = 0
            while True:
                x, y = start[0] + i * offset, start[1]
                if offset == -1 and x < leftBound - maxFail:
                    break
                if offset == 1 and x > rightBound + maxFail:
                    break
                if all(img[y, x, :] > colorThreshold):
                    img[y, x, :] = colorThreshold
                    retval, image, mask, rect = cv2.floodFill(
                        image=img, mask=None, seedPoint=(x, y), newVal=(0, 0, 0),
                        loDiff=(0, 0, 0), upDiff=tuple([255 - colorThreshold] * 3), flags=cv2.FLOODFILL_FIXED_RANGE)
                    x, y, dx, dy = rect
                    if dx > tileThreshold[0] and dy > tileThreshold[1]:
                        tile_img = screen_img[y:y + dy, x:x + dx, :]
                        tileStr = self.classify(tile_img)
                        result.append((tileStr, (x, y, x+dx, y+dy)))
                        leftBound = min(leftBound, x)
                        rightBound = max(rightBound, x + dx)
                i += 1
        result = sorted(result, key=lambda x: x[1][0])
        if len(result) == 0:
            return True  # 其他人先抢先Meld了！
        assert (len(result) % 2 == 0)
        for i in range(0, len(result), 2):
            if result[i][0] == tiles[0] and result[i + 1][0] == tiles[1]:
                x, y, m, n = result[i][1]
                self._click_area(x, y, m, n)
                return True
        raise ActionFailed('combination not found, tiles:',
                           tiles, ' combination:', result)

    def actionReturnToMenu(self):
        # 在终局以后点击确定跳转回菜单主界面
        x, y = np.int32(PosTransfer((1785, 1003), self.M))  # 终局确认按钮
        while True:
            time.sleep(5)
            x0, y0 = np.int32(PosTransfer([0, 0], self.M))
            x1, y1 = np.int32(PosTransfer(Layout.size, self.M))
            img = screenShot()
            S = Similarity(self.menuImg, img[y0:y1, x0:x1, :])
            if S > 0.5:
                return True
            else:
                pyautogui.click(x=x, y=y, duration=0.5)

    def actionBeginGame(self, level: int, wind: int):
        """
        从开始界面点击匹配对局

        :param level: 0~4对应铜/银/金/玉/王座
        :param wind: 0对应四人东，1对应四人南
        """
        x, y = np.int32(PosTransfer(Layout.duanWeiChang, self.M))
        pyautogui.click(x, y)
        time.sleep(2)
        if level == 4:
            # 王座之间在屏幕外面需要先拖一下
            x, y = np.int32(PosTransfer(Layout.menuButtons[2], self.M))
            pyautogui.moveTo(x, y)
            time.sleep(1.5)
            x, y = np.int32(PosTransfer(Layout.menuButtons[0], self.M))
            pyautogui.dragTo(x, y)
            time.sleep(1.5)
        x, y = np.int32(PosTransfer(Layout.menuButtons[level], self.M))
        pyautogui.click(x, y)
        time.sleep(2)
        x, y = np.int32(PosTransfer(Layout.menuButtons[wind], self.M))
        pyautogui.click(x, y)
