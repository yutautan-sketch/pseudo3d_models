import os
import re
import cv2
import glob
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
import math
from tqdm import tqdm
import numpy as np


# XML パーサ
def parse_xml_for_start_end(xml_path, normalize=True):
    """
    <name>=start or <name>=end の xmin,ymin を返す
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = float(root.find("size").find("width").text)
    height = float(root.find("size").find("height").text)

    result = {}
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in ("start", "end"):
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            if normalize:
                result[name] = (xmin / width, ymin / height)
            else:
                result[name] = (xmin, ymin)

    return result


def parse_xml_for_leg_cxcy(xml_path, normalize=True):
    """
    <name>=leg の中心座標 (cx,cy) を返す
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = float(root.find("size").find("width").text)
    height = float(root.find("size").find("height").text)

    for obj in root.findall("object"):
        if obj.find("name").text == "leg":
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            return (cx / width, cy / height)

    return None


def parse_xml_for_leg_xyxy(xml_path, normalize=True):
    """
    <name>=leg の座標 (xmin, ymin, xmax, ymax) をすべて返す
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = float(root.find("size").find("width").text)
    height = float(root.find("size").find("height").text)

    results = []
    for obj in root.findall("object"):
        if obj.find("name").text == "leg":
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)
            if normalize:
                results.append((xmin/width, ymin/height, xmax/width, ymax/height))
            else:
                results.append((xmin, ymin, xmax, ymax))

    return results if results else None


def get_image_size(xml_path):
    """
    画像サイズ (width, height) を返す
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    return width, height


def get_frame_idx(fname):
    """
    filename 末尾の 00019 を取得する
    """
    return int(os.path.splitext(fname)[0].split("_")[-1])


def nearest_track(points, xy, frame_idx):
    best_key = None
    best_score = 1e9
    print(f"frame_idx: {frame_idx}")
    print(f"points: {points}")
    for key, track in points.items():
        
        # track の最後のフレーム番号
        last_frame = max(k for k, v in track.items())
        last_val   = track[last_frame]   # XY または None（＝終了）
        
        # ---- 終端マーカーが付いている track は除外 ----
        print(f"last_frame: {last_frame}, last_val: {last_val}")
        if last_val is None:
            continue

        # ---- 時系列制約：過去に戻る場合は除外 ----
        if frame_idx <= last_frame:
            continue

        last_xy = track[last_frame]
        if last_xy is None:
            continue

        d = (last_xy[0] - xy[0])**2 + (last_xy[1] - xy[1])**2  # 近さ評価

        if d < best_score:
            best_score = d
            best_key = key
    
    print(key)

    return best_key


# 軌跡距離を計算する関数
def compute_trajectory_distance(track_dict):
    """
    track_dict: OrderedDict({frame: (x,y) or None})
    → Noneを除外し、フレーム順に距離を積算
    """
    pts = [(frame, xy) for frame, xy in sorted(track_dict.items()) if xy is not None]

    if len(pts) < 2:
        return 0.0
    
    pts = sorted(pts, key=lambda x: x[0])

    dist_sum = 0.0
    for i in range(len(pts) - 1):
        xy1 = pts[i][1]
        xy2 = pts[i+1][1]
        dist_sum += math.dist(xy1, xy2)

    return dist_sum


# 画像を切り抜く関数
def crop_img(img, xyxy):
    """
    画像を xyxy の範囲で切り抜く

    Args:
        img ([h, w, c]): 画像
        xyxy ((xmin, ymin, xmax, ymax)): 範囲

    Returns:
        crop ([ymax-ymin, xmax-xmin, c]) : 切り抜かれた画像
    """
    xmin, ymin, xmax, ymax = map(int, xyxy)
    crop = img[ymin:ymax, xmin:xmax]
    return crop


# 画像を二値化する関数
def binary_img(gray_img, thresh=128):
    """
    グレースケール画像を二値化する

    Args:
        gray_img ([h, w]): グレースケール画像
        thresh (int): 閾値

    Returns:
        binary_img ([h, w]): 二値化画像
    """
    # 1. メディアンフィルタ
    background = cv2.medianBlur(gray_img, 35)
    subtracted = cv2.subtract(gray_img, background)
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. CLAHE による局所コントラスト補正
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 3. 輝度補正 + CLAHE
    clahe_subtracted = clahe.apply(subtracted)
    
    # 4. Otsuの二値化
    thresh, binary = cv2.threshold(clahe_subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh, binary


# 輪郭抽出 + 重心を計算する関数
def contour_centroid(binary, combine_num=1):
    """
    輪郭の重心を計算する

    Args:
        binary ([h, w, c]): 二値化画像

    Returns:
        center (x, y): 最大輪郭の重心座標
    """
    # 1. 輪郭抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_num = len(contours)
    if contours_num == 0:
        return None
    
    # 2. 面積でソート
    top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:min(combine_num, contours_num)]
    combined_contour = np.vstack(top_contours)
    
    # 3. 重心計算
    M = cv2.moments(combined_contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)


# 推論データセットにおける video フレーム群の最初のフレーム番号を取得する関数
def get_first_frame_num(video_path):
    first_vid = sorted(os.listdir(video_path))[0]
    match = re.search(r'(\d+)(?=\.jpg$)', first_vid)
    if match:
        return int(match.group(1))


# メイン処理
def process(root_1, root_2, root_3, root_4):

    result = []

    for video in tqdm(sorted(os.listdir(root_2))):
        print(f"\n================= VIDEO: {video} =================")

        # root_2: start / end を収集
        xmls_2 = sorted(glob.glob(f"{root_2}/{video}/*.xml"))

        points = OrderedDict()     # { frame_start : OrderedDict({frame_start:(x,y), ...}) }
        end_points = []            # [(frame_end, (x,y))]

        for fp in xmls_2:
            frame_idx = get_frame_idx(os.path.basename(fp))
            info = parse_xml_for_start_end(fp)

            if "start" in info:
                points[frame_idx] = OrderedDict({frame_idx-0.4: info["start"]})

            if "end" in info:
                end_points.append((frame_idx, info["end"]))

        if not points:
            print(" → start が無いためスキップ")
            continue

        start_min = int(min(points.keys()) + 0.5)
        print(f" 🔹 start frames = {list(points.keys())}")
        print(f" 🔹 end frames   = {end_points}")
        
        # root_4: 修正する frame_idx の値を取得
        adj_fidx_path = os.path.join(root_4, video, "object")
        adj_fidx_num = get_first_frame_num(adj_fidx_path) - 1

        # root_1: leg 追跡
        xmls_1 = sorted(glob.glob(f"{root_1}/{video}/*.xml"))
        xmls_1 = [f for f in xmls_1 if (get_frame_idx(os.path.basename(f))-adj_fidx_num) >= start_min]

        for fp in xmls_1:
            frame_idx = get_frame_idx(os.path.basename(fp))
            frame_idx -= adj_fidx_num
            leg_xyxy_list = parse_xml_for_leg_xyxy(fp, normalize=False)
            if leg_xyxy_list is None:
                continue
            
            # 対応する動画の取得
            frame_path = os.path.join(root_3, video, f"{os.path.basename(fp)[:-4]}.jpg")
            frame = cv2.imread(frame_path)
            
            width, height = get_image_size(fp)
            
            # 複数のlegに対応
            for leg_xyxy in leg_xyxy_list:
                # leg_xyxy の範囲で切り抜き -> 二値化
                crop = crop_img(frame, leg_xyxy)
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, crop_bin = binary_img(crop_gray)
                
                # 輪郭抽出 -> 面積最大の重心を leg_xy とする
                leg_xy = contour_centroid(crop_bin)
                if leg_xy is None:
                    leg_xy = parse_xml_for_leg_cxcy(fp)
                    if leg_xy is None:
                        continue
                
                # 座標修正 + 正規化
                leg_xy = ((leg_xy[0] + leg_xyxy[0]) / width, (leg_xy[1] + leg_xyxy[1]) / height)

                key = nearest_track(points, leg_xy, frame_idx)
                if key is None:
                    continue

                points[key][frame_idx] = leg_xy

            # ===== END FRAME に到達したかチェック =====
            matched_ends = [e for e in end_points if e[0] == frame_idx]
            if matched_ends:
                for e_idx, e_xy in matched_ends:
                    key = nearest_track(points, e_xy, (e_idx+0.4))
                    if key is None:
                        print("⚠️ 終端マーカーの nearest track が見つからない")
                        continue
                    points[key][e_idx+0.4] = e_xy
                    points[key][e_idx+0.41] = None   # ★終端マーカー

        # JSON の形式に変換
        item = {
            "id": len(result) + 1,
            "femur_points": [dict(sorted(v.items())) for v in points.values()],
            "femur_traj_len": [
                compute_trajectory_distance(v) for v in points.values()
            ],
            "videos": [video]
        }
        result.append(item)
        # if len(result)+1 == 84:
        #     exit()
    return result


# 実行
root_1 = "/Users/yutakodaira/Desktop/cocoアノテーション作成用_頭+お腹+足/251117_検証用/anno"
root_2 = "/Users/yutakodaira/Desktop/大腿骨最初-最後アノテーション/femur_end_anno"
root_3 = "/Users/yutakodaira/Desktop/cocoアノテーション作成用_頭+お腹+足/251117_検証用/frames"
root_4 = "/Users/yutakodaira/Desktop/cocoアノテーション作成用_頭+お腹+足/251117_検証用/251117_val"

merged = process(root_1, root_2, root_3, root_4)

with open("/Users/yutakodaira/Desktop/大腿骨最初-最後アノテーション/anno_hbl_251220_legtraj.json", "w") as f:
    json.dump(merged, f, indent=2)

print("\n✅ merged_tracks.json を出力しました\n")
