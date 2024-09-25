#  Copyright (c) 2024 by Tomoya Konishi
#  All rights reserved.
#
#  License:
#  This program is permitted under the principle of "NO WARRANTY" and
#  "NO RESPONSIBILITY". The author shall not be liable for any event
#  arising in any way out of the use of these resources.
#
#  Redistribution in source and binary forms, with or without
#  modification, is also permitted provided that the above copyright
#  notice, disclaimer and this condition are retained.
#
#
#

import streamlit as st
import numpy as np
import pandas as pd
import fitz
import cv2
import imutils
from imutils.perspective import four_point_transform
import re
import base64

# 参考
# https://qiita.com/not13/items/dcd8c12d64982dc0e819

def main():
    # UIの構築
    st.title('OMR2 - マークシートリーダー')

    st.sidebar.write("""
    ## ● マークシート
    """)
    file_path = st.sidebar.file_uploader("PDFファイル", type="pdf")
    if not file_path:
        st.sidebar.write("マークシートを読み込んでください")
        return
    else:
        pdf_document = fitz.open(stream=file_path.read(), filetype="pdf")
        if pdf_document.page_count > 1:
            page = st.sidebar.slider('ページ選択 [←] [→]', 1, pdf_document.page_count, 1)
        else:
            page = 1

        img_width = st.sidebar.slider('表示サイズ', 0, 1000, 500, step=10)

    st.sidebar.write("""
    ## ● マーク検出設定
    """)
    if st.sidebar.checkbox('画像2値化閾値の自動設定', value=True):
        threshold = 0
    else:
        threshold = st.sidebar.slider('', 0, 255, 170)

    is_double_mark = st.sidebar.checkbox('ダブルマークを許可', value=True)
    str_dimensions = st.sidebar.text_input('各フレームのマーク数(行x列)', value='(4x10), (30x10), (30x10)')
    # タプルのリストに変換
    matches = re.findall(r'\((\d+)[x,](\d+)\)', str_dimensions)
    dim_list = [(int(x), int(y)) for x, y in matches]

    # 処理開始
    download_button = st.sidebar.button("全てのシートを処理")

    if download_button:
        data_list = do_all_omr(pdf_document, threshold, is_double_mark, dim_list)
        csv = pd.DataFrame(data_list).to_csv(index=False)
        csv_b64 = base64.b64encode(csv.encode()).decode()

        st.success("全てのシートの読み取りが完了しました")
        st.balloons()
        st.markdown(f'<a href="data:file/csv;base64,{csv_b64}" download="result.csv">Download csv file</a>',
                    unsafe_allow_html=True)
        st.button('戻る')
    else:
        # プレビュー
        do_omr(pdf_document, page, img_width, threshold, is_double_mark, dim_list)


def do_all_omr(pdf_document, threshold, is_double_mark, dim_list):
    data_list = []
    for page in range(1, pdf_document.page_count+1):
        data = do_omr(pdf_document, page, 0, threshold, is_double_mark, dim_list, is_show_result=False)
        data_list.append(data)

    return data_list


def do_omr(pdf_document, page, img_width, threshold, is_double_mark, dim_list, is_show_result=True):
    # 処理開始
    data_list = []
    img = get_image_from_pdf(pdf_document, page)
    img = correct_tilt(img)
    frame_list = find_frames(img)

    if len(frame_list) == 0:
        st.warning("エラー！ 解答欄のフレームが検出できませんでした。")

    for i, frame in enumerate(frame_list):
        mark_list, frame_img = find_marks(img, frame, threshold)
        frame_width = np.max(frame[:,0]) - np.min(frame[:,0])
        frame_height = np.max(frame[:,1]) - np.min(frame[:,1])
        try:
            _ = dim_list[i]
        except IndexError:
            st.warning("フレームの数とマーク数の設定が合っていません（フレーム数: {}, マーク数の設定: {}）"
                       .format(len(frame_list), len(dim_list)))
            st.stop()

        data = decode_marks((frame_height, frame_width), mark_list, dim_list[i], is_double_mark)
        data_list.append(data)

        if is_show_result:
            # 検出したマークの描画
            frame_origin = (np.min(frame[:,0]), np.min(frame[:,1]))
            for mark in mark_list:
                for pt in mark:
                    pt[0] += frame_origin

                cv2.drawContours(img, [mark], -1, (255, 0, 0), 2)

    if is_show_result:
        # フレームの描画
        for i,c in enumerate(frame_list):
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            cv2.putText(img, str(i), (np.min(c[:,0]), np.min(c[:,1])-20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            # cv2.circle(img, (np.min(c[0,0]), np.min(c[0,1])), 5, (255, 0, 0), -1)
            # cv2.circle(img, (np.min(c[1,0]), np.min(c[1,1])), 5, (0, 255, 0), -1)
            # cv2.circle(img, (np.min(c[2,0]), np.min(c[2,1])), 5, (0, 0, 255), -1)

    if is_show_result:
        # 加工済み画像の表示
        st.write("検出結果プレビュー")
        st.image(img, width=img_width)

        # 結果の表示
        st.write("マーク読み取り結果")
        df = pd.DataFrame(data_list)
        st.dataframe(df)

    # 1次元リストに変換
    data_list = [item for sublist in data_list for item in sublist]
    return data_list


def is_binary_image(image):
    # Get the unique pixel values in the image
    unique_values = np.unique(image)

    # Check if there are exactly two unique values: 0 and 255
    return len(unique_values) == 2 and set(unique_values).issubset({0, 255})

def decode_marks(frame_dim, mark_list, mark_array_dim, is_double_mark):
    MARK = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    mark_index = {char: index for index, char in enumerate(MARK)}

    frame_height, frame_width = frame_dim
    mark_rows, mark_cols = mark_array_dim

    data_list = ["" for i in range(mark_rows)]

    mark_centroid_list = [np.mean(mark, axis=0)[0] for mark in mark_list]
    mark_centroid_list = sorted(mark_centroid_list, key=lambda x: x[1])
    for mark in mark_centroid_list:
        x, y = mark
        # マークがどのセルに属するか
        row = int((y/frame_height)*mark_rows)
        col = int((x/frame_width)*mark_cols)
        if is_double_mark:
            # ダブルマークを許可
            data_list[row] += MARK[col]
        else:
            # ダブルマークを許可しない
            if data_list[row] == "":
                data_list[row] = MARK[col]
            else:
                data_list[row] = "X"

    # マークの並び替え
    data_list = [''.join(sorted(data, key=lambda x: mark_index[x]))
                 for data in data_list
                     if "X" not in data]
    return data_list

@st.cache_data
def find_marks(img, frame, threshold):
    frame_img = imutils.perspective.four_point_transform(img, frame)
    warped = threshold_image(frame_img, threshold)
    warped = remove_lines(warped)

    # 収縮・膨張によりノイズを除去
    kernel = np.ones((5, 5), np.uint8)
    warped = cv2.erode(warped, kernel, iterations=2)
    warped = cv2.dilate(warped, kernel, iterations=2)

    # マークの検出
    cnts = cv2.findContours(warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return [], frame_img

    # マークの選別
    mark_cnts = []
    max_area = np.max([cv2.contourArea(c) for c in cnts])
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # 形の悪いもの、小さいものは無視
        if ar < 0.6 or ar > 1.4 or cv2.contourArea(c) < 0.4*max_area:
            continue
        # マークが半分以上埋まっていないものは無視
        elif count_pixels_in_rect(warped,(x, y, w, h)) < 0.5*w*h:
            continue
        else:
            mark_cnts.append(c)

    mark_cnts = sorted(mark_cnts, key=lambda x: (np.mean(x[:]), np.mean(x[:0])))
    return mark_cnts, frame_img


def count_pixels_in_rect(img, rect):
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, -1)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return cv2.countNonZero(mask)


def correct_tilt(img):
    ### 画像の傾き補正
    # https://note.com/bibinbeef/n/ne399c766d9d5
    frame_list = find_frames(img)
    if len(frame_list) > 0:
        max_frame = sorted(frame_list, key=cv2.contourArea, reverse=True)[0]
        vec = max_frame[1] - max_frame[0]
        rot_theta = np.arctan2(vec[0], vec[1]) *180/3.14
        img = imutils.rotate(img, -rot_theta)
    return img


def get_image_from_pdf(pdf_document, page):
    ### 画像の読み込み
    # https://note.com/jolly_ixia1223/n/n499b3480fedc
    # https://qiita.com/inoshun/items/ded26487bf0065794d2c

    pdf_page = pdf_document[page-1]
    image_list = pdf_page.get_images(full=True)
    image_index = image_list[0][0]
    base_image = pdf_document.extract_image(image_index)
    image_bytes = base_image["image"]
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    rotation = pdf_page.rotation
    if rotation == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    margin_w, margin_h = img.shape[1]*0.05, img.shape[0]*0.03
    img = img[int(margin_h):int(-margin_h), int(margin_w):int(-margin_w)]

    if is_binary_image(img):
        st.warning("注意！ 2値化された画像です。グレースケールで読み込んでください。")

    width = 2000
    img = cv2.resize(img, (int(img.shape[1]*width/img.shape[0]), width))

    return img


def find_frames(img):
    ### フレーム(マーク領域)の検出

    # img = cv2.blur(img, (5, 5))

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)

    ksize = 1
    img = cv2.medianBlur(img, ksize)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 5)
    edged = imutils.auto_canny(img)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt_list = []

    if len(cnts) > 0:
        for c in cnts:
            # 小さい領域は無視
            if cv2.contourArea(c) < img.shape[0]*img.shape[1]*0.02:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                # sort the approx points starting from the top-left
                # x座標とy座標の和が最小の点を左上にする
                w = [i[0][0]+i[0][1] for i in approx]
                idx = w.index(min(w))
                idx_list = list(range(len(w)))
                idx_list = idx_list[idx:] + idx_list[:idx]
                approx = approx[idx_list]
                cnt_list.append(approx.reshape(4, 2))

    cnt_list = sorted(cnt_list, key=lambda x: (np.mean(x[:]), np.mean(x[:0])))
    return cnt_list


def split_image_vertical_and_choose_greatest(img):
    vp = np.sum((img != 0).astype(np.uint8), axis=0)
    loc_x_spike = np.where(vp == np.max(vp[:]))[0]
    diff_loc_x_spike = np.diff(loc_x_spike)
    idx = np.where(diff_loc_x_spike == max(diff_loc_x_spike))[0][0]
    img = img[:, loc_x_spike[idx]:loc_x_spike[idx+1]]
    return img


def remove_lines(img):
    # 縦横の線を削除
    vp = np.sum((img != 0).astype(np.uint8), axis=0)
    loc_x_spike = np.where(vp > np.max(vp[:])*0.9)[0]
    for x in loc_x_spike:
        line_color = (0, 0, 0)
        cv2.line(img, (x, 0), (x, img.shape[0]), line_color, 3)

    hp = np.sum((img != 0).astype(np.uint8), axis=1)
    loc_y_spike = np.where(hp > np.max(hp[:])*0.9)[0]
    for y in loc_y_spike:
        line_color = (0, 0, 0)
        cv2.line(img, (0, y), (img.shape[1], y), line_color, 3)

    return img


def threshold_image(img, threshold):
    """
    画像の2値化
    :param img:
    :param threshold: 自動設定の場合は0
    :return:
    """

    # ぼかし処理
    img = cv2.GaussianBlur(img,(5,5),0)
    # グレースケール化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2値化閾値自動設定の場合
    if threshold == 0:
        threshold_type = cv2.THRESH_OTSU
    else:
        threshold_type = cv2.THRESH_BINARY

    _, img = cv2.threshold(img, threshold, 255, threshold_type)
    img = 255 - img
    return img


if __name__ == "__main__":
    main()