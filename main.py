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

import const
import my_img

# 参考
# https://qiita.com/not13/items/dcd8c12d64982dc0e819

def main():
    # UIの構築
    st.markdown(const.STYLE, unsafe_allow_html=True)
    st.title('OMR2 - マークシートリーダー')

    st.sidebar.write("""
    ## ● マークシート
    """)
    file_path = st.sidebar.file_uploader("PDFファイル", type="pdf")

    if not file_path:
        # st.sidebar.write("マークシートを読み込んでください")
        st.info("マークシートを読み込んでください", icon="ℹ️")
        return
    else:
        try:
            if st.session_state.file_path != file_path:
                st.cache_data.clear()
        except AttributeError:
            pass

        st.session_state.file_path = file_path

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

    if st.sidebar.checkbox('小さいマークを自動で除外', value=True):
        acceptable_small_size = 0.4
    else:
        acceptable_small_size = st.sidebar.slider('', 0.0, 1.0, 0.4, step=0.05)

    is_double_mark = st.sidebar.checkbox('ダブルマークを許可', value=True)
    str_dimensions = st.sidebar.text_input('各フレームのマーク数(行x列)', value='(4x10), (30x10), (30x10)')
    # タプルのリストに変換
    matches = re.findall(r'\((\d+)[x,](\d+)\)', str_dimensions)
    dim_list = [(int(x), int(y)) for x, y in matches]

    # 処理開始
    download_button = st.sidebar.button("全てのシートを一括処理 🚀")

    if download_button:
        data_list, image_list = do_all_omr(pdf_document, threshold, acceptable_small_size, is_double_mark, dim_list)

        # CSVファイルの作成
        csv = pd.DataFrame(data_list).to_csv(index=False)
        csv_b64 = base64.b64encode(csv.encode()).decode()

        # PDFファイルの作成
        pdf_doc = fitz.open()
        for image in image_list:
            _, jpg_buf = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 50))
            page = pdf_doc.new_page()
            page.insert_image(page.rect, stream=jpg_buf.tobytes())
        pdf_b64 = base64.b64encode(pdf_doc.tobytes()).decode()

        st.balloons()
        st.success("全てのシートの読み取りが完了しました", icon="✅")
        st.markdown(const.RESULTS.format(csv_b64=csv_b64, pdf_b64=pdf_b64), unsafe_allow_html=True)

        st.button('戻る')
    else:
        # プレビュー
        table, img = do_omr(pdf_document, page, threshold, acceptable_small_size, is_double_mark, dim_list)

        # 加工済み画像の表示
        st.write("### マーク検出結果プレビュー")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=img_width)

        # 結果の表示
        st.write("### マーク読み取り結果")
        df = pd.DataFrame(table,
                          index=(range(1, len(dim_list)+1)),
                          columns=(range(1, max(dim_list)[0]+1)))
        st.table(df)


@st.cache_data
def do_all_omr(_pdf_document, threshold, acceptable_small_size, is_double_mark, dim_list):
    data_list = []
    image_list = []
    for page in range(1, _pdf_document.page_count+1):
        table_list, image = do_omr(_pdf_document, page, threshold, acceptable_small_size, is_double_mark, dim_list)
        # 1次元リストに変換
        table_list = [item for sublist in table_list for item in sublist]
        data_list.append(table_list)
        image_list.append(image)

    return data_list, image_list


@st.cache_data
def do_omr(_pdf_document, page, threshold, acceptable_small_size, is_double_mark, dim_list):
    # 処理開始
    table_list = []
    img = get_image_from_pdf(_pdf_document, page)
    img = correct_tilt(img)
    frame_list = find_frames(img)

    if len(frame_list) == 0:
        st.error("エラー！ 解答枠が検出できませんでした。", icon="❌")
        st.stop()

    for i, frame in enumerate(frame_list):
        mark_list, frame_img = find_marks(img, frame, threshold, acceptable_small_size)

        try:
            _ = dim_list[i]
        except IndexError:
            st.error("解答枠の個数({})に対してマーク数設定(nxm)の数({})が不足しています"
                       .format(len(frame_list), len(dim_list)), icon="❌")
            st.stop()

        frame_width = np.max(frame[:,0]) - np.min(frame[:,0])
        frame_height = np.max(frame[:,1]) - np.min(frame[:,1])
        data = decode_marks((frame_height, frame_width), mark_list, dim_list[i], is_double_mark)
        table_list.append(data)

        # 検出したマークの描画
        frame_origin = (np.min(frame[:,0]), np.min(frame[:,1]))
        for mark in mark_list:
            for pt in mark:
                pt[0] += frame_origin

            cv2.drawContours(img, [mark], -1, (0, 0, 255), 2)

    # 解答枠の描画
    for i,c in enumerate(frame_list):
        cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
        cv2.putText(img, str(i), (np.min(c[:,0]), np.min(c[:,1])-20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # cv2.circle(img, (np.min(c[0,0]), np.min(c[0,1])), 5, (255, 0, 0), -1)
        # cv2.circle(img, (np.min(c[1,0]), np.min(c[1,1])), 5, (0, 255, 0), -1)
        # cv2.circle(img, (np.min(c[2,0]), np.min(c[2,1])), 5, (0, 0, 255), -1)

    return table_list, img


def get_image_from_pdf(pdf_document, page):
    """
    PDFドキュメントの指定されたページから画像を読み込みます。

    Args:
        pdf_document (fitz.Document): PDFドキュメントオブジェクト。
        page (int): 読み込むページ番号（1から始まる）。

    Returns:
        numpy.ndarray: 読み込まれた画像。
    """

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

    if my_img.is_binary_image(img):
        st.warning("2値化された画像です。なるべくカラーで読み込んでください。", icon="⚠️")

    elif my_img.is_gray_image(img):
        st.warning("グレースケール画像です。なるべくカラーで読み込んでください。", icon="⚠️")
        # st.warning("グレースケール画像です。なるべくカラーで読み込んでください。", icon=":material/warning:")

    # elif my_img.is_color_image(img):
    #    st.write("カラー画像です。")

    width = 2000
    img = cv2.resize(img, (int(img.shape[1]*width/img.shape[0]), width))

    return img


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



def threshold_image(image, threshold):
    """
    画像の2値化
    :param image:
    :param threshold: 自動設定の場合は0
    :return:
    """
    img = cv2.GaussianBlur(image,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold == 0:
        threshold_type = cv2.THRESH_OTSU
    else:
        threshold_type = cv2.THRESH_BINARY
    _, img = cv2.threshold(img, threshold, 255, threshold_type)
    return 255 - img


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

    frame_contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contours = imutils.grab_contours(frame_contours)
    cnt_list = []

    if len(frame_contours) > 0:
        for c in frame_contours:
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


@st.cache_data
def find_marks(image, frame, threshold, acceptable_small_size = 0.4):
    """
    しきい値処理とノイズ除去を行った後、画像の指定されたフレーム内のマークを検出します。

    Args:
        image (numpy.ndarray): 入力画像。
        frame (numpy.ndarray): 画像内のフレームの座標。
        threshold (int): 2値化のためのしきい値。0の場合は大津の方法を使用。
        acceptable_small_size (float): マークとして認識する最小のサイズ。

    Returns:
        tuple: 以下を含むタプル:
            - list: 検出されたマークを表す輪郭のリスト。
            - numpy.ndarray: 処理されたフレーム画像。
    """
    warped_img = imutils.perspective.four_point_transform(image, frame)
    bin_img = threshold_image(warped_img, threshold)
    bin_img = my_img.remove_lines(bin_img)

    # 収縮・膨張によりノイズを除去
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.erode(bin_img, kernel, iterations=2)
    bin_img = cv2.dilate(bin_img, kernel, iterations=2)

    # マークの検出
    mark_contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mark_contours = imutils.grab_contours(mark_contours)
    if len(mark_contours) == 0:
        return [], warped_img

    # 線状の領域を削除
    results = []
    for c in mark_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if 0.5 < ar < 2:
            results.append(c)
    mark_contours = results

    # acceptable_small_sizeより小さいマークを削除
    max_area = np.max([cv2.contourArea(c) for c in mark_contours])
    mark_contours = [c for c in mark_contours
                     if cv2.contourArea(c) > max_area * acceptable_small_size]

    # マークの並び替え（左上から右下へ）
    mark_contours = sorted(mark_contours, key=lambda arg: (np.mean(arg[:]), np.mean(arg[:0])))
    return mark_contours, warped_img


def decode_marks(frame_dim, mark_list, mark_array_dim, is_double_mark):
    """
    指定された解答欄の寸法とマークリストに基づいてマークをデコードする。

    Args:
        frame_dim (tuple): 解答欄の寸法 (高さ, 幅)。
        mark_list (list): 検出されたマークのリスト、それぞれのマークはその重心で表される。
        mark_array_dim (tuple): マーク配列の寸法 (行, 列)。
        is_double_mark (bool): ダブルマークを許可するかどうかのフラグ。

    Returns:
        list: 各行のデコードされたマーク文字列のリスト。ダブルマークが許可されていない場合、
              複数のマークがある行は 'X' でマークされます。
    """

    mark_index = {char: index for index, char in enumerate(const.MARK)}

    frame_height, frame_width = frame_dim
    mark_rows, mark_cols = mark_array_dim

    data_list = [""] * mark_rows

    mark_centroid_list = [np.mean(mark, axis=0)[0] for mark in mark_list]
    mark_centroid_list = sorted(mark_centroid_list, key=lambda arg: arg[1])
    for mark in mark_centroid_list:
        x, y = mark
        # マークがどのセルに属するか
        row = int((y/frame_height)*mark_rows)
        col = int((x/frame_width)*mark_cols)
        if is_double_mark:
            # ダブルマークを許可
            data_list[row] += const.MARK[col]
        else:
            if data_list[row] == "":
                data_list[row] = const.MARK[col]
            else:
                data_list[row] = "X"

    # マーク番号の並び替え("X"が含まれている場合はそのまま)
    return ["X" if "X" in data else ''.join(sorted(data, key=lambda arg: mark_index[arg]))
            for data in data_list]


if __name__ == "__main__":
    main()