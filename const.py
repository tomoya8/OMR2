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
PDF_JPEG_QUALITY = 85

OMR_IMAGE_PROCESSING_WIDTH = 2000
IGNORE_FRAMES_SMALLER_THAN = 0.02

# aspect ratio of mark
DEFAULT_MARK_SMALL_LIM = 0.4
MARK_MIN_AR = 0.5
MARK_MAX_AR = 2.0

MARK_LAYOUT = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")

STYLE = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
<style>
.my-red {
    color: red;
}
.my-green {
    color: green;
}
</style>
"""

RESULTS = """
### <i class="far fa-angle-double-down"></i> ダウンロード

- <a href="data:file/csv;base64,{csv_b64}" download="{csv_name}">マーク結果一覧</a> <i class="fas fa-file-csv fa-lg my-green"></i>
- <a href="data:file/pdf;base64,{pdf_b64}" download="{pdf_name}">マーク検出画像</a> <i class="fas fa-file-pdf fa-lg my-red"></i>

"""
