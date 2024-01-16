
import os
import pandas as pd
import numpy as np
import re
import torch
import shutil
import string

from pathlib import Path
from glob import glob
from tqdm import tqdm
from rouge import Rouge
from typing import List, Union

puncts = string.punctuation
rouge = Rouge(metrics=['rouge-l'])


def read_txt_file(file_path: Union[Path, str]):
    f = open(file_path, 'r')
    return f.read()


def preprocess_corpus_files(corpus_path: Union[Path, str], output_path: Union[Path, str] = 'corpus_preproc'):
    """From the original corpus, try to recover the original name of each file (the Vietnamese name of the disease)."""
    output_path = Path(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in glob(f'{corpus_path}/*'):
        file = Path(file)
        content = read_txt_file(file)
        # Most of the files contain the name of the disease at the start of the file.
        # Use regex to retrieve that.
        new_filename = re.findall(r"\s> \n(.*?):", content)
        # In case we fail, keep the original name.
        # We manually change the file name later.
        new_filename = new_filename[0] if len(new_filename) > 0 else file.stem
        # Some diseases name contains "/" so the code will fail, try to catch that.
        try:
            shutil.copyfile(file, output_path / new_filename)
        except:
            shutil.copyfile(file, output_path / file.stem)

    new_manual_names = {
        'ngua-hau-mon': 'Ngứa hậu môn',
        'u-mau': 'U máu',
        'hiem-muon': 'Hiếm muộn',
        'seo-ro': 'Sẹo rỗ',
        'u-nhu-sinh-duc': 'U nhú sinh dục',
        'an-khong-tieu': 'Ăn không tiêu',
        'chong-mat': 'Chống mặt',
        'mo-vu-day-dac': 'Mô vú dày đặc',
        'liet-mat': 'Liệt mặt',
        'ra-mo-hoi-tay-chan-nhieu': 'Ra mồ hôi tay chân nhiều',
        'dau-dau-sau-gay': 'Đau đầu sau gáy',
        'benh-thoat-vi-ben': 'Bệnh thoát vị bẹn',
        'xo-cung-bi-toan-the': 'Xơ cứng bì toàn thể',
        'dong-mach-canh': 'Động mạch cảnh',
        'tiet-dich-num-vu': 'Tiết dịch núm vú',
        'dot-quy': 'Đột quỵ',
        'benh-nhiem-san-cho': 'Bệnh nhiễm sán chó',
        'gan-to': 'Gan to',
        'benh-giun-luon': 'Bệnh giun lươn',
        'nam-xuong-bi-chong-mat': 'Nằm xuống bị chóng mặt',
        'nghe-kem': 'Nghe kém',
        'yeu-sinh-ly': 'Yếu sinh lý',
        'rau-bam-mep': 'Rau bám mép',
        'u-tuyen-giap': 'U tuyến giáp',
        'tui-phinh-mach-mau-nao': 'Túi phình mạch máu não',
        'hoi-chung-suy-giam-mien-dich': 'Hội chứng suy giảm miễn dịch',
        'omicron-tang-hinh': 'Omicron tàng hình',
        'dot-quy-thieu-mau-cuc-bo': 'Đột quỵ thiếu máu cục bộ',
        'benh-crohn': 'Bệnh crohn',
        'truyen-mau-song-thai': 'Truyền máu song thai',
        'roi-loan-lipid-mau': 'Rối loạn lipid',
        'mun-cam': 'Mụn cám',
        'benh-ho': 'Ho',
        'dau-nua-dau-sau': 'Đau nửa đầu sau',
        'rung-toc': 'Rụng tóc',
        'thai-luu': 'Thai lưu',
        'dau-dau-van-mach': 'Đau đầu vận mạch',
        'benh-lau': 'Bệnh lậu',
        'ro-hau-mon': 'Bệnh rò hậu môn',
        'nam-da': 'Nám da',
        'dau-nua-dau-phia-truoc': 'Đau nửa đầu phía trước',
        'dau-dinh-dau': 'Đau đỉnh đầu',
        'hoi-chung-loi-thoat-long-nguc': 'Hội chứng lối thoát lồng ngực',
        'mun-boc': 'Mụn bọc',
        'vo-sinh-hiem-muon': 'Vô sinh hiếm muộn',
        'u-tuyen-tung': 'U tuyến tùng',
        'benh-nhuoc-co': 'Bệnh nhược cơ',
        'con-go-chuyen-da-gia': 'Chuyển dạ giả',
        'om-nghen-khi-mang-thai': 'Ốm nghén khi mang thai',
        'mun-an': 'Mụn ẩn',
        'hoi-chung-swyer': 'Hội chứng Swyer',
        'benh-san-day': 'Bệnh sán dây',
        'viem-tai-giua': 'Viêm tai giữa',
    }
    # For files that can't rename using file content, use the manual map.
    for file in glob(str(output_path/'*')):
        if '-' in file:
            old_name = Path(file).stem
            if old_name in new_manual_names:
                os.rename(file, output_path/new_manual_names[old_name])


def preprocess_option(text: str) -> str:
    """Remove question key (if exists). For example: A. Bệnh đái tháo đường => Bệnh đái tháo đường"""
    if not isinstance(text, str) or text == '':
        return text
    pat = r'[A-F]\.'
    if len(re.findall(pat, text)) > 0:
        return re.sub(pat, '', text).strip()
    return text.strip()


def merge_options(row):
    """Merge all options into an unified string."""
    option_str = row['option_1_pp']
    for i in range(2, 7):
        try:
            option_str += ' ' + row[f'option_{i}_pp']
        except:
            continue
    return option_str


def get_topn_disease_for_query(query: List, diseases: List, n: int = 5, exclude: List | None = None):
    """
    For each query, find the top-n related disease by:
        1. Compute the ROUGE-L score between the query and all diseases in corpus.
        2. Get top-n diseases with highest score.
    """
    if not exclude:
        exclude = []
    r_scores = []

    for disease in diseases:
        if disease in exclude:
            score = 0
        else:
            score = rouge.get_scores(' '.join(query), ' '.join(disease))[
                0]['rouge-l']['r']
        r_scores.append(score)
    topks_r_scores = torch.tensor(r_scores)

    top_scores, top_indices = torch.topk(topks_r_scores, n)
    return top_indices.cpu(), top_scores


def hybrid_search_topn_diseases(disease_files, diseases, questions, answers, ids, n=5):
    """
    Based on the file names, retrive top-n most related files to a question AND answer.
    """
    topn_diseases_dict = {}

    for idx, question in enumerate(tqdm(questions)):
        # Try to find top-n files using the question first.
        topn_indices_q, topn_scores_q = get_topn_disease_for_query(
            question, diseases, n=5)
        topn_diseases_q = [disease_files[j].split(
            '/')[-1] for j in topn_indices_q]
        print('Question:', ' '.join(question))
        print('Top-n from question:', topn_diseases_q)
        print('\n')

        # If max score is too low, use proposed answers as query instead.
        if max(topn_scores_q) <= 0.5:
            answer = answers[idx]
            # Exclude files that already searched.
            topn_indices_a, topn_scores_a = get_topn_disease_for_query(
                answer, diseases, n=5, exclude=[diseases[i] for i in topn_indices_q])
            topn_diseases_a = [disease_files[j].split(
                '/')[-1] for j in topn_indices_a]

            # Merge topk from answers with topk from questions.
            topn_diseases_q = topn_diseases_q + topn_diseases_a
            topn_scores_merged = torch.cat([topn_scores_q, topn_scores_a])
            topn_scores_q, topn_indices = torch.topk(topn_scores_merged, 5)
            topn_diseases_q = [topn_diseases_q[j] for j in topn_indices.cpu()]

            # print('Answer:', ' '.join(answer), answer)
            # print('Top-n from answer:', topn_diseases_a)
            # print('\n')

            # print('Top-n from Hybrid:', topn_diseases_q)
            # print('=' * 50)

        if idx not in topn_diseases_dict:
            topn_diseases_dict[ids[idx]] = {
                'topn_disease': [],
                'topn_score': []
            }
        topn_diseases_dict[ids[idx]]['topn_disease'] = topn_diseases_q
        topn_diseases_dict[ids[idx]]['topn_score'] = topn_scores_q.tolist()

    return topn_diseases_dict
