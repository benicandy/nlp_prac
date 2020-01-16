import sentencepiece as spm
import numpy as np
import random
from typing import List, Sequence, Tuple

ENCODER_INPUT_NODE = 'transformer/encoder_input:0'
DECODER_INPUT_NODE = 'transformer/decoder_input:0'
IS_TRAINING_NODE = 'transformer/is_training:0'

"""
:param batch_num:  バッチの数
:param batch_size: 文の数
:param length: 文中の単語の数
"""

class BatchGenerator:
    def __init__(
            self,
            max_length=50,
            spm_model_path: str = 'transformer/preprocess/spm_natsume.model'
    ) -> None:
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.bos = self.sp.piece_to_id('<s>')
        self.eos = self.sp.piece_to_id('</s>')
        self.pad = 0

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def load(self, file_path: str) -> None:
        with open(file_path, encoding='utf-8') as f:
            # 一文区切りのリスト: [length]
            lines = [line.strip() for line in f.readlines()]
        self.data = self._create_data(lines)

    def get_batch(self, batch_size: int = 128, shuffle=True):
        while True:
            if shuffle:
                random.shuffle(self.data)
            # データをバッチサイズで分割: [batch_num, batch_size, length]
            raw_batch_list = self._split(self.data, batch_size)
            # バッチごとに訓練データを生成
            for raw_batch in raw_batch_list:  # [batch_size, length]
                questions, answers = zip(*raw_batch)  # [batch_size, length]
                yield {
                    ENCODER_INPUT_NODE: self._convert_to_array(questions),
                    DECODER_INPUT_NODE: self._convert_to_array(answers),
                    IS_TRAINING_NODE: True,
                }

    def _create_data(self, lines: Sequence[str]) -> List[Tuple[List[int], List[int]]]:
        # 一文を単語区切りにする -> id に変換 -> questions と answers のタプルのリスト生成
        questions = [self._create_question(line) for line in lines[:-1]]
        answers = [self._create_answer(line) for line in lines[1:]]
        return list(zip(questions, answers))

    # encoder への入力データ
    def _create_question(self, sentence) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)  # sentence to id
        return ids[:self.max_length]

    # decoder への入力データ
    def _create_answer(self, sentence: str) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)  # sentence to id
        return [self.bos] + ids[:self.max_length - 2] + [self.eos]

    # 入力データをバッチサイズに分割
    def _split(self, nd_list: Sequence, batch_size: int) -> List[List]:
        return [list(nd_list[i - batch_size:i]) for i in range(batch_size, len(nd_list) + 1, batch_size)]

    # list(文集合) の中の list(単語集合) の最大サイズに合わせて array を生成
    def _convert_to_array(self, id_list_list: Sequence[Sequence[int]]) -> np.ndarray:
        max_len = max([len(id_list) for id_list in id_list_list])
        # 最大サイズに届かない文には足りない個数だけ pad を追加
        return np.array(
            [list(id_list) + [self.pad] * (max_len - len(id_list)) for id_list in id_list_list],
            dtype=np.int32,
        )