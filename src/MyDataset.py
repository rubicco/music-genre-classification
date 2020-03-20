import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gensim.downloader as api
import sys
import spacy
import os
import torch
from datetime import datetime

from torch.utils.data import Dataset

settings = {
    "embedding_settings": {
        "embedding_type": "glove",
        "embedding_model": "glove-wiki-gigaword-300",
        "index_or_tokens": "index"
    },
    "train_data": "./data/glove_ids_train.csv.zip",
    "val_data": "./data/glove_ids_val.csv.zip",
    "test_data": "./data/glove_ids_test.csv.zip"
}


class MyDataset(Dataset):
    def __init__(self,
                 path_data,
                 max_seq_len=250):
        self.max_seq_len = max_seq_len
        self.X, self.y_id, self.y_label = self._load_data(path_data)

    def _load_data(self, data_path):
        data = pd.read_csv(data_path, index_col=[0])
        X = [[int(id) for id in row.split()] for row in data.lyrics.values]
        y_label = data.genre.values
        y_id = data.genre_id.values
        return (X, y_id, y_label)

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        _X = np.zeros(self.max_seq_len, dtype=np.int64)
        # all indexes are incremented since we have additional Padding token is added
        # to the model embeddings for index 0 in the model definitions
        item = self.X[index]
        max_i = self.max_seq_len if self.max_seq_len < len(item) else len(item)
        _X[:max_i] = np.int64(item[:max_i]) + 1
        return (index, _X, self.y_id[index], self.y_label[index])


def _batch_to_tensor(batch):
    index, _X, target_id, target_label = zip(*batch)
    return torch.LongTensor(_X), torch.LongTensor(target_id)


class MyDataset_WithPreprocess(Dataset):
    def __init__(self,
                 path_data,
                 emb_model,
                 emb_type,
                 max_seq_len=250,
                 input_type="index",
                 store_processed=False,
                 output_dir=""):
        """
        :param path_data:
            path for data location
        :param emb_model:
            a gensim.models.keyedvectors object for glove or word2vec embeddings
            or BertTokenizer for BERT embeddings
        :param emb_type:
            "gensim" or "bert"
        :param max_seq_len:
            max_seq_len that will be input of the model
            default:250
            smaller sequences will be adjusted with PADDING
        :param input_type:
            "unclean":
                It will expect the sentences as it is in the dataset and it will apply following steps:
                    - pre-processing (lowercase, rm stopwords),
                    - tokenize (split the text)
                    - converting to id
            "clean":
                It will expect the sentences already preprocessed and it will apply only following steps:
                    - tokenize,
                    - converting to id
            "index":
                It will expect the tokens in the sentence already converted to ids for the current
                embedding model and it will apply only following steps:
                    - tokenize (split text, ids are stored as text and separated with whitespace.
        :param output_dir:
            If it is sent, resulting preprocessed data will be stored at that dir as csv.
            If store_preprocessed is False, this parameter will be ignored.
        """
        # if store_preprocess = True: then it will store all indexes and training will be faster
        # else: it will preprocess it during __getitem__ function and this will make training slower...
        self.input_type = input_type
        self.emb_model = emb_model
        self.emb_type = emb_type
        self.store_preprocess = store_processed
        self.max_seq_len = max_seq_len
        self.output_dir = os.path.abspath(output_dir)

        if self.input_type == "unclean":
            self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

        if self.emb_type=="bert":
            self.func_w2i = self._bert_w2id
        elif self.emb_type=="gensim":
            self.func_w2i = self._gensim_w2id
        else:
            sys.exit("Unrecognized emb_type in MyDataset_WithPreprocess Class")



        self.X, self.y_id, self.y_label = self._load_data(path_data)

    def _load_data(self, data_path):
        data = pd.read_csv(data_path, index_col=[0])
        if self.store_preprocess and self.input_type != "index":
            print(f"Store Preprocess mode is activated. \nPreprocessing is started for {self.input_type} settings")
            if self.input_type == "unclean":
                print("Preprocess and tokenize the input...")
                X = [self.preprocess(self.nlp, lyric) for lyric in data.lyrics.values]   # clean the tokens
                print("Convert tokens to ids...")
                X = [self.func_w2i(token_list) for token_list in X]                 # convert tokens to ids
            elif self.input_type == "clean":
                X = [lyric.split() for lyric in data.lyrics.values]                 # clean tokens
                X = [self.func_w2i(token_list) for token_list in X]                 # convert tokens to ids
            else:
                sys.exit("Unrecognized input type in MyDataset_WithPreprocess Class")
            path_out = os.path.join(self.output_dir, f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}_ids.csv.zip")
            print(f"Preprocessing ended! \nResults are saving to: \n\t{path_out}")
            tmp = pd.DataFrame({
                "lyrics": [" ".join([str(i) for i in id_list]) for id_list in X],
                "_lyrics": data.lyrics.values,
                "genre": data.genre.values,
                "genre_id": data.genre_id.values
            })
            tmp.to_csv(path_out, compression="zip")

        else:
            # load what we have in the lyrics directly
            if self.input_type == "index":
                X = [[int(word) for word in lyric.split()] for lyric in data.lyrics.values]
            else:
                X = [[word for word in lyric.split()] for lyric in data.lyrics.values]

        y_label = data.genre.values
        y_id = data.genre_id.values
        return (X, y_id, y_label)

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        _X = np.zeros(self.max_seq_len, dtype=np.int64)
        # all indexes are incremented since we have additional Padding token is added
        # to the model embeddings for index 0 in the model definitions
        item = self.X[index]
        if not self.store_preprocess:
            if self.input_type == "unclean":
                item = self.preprocess(self.nlp, " ".join(item))   # clean tokens
                item = self.func_w2i(item)                        # convert tokens to ids

            elif self.input_type == "clean":
                item = self.func_w2i(item)
            elif self.input_type == "index":
                pass
            else:
                sys.exit("Unrecognized input type in MyDataset_WithPreprocess Class")
        max_i = self.max_seq_len if self.max_seq_len < len(item) else len(item)
        const = 1 if self.emb_type == "gensim" else 0
        _X[:max_i] = np.int64(item[:max_i]) + const
        return index, _X, self.y_id[index], self.y_label[index]

    def _gensim_w2id(self, tokens):
        """
        :param tokens: <list> that contains tokens
        :return: <list> contains ids for the
        """
        ids = [self.emb_model.index2word.index(token)
               for token
               in tokens
               if token in self.emb_model.index2word]
        return ids

    def _bert_w2id(self, tokens):
        """
        :param tokens: <list> that contains tokens
        :return: <list> contains ids for the
        """
        ids = [self.emb_model.encode(token)[0]
               for token in tokens]
        return ids

    @staticmethod
    def preprocess(nlp, text):
        """returns a list of preprocessed tokens (tokens are strings/words)"""
        doc = nlp(text)
        out = [token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_.isalpha()]
        return out
