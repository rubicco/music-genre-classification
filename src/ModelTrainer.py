import os
import sys
import time
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gensim.downloader as api
from transformers import AutoTokenizer

from src.model.MyLSTM import MyLSTM
from src.Util.CSVLogger import CSVLogger
from src.model.ModelBertLSTM import MyLSTM_Bert
from src.MyDataset import MyDataset_WithPreprocess as MyDataset, _batch_to_tensor


glove_settings = {
    "run_name": "GloveLSTM",
    "output_dir": "./output/GloveRun",
    "load_checkpoint": "",
    "mode": "train",  # train, eval

    "embedding_settings": {
        "embedding_type": "gensim",     # gensim, bert
        ## embeding_model: ["glove-wiki-gigaword-300", "word2vec-google-news-300", "distilbert-base-uncased"]
        "embedding_model": "glove-wiki-gigaword-300"
    },

    "data_settings": {
        "train_data": "./data/glove_ids_train.csv.zip",
        "val_data": "./data/glove_ids_val.csv.zip",
        "test_data": "./data/glove_ids_test.csv.zip",
        "max_seq_len": 200,
        "input_type": "index",    # index, clean, unclean
        "store_processed": False,
        "batch_size": 96
    },

    "model_settings": {
        "lstm_layer": 2,
        "hidden_dim": 128,
        "target_size": 7,
        "dropout_prob": 0.2
    },

    "train_settings": {
        "epochs": 100,
        "learning_rate": 0.000005,
        "grad_clip": 5
    }
}

word2vec_settings = {
    "run_name": "Word2VecLSTM",
    "output_dir": "./output/Word2VecRun",
    "load_checkpoint": "",
    "mode": "train",  # train, eval

    "embedding_settings": {
        "embedding_type": "gensim",       # gensim, bert
        ## embeding_model: ["glove-wiki-gigaword-300", "word2vec-google-news-300", "distilbert-base-uncased"]
        "embedding_model": "word2vec-google-news-300"
    },

    "data_settings": {
        "train_data": "./data/glove_ids_train.csv.zip",
        "val_data": "./data/glove_ids_val.csv.zip",
        "test_data": "./data/glove_ids_test.csv.zip",
        "max_seq_len": 200,
        "input_type": "index",        # index, clean, unclean 
        "store_processed": False,
        "batch_size": 96
    },

    "model_settings": {
        "lstm_layer": 2,
        "hidden_dim": 128,
        "target_size": 7,
        "dropout_prob": 0.2
    },

    "train_settings": {
        "epochs": 100,
        "learning_rate": 0.000005,
        "grad_clip": 5
    }
}


bert_settings = {
    "run_name": "BertLSTM",
    "output_dir": "./output/BertRun",
    # if load checkpoint is not empty (""), trainer will try to import checkpoint!
    "load_checkpoint": "",
    "mode": "train",  # train, eval

    "embedding_settings": {
        ## embedding_type can be: ["gensim" | "bert"]
        "embedding_type": "bert",
        ## embeding_model can be: ["glove-wiki-gigaword-300" | "word2vec-google-news-300" | "distilbert-base-uncased"]
        "embedding_model": "distilbert-base-uncased"
    },

    "data_settings": {
        "train_data": "./data/bert_ids_train.csv.zip",
        "val_data": "./data/bert_ids_val.csv.zip",
        "test_data": "./data/bert_ids_test.csv.zip",
        "max_seq_len": 200,
        "input_type": "index",
        "store_processed": False,
        "batch_size": 96
    },

    "model_settings": {
        "lstm_layer": 2,
        "hidden_dim": 128,
        "target_size": 7,
        "dropout_prob": 0.2,
        "train_bert": False
    },

    "train_settings": {
        "epochs": 100,
        "learning_rate": 0.000005,
        "grad_clip": 5
    }
}


class ModelTrainer:
    def __init__(self, settings, emb_model=None):
        # ################################ #
        # IDENTIFY THE DEVICE FOR TRAINING #
        # ################################ #
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.load = settings["load_checkpoint"]
        self.run_mode = settings["mode"]
        if self.load != "" and self.run_mode == "train":
            checkpoint = self.get_chkp(settings["load_checkpoint"])
            self.output_dir = os.path.abspath("/".join(self.load.split("/")[:-2]))
            self.settings = checkpoint["settings"]
        else:
            self.settings = settings
            self.output_dir = settings["output_dir"]
            self.output_dir = os.path.join(self.output_dir,
                                           f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}_{settings['run_name']}")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        self.embedding_model_name = settings["embedding_settings"]["embedding_model"]
        if isinstance(emb_model, type(None)):
            if self.embedding_model_name != "distilbert-base-uncased":
                print(f"embedding model ({self.embedding_model_name}) is loading...")
                self.emb_model = api.load(self.embedding_model_name)
            elif self.embedding_model_name == "distilbert-base-uncased":
                print(f"embedding model ({self.embedding_model_name}) is loading...")
                self.emb_model = AutoTokenizer.from_pretrained(self.embedding_model_name)
        else:
            print(f"embedding model is loaded from the argument.")
            self.emb_model = emb_model

        # ########### #
        # IMPORT DATA #
        # ########### #
        print("Importing Data...")
        self.data_train = MyDataset(path_data=settings["data_settings"]["train_data"],
                                    emb_model=self.emb_model,
                                    emb_type=settings["embedding_settings"]["embedding_type"],
                                    max_seq_len=settings["data_settings"]["max_seq_len"],
                                    input_type=settings["data_settings"]["input_type"],
                                    store_processed=settings["data_settings"]["store_processed"],
                                    output_dir=self.output_dir)
        self.data_val = MyDataset(path_data=settings["data_settings"]["val_data"],
                                  emb_model=self.emb_model,
                                  emb_type=settings["embedding_settings"]["embedding_type"],
                                  max_seq_len=settings["data_settings"]["max_seq_len"],
                                  input_type=settings["data_settings"]["input_type"],
                                  store_processed=settings["data_settings"]["store_processed"],
                                  output_dir=self.output_dir)
        self.data_test = MyDataset(path_data=settings["data_settings"]["test_data"],
                                   emb_model=self.emb_model,
                                   emb_type=settings["embedding_settings"]["embedding_type"],
                                   max_seq_len=settings["data_settings"]["max_seq_len"],
                                   input_type=settings["data_settings"]["input_type"],
                                   store_processed=settings["data_settings"]["store_processed"],
                                   output_dir=self.output_dir)

        # ############ #
        # Data Loaders #
        # ############ #
        print("Creating DataLoaders...")
        self.batch_size = settings["data_settings"]["batch_size"]
        self.dataloader_train = DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=_batch_to_tensor,
                                           drop_last=True)

        self.dataloader_val = DataLoader(self.data_val,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         collate_fn=_batch_to_tensor,
                                         drop_last=True)

        self.dataloader_test = DataLoader(self.data_test,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          collate_fn=_batch_to_tensor,
                                          drop_last=True)

        # #################### #
        # INITIALIZE THE MODEL #
        # #################### #
        print("Initializing Model...")
        if settings["embedding_settings"]["embedding_type"]=="bert":
            self.model = MyLSTM_Bert(lstm_layers=settings["model_settings"]["lstm_layer"],
                                     hidden_dim=settings["model_settings"]["hidden_dim"],
                                     target_size=settings["model_settings"]["target_size"],
                                     dropout_prob=settings["model_settings"]["dropout_prob"],
                                     device=self.device,
                                     bert_pretrained=settings["embedding_settings"]["embedding_model"],
                                     seq_len=settings["data_settings"]["max_seq_len"],
                                     train_bert=settings["model_settings"]["train_bert"])
        else:
            self.model = MyLSTM(emb_vectors=self.emb_model.vectors,
                                lstm_layers=settings["model_settings"]["lstm_layer"],
                                hidden_dim=settings["model_settings"]["hidden_dim"],
                                target_size=settings["model_settings"]["target_size"],
                                dropout_prob=settings["model_settings"]["dropout_prob"],
                                device=self.device,
                                seq_len=settings["data_settings"]["max_seq_len"])
        self.model = self.model.to(self.device)
        self.loss_function = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=settings["train_settings"]["learning_rate"])

        # ################# #
        # TRAINING SETTINGS #
        # ################# #
        if self.load != "" and self.run_mode == "eval":
            self.eval(data_loader=self.dataloader_test)
        elif self.load != "" and self.run_mode == "train":
            # load the last training
            print(f"The Model is loading... {self.load}")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.epochs = settings["train_settings"]["epochs"]
            self.batch_counter = 0
            self.epoch_counter = checkpoint["epoch"]
            self.grad_clip = settings["train_settings"]["grad_clip"]
            self.valid_loss_min = checkpoint["valid_loss_min"]
            self.valid_acc_max = checkpoint["valid_acc_max"]
            self.metrics = checkpoint["metrics"]
            self.csv_logger = CSVLogger(self.metrics.keys())
            for i in range(len(self.metrics["train_loss"])):
                self.csv_logger.log({
                    "train_loss": self.metrics["train_loss"][i],
                    "val_loss": self.metrics["val_loss"][i],
                    "train_acc": self.metrics["train_acc"][i],
                    "val_acc": self.metrics["val_acc"][i]
                })
        elif self.load == "":
            self.epochs = settings["train_settings"]["epochs"]
            self.batch_counter = 0
            self.epoch_counter = 0
            self.grad_clip = settings["train_settings"]["grad_clip"]
            self.valid_loss_min = np.Inf
            self.valid_acc_max = np.NINF
            # Initialize metric container
            self.metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": []
            }
            self.csv_logger = CSVLogger(self.metrics.keys())
            with open(os.path.join(self.output_dir, "settings.json"), "w") as f:
                json.dump(settings, f)
        else:
            sys.exit("load and mode Settings are not identified!")

    def train(self):
        print_step_every = 100
        self.model.switch_train()
        while self.epoch_counter < self.epochs:
            self.epoch_counter += 1
            epoch = self.epoch_counter
            # initialize intermediate metrics
            self.batch_counter = 0
            training_loss = 0
            training_acc = 0
            # time vars
            ep_start_time = time.time()
            temp_time = time.time()
            # initialize hiddens
            h = self.model.init_hidden(self.batch_size)
            for inputs, labels in self.dataloader_train:
                self.batch_counter += 1

                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output, h = self.model.forward(inputs, h)

                preds = torch.max(output, dim=1)[1]
                loss = self.loss_function(output, labels)

                training_loss += loss.item()
                training_acc += torch.sum(preds == labels).item() / labels.shape[0]

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                if self.batch_counter % print_step_every == 0 or self.batch_counter == 1:
                    print("Epoch: {}/{}...".format(epoch + 1, self.epochs),
                          "Step: {}...".format(self.batch_counter),
                          "Loss: {:.6f}...".format(training_loss / self.batch_counter),
                          "Acc: {:.2f}".format(training_acc / self.batch_counter),
                          "Time: {}... secs".format(time.time() - temp_time))
                    temp_time = time.time()

                # ##### END OF EPOCH ##### #
                if self.batch_counter == len(self.dataloader_train) - 1:
                    val_loss, val_acc = self.eval(self.dataloader_val)

                    row = {
                        "train_loss": training_loss / len(self.dataloader_train),
                        "train_acc": training_acc / len(self.dataloader_train),
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    }
                    self._log_metrics(row)

                    print("Epoch: {}/{}...".format(epoch + 1, self.epochs),
                          "Step: {}...".format(self.batch_counter),
                          "Loss: {:.6f}...".format(row["train_loss"]),
                          "Acc: {:.2f}".format(row["train_acc"]),
                          "Val Loss: {:.6f}".format(row["val_loss"]),
                          "Val Acc: {:.2f}".format(row["val_acc"]))
                    # ## SAVE MODEL ## #
                    # save best validation accuracy
                    if self.valid_acc_max < row["val_acc"]:
                        self.valid_acc_max = row["val_acc"]
                        # self.save_model(fname=f"minAcc_epoch")
                    # save best validation loss
                    if self.valid_loss_min > row["val_loss"]:
                        self.valid_loss_min = row["val_loss"]
                        self.save_model(fname=f"minLoss_epoch")
                    # save last epoch
                    self.save_model(fname="last_epoch")

                    ep_end_time = time.time()
                    print("Epoch: {} Ended in {} secs...".format(epoch, ep_end_time - ep_start_time))


    def _log_metrics(self, row):
        self.metrics["train_loss"].append(row["train_loss"])
        self.metrics["train_acc"].append(row["train_acc"])
        self.metrics["val_loss"].append(row["val_loss"])
        self.metrics["val_acc"].append(row["val_acc"])
        self.csv_logger.log(row)

    def eval(self, data_loader=None):
        test_loss = 0
        test_acc = 0
        test_h = self.model.init_hidden(self.batch_size)

        self.model.eval()
        data_loader = self.dataloader_test if isinstance(data_loader, type(None)) else data_loader
        for inp, lab in data_loader:
            test_h = tuple([each.data for each in test_h])

            inp, lab = inp.to(self.device), lab.to(self.device)
            out, test_h = self.model(inp, test_h)

            preds = torch.max(out, dim=1)[1]
            loss = self.loss_function(out, lab)

            test_loss += loss.item()
            test_acc += torch.sum(preds == lab).item() / lab.shape[0]
        self.model.switch_train()

        loss = test_loss / len(data_loader)
        acc = test_acc / len(data_loader)
        return loss, acc

    def predict(self, data_loader=None):
        all_preds = []
        all_y = []
        test_h = self.model.init_hidden(self.batch_size)

        self.model.eval()
        data_loader = self.dataloader_test if isinstance(data_loader, type(None)) else data_loader
        for inp, lab in data_loader:
            test_h = tuple([each.data for each in test_h])

            inp, lab = inp.to(self.device), lab.to(self.device)
            out, test_h = self.model(inp, test_h)
            preds = torch.max(out, dim=1)[1]

            all_preds.extend(preds.detach().numpy())
            all_y.extend(lab.detach().numpy())
        self.model.switch_train()

        return np.array(all_preds), np.array(all_y)

    def save_model(self, fname="last_epoch"):
        state = {
            "settings": self.settings,
            "epoch": self.epoch_counter,
            "metrics": self.metrics,
            "valid_acc_max": self.valid_acc_max,
            "valid_loss_min": self.valid_loss_min,
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.model.state_dict()
        }
        o_dir = os.path.join(self.output_dir, "checkpoints")
        if not os.path.exists(o_dir):
            os.makedirs(o_dir)
        path_out = os.path.join(o_dir, f"{fname}.chkp")
        torch.save(state, path_out)

    def load_model(self, chkp_path):
        checkpoint = self.get_chkp(chkp_path)
        print(f"The Model is loading... {self.load}")
        self.output_dir = os.path.abspath("/".join(chkp_path.split("/")[:-2]))
        self.settings = checkpoint["settings"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.epoch_counter = checkpoint["epoch"]
        self.valid_loss_min = checkpoint["valid_loss_min"]
        self.valid_acc_max = checkpoint["valid_acc_max"]
        self.metrics = checkpoint["metrics"]

    def get_chkp(self, chkp_path):
        return torch.load(chkp_path, map_location=self.device)

    def save_plots(self, fname="learningCurve"):
        x = np.arange(len(self.metrics["train_loss"])) + 1
        fig, ax = plt.subplots()
        ax.plot(x, self.metrics["train_loss"], label="train")
        ax.plot(x, self.metrics["val_loss"], label="val")
        ax.set_title('Learning Curve - Loss')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dir, f"{fname}_Loss.png"))

        fig, ax = plt.subplots()
        ax.plot(x, self.metrics["train_acc"], label="train")
        ax.plot(x, self.metrics["val_acc"], label="val")
        ax.set_title('Learning Curve - Accuracy')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dir, f"{fname}_Acc.png"))