import os
import pickle
import re
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from nltk.tokenize import RegexpTokenizer
from mgca.constants import *
from .utils import get_imgs
from .pretrain_dataset import check_element_type
from tqdm import tqdm
from copy import deepcopy
import random
from .transforms import DataTransforms
from transformers import BertTokenizer, MarianMTModel, MarianTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MIMICPretrainingDataset(data.Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        data_pct=1.0,
        imsize=256,
        max_words=160,
        sent_num=3,
        llm_type="bert",
        full_train=False,
        c1=False,
        c2=False,
        c3=False,
        raw_caption=True,
        simple_cap=False,
        class_caption=False,
        cls_prompt=False,
        pred_only=False,
        slip=False,
        prob_diff_img=0.5,
        load_large=False,
        train_orig_mimic=False,
        tt_aug=False,
        train_c2=False,
        pos_sample_only=False,
        train_head=False,
        train_tail=False,
        cap_idx=0,
        c3_cap_aug=False,
        instance_test_cap=False,
        c3_rand_cap=False,
        aug_text=False,
        heavy_aug=False,
        extra_cap=None,
        c3_cap_trans=False,
        gpt_cap=False,
        extract_train=False,
        retrieve_cap_path=None,
        retrieve_idx=0,
        gold_test=False,
        **kwargs,
    ):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.split = split
        self.transform = transform
        self.imsize = imsize
        self.max_words = max_words
        self.llm_type = llm_type
        self.raw_caption = raw_caption
        self.simple_cap = simple_cap
        self.class_caption = class_caption
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.cls_prompt = cls_prompt
        self.pred_only = pred_only
        self.slip = slip
        self.prob_diff_img = prob_diff_img
        self.load_large = load_large
        self.tt_aug = tt_aug
        self.train_orig_mimic = train_orig_mimic
        self.train_c2 = train_c2
        self.train_head = train_head
        self.train_tail = train_tail
        self.pos_sample_only = pos_sample_only
        self.cap_idx = cap_idx
        self.c3_cap_aug = c3_cap_aug
        self.instance_test_cap = instance_test_cap
        self.c3_rand_cap = c3_rand_cap
        self.extra_cap = extra_cap
        self.c3_cap_trans = c3_cap_trans
        self.gpt_cap = gpt_cap
        self.retrieve_cap_path = retrieve_cap_path
        self.extract_train = extract_train
        self.retrieve_idx = retrieve_idx

        self.no_label = False
        if split == "train":
            if full_train:
                self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV_ALL)
                split = "train_full"
            elif self.train_orig_mimic:
                self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV_ORIG)
            else:
                self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
        elif split == "valid":
            if self.train_orig_mimic:
                self.df = pd.read_csv(MIMIC_CXR_VALID_CSV_ORIG)
            else:
                self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
        elif split == "test":
            # if self.pred_only:
            if self.train_orig_mimic:
                self.df = pd.read_csv(MIMIC_CXR_TEST_CSV_ORIG)
                self.no_label = False
            elif gold_test:
                self.df = pd.read_csv(MIMIC_CXR_GOLD_TEST_CSV)
                self.no_label = False
                split = "test_gold"
            elif c1:
                self.df = pd.read_csv(MIMIC_CXR_TEST_CSV_C1)
                # self.no_label = True
                self.no_label = False
            elif c2:
                self.df = pd.read_csv(MIMIC_CXR_TEST_CSV_C2)
                # self.no_label = True
                self.no_label = False
            elif c3:
                self.df = pd.read_csv(MIMIC_CXR_TEST_CSV_C3)
                # self.no_label = True
                self.no_label = False
            else:
                self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
            # else:
            #     if extract_train:
            #         self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV_ALL)
            #     else:
            #         self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)
            self.cls_prompt = True

        if self.train_orig_mimic:
            findings = MIMIC_CXR_14_FINDINGS
            split += "_orig_mimic"
        elif self.train_c2:
            findings = MIMIC_CXR_LT_FINDINGS_C2
            split = split + "_c2"
        elif self.split == "train":
            findings = MIMIC_CXR_LT_FINDINGS
            if self.train_head:
                findings = HEAD_CLS
                split = split + "_head"
            elif self.train_tail:
                findings = TAIL_CLS
                split = split + "_tail"
        elif self.c1:
            findings = MIMIC_CXR_LT_FINDINGS_C1
            split = "test_c1"
            if self.train_head:
                findings = HEAD_CLS
                split = split + "_head"
            elif self.train_tail:
                findings = TAIL_CLS
                split = split + "_tail"
        elif self.c2:
            findings = MIMIC_CXR_LT_FINDINGS_C2
            split = "test_c2"
            if self.train_head:
                findings = [f for f in findings if f in HEAD_CLS]
                split = split + "_head"
            elif self.train_tail:
                findings = [f for f in findings if f in TAIL_CLS]
                split = split + "_tail"
        elif self.c3:
            findings = MIMIC_CXR_LT_FINDINGS_C3
            split = "test_c3"
        elif self.train_head:
            findings = HEAD_CLS
            split = split + "_head"
        elif self.train_tail:
            findings = TAIL_CLS
            split = split + "_tail"
        else:
            findings = MIMIC_CXR_LT_FINDINGS
        self.findings = findings
        print("### Findings: ", findings)

        self.report_df = pd.read_csv(MIMIC_CXR_TEXT_CSV)

        # Use all views for challenge
        # self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(
                MIMIC_CXR_DATA_DIR, x.replace(".jpg", "_resized.jpg")
            )
        )

        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        # Use longer caption for the model when using retrieval
        self.max_words = 196 if self.retrieve_cap_path != None else max_words

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        self.study2path = None
        if self.slip:
            self.study2path = self.build_study2path()
        self.orig_crop_transform = DataTransforms(True, self.imsize)
        if self.tt_aug:
            img_size = self.transform.img_size
            crop_size = self.transform.crop_size
            print("### Using flipped image as input")
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.RandomHorizontalFlip(1.0),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None

        if aug_text:
            try:
                from .transforms import TextTransform
            except ImportError:
                raise ImportError("TextTransform is required for aug_text=True but not found in transforms module")
            if heavy_aug:
                self.text_transform = TextTransform(
                    is_train=(split == "train"),
                    stop_token=self.tokenizer.sep_token,
                    remove_stop_word_prob=0.3,
                    synonym_replacement_prob=0.2,
                    random_swap_prob=0.1,
                    random_deletion_prob=0.1,
                    random_sent_swap_prob=0.1,
                )
            else:
                self.text_transform = TextTransform(
                    is_train=(split == "train"),
                    stop_token=self.tokenizer.sep_token,
                )
        else:
            self.text_transform = None

        if self.instance_test_cap and self.gpt_cap:
            with open("data/mimic_cxr_lt_c3_zs_caption_instance_top3.pkl", "rb") as f:
                self.instance_cap_dict = pickle.load(f)

        if self.retrieve_cap_path != None:
            with open(self.retrieve_cap_path, "rb") as f:
                self.retrieve_cap_dict = pickle.load(f)

    def load_text_data(self, split):
        # get study to captions mapping
        base_filename = f"{split}_captions.pickle"
        if self.simple_cap:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        if self.llm_type != "gpt":
            base_filename = base_filename.replace(".pickle", f"_{self.llm_type}.pickle")
        if self.raw_caption:
            base_filename = base_filename.replace(".pickle", "_raw.pickle")
        if self.extract_train:
            base_filename = "train_full_captions_raw.pickle"
        filepath = os.path.join(MIMIC_CXR_DATA_DIR, base_filename)

        if self.no_label:
            print("### No label for the dataset")
            path2sent = {p: "" for p in self.df[MIMIC_CXR_PATH_COL].tolist()}
        elif not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            print(f"Reuse caption file {filepath}...")
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        if self.extra_cap != None and "train" in split:
            print(
                f"### Add extra caption at {self.extra_cap} (augmented) to the training"
            )
            with open(self.extra_cap, "rb") as f:
                self.extra_path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        path2label = {}
        path2view = {}
        labels = []
        view_is_na = self.df["ViewCodeSequence_CodeMeaning"].isna()
        valid_img_path = set(path2sent.keys())
        for idx, row in self.df.iterrows():
            path = row[MIMIC_CXR_PATH_COL]
            if path in valid_img_path:
                # extract labels
                multi_hot_label = np.zeros(len(self.findings))
                if not self.no_label:
                    for i, col in enumerate(self.findings):
                        if row[col] == 1:
                            multi_hot_label[i] = 1
                # Only train with images with positive annotations
                if self.pos_sample_only and multi_hot_label.sum() == 0:
                    continue
                filenames.append(path)
                path2label[path] = multi_hot_label
                labels.append(multi_hot_label)
                if not view_is_na[idx]:
                    path2view[path] = row["ViewCodeSequence_CodeMeaning"]
                else:
                    path2view[path] = None
        labels = np.array(labels)
        self.path2label = path2label
        self.labels = labels
        self.path2view = path2view

        print("### Valid images: ", len(filenames))
        print("### Class frequency: ", labels.sum(axis=0))

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        view_is_na = self.df["ViewCodeSequence_CodeMeaning"].isna()

        sid_2_col = {sid: col for col, sid in enumerate(self.report_df["study"])}
        annotated_sid = set(sid_2_col.keys())
        all_sid = set(self.df["study_id"].unique())
        sid_no_annotation = all_sid - annotated_sid
        print(f"#### No annotation for {len(sid_no_annotation)} studies")

        # iterrows is not faster than itertuples ...  but it is ok
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""

            if not self.simple_cap:
                sid = row["study_id"]
                if sid in sid_no_annotation:
                    continue
                # provide view info
                cur_view = row["ViewCodeSequence_CodeMeaning"]
                if not view_is_na[idx]:
                    captions += MIMIC_CXR_VIEW_CAPTION.replace("{{VIEW}}", cur_view)

                study_text_row = self.report_df.iloc[sid_2_col[sid]]
                impression = study_text_row["impression"]
                # validate impression and find
                if check_element_type(impression):
                    captions += " "
                    captions += impression

                find = study_text_row["findings"]
                if check_element_type(find):
                    captions += " "
                    captions += find
            if self.class_caption:
                # provide class info
                captions += " " + MIMIC_CXR_LABEL_CAPTION[self.cap_idx]

                label_cnt = 0
                for col in self.findings:
                    if row[col] == 1:
                        captions += col
                        captions += ", "
                    label_cnt += 1

                if label_cnt == 0:
                    continue

            # use space instead of newline
            captions = captions.replace("\n", " ")
            if len(captions) == 0:
                continue

            if self.raw_caption:
                captions = captions.replace("-", "")
                sent_lens.append(len(captions.split(" ")))
                num_sents.append(len(captions.split(".")))
                # replace period with sep_token
                # Every sep_token is a new sentence, use for localized loss
                if "." in captions:
                    captions = captions.replace(".", " " + self.tokenizer.sep_token)
                else:
                    captions = captions + " " + self.tokenizer.sep_token

                # add bos/eos tokens
                captions = (
                    self.tokenizer.cls_token
                    + " "
                    + captions
                    + " "
                    + self.tokenizer.sep_token
                )
                path2sent[row[MIMIC_CXR_PATH_COL]] = [
                    captions,
                ]
            else:
                # split sentences
                splitter = re.compile("[0-9]+\.")
                captions = splitter.split(captions)
                captions = [point.split(".") for point in captions]
                captions = [sent for point in captions for sent in point]

                cnt = 0
                study_sent = []
                # create tokens from captions
                for cap in captions:
                    if len(cap) == 0:
                        continue

                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(cap.lower())
                    # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                    if len(tokens) <= 1:
                        continue

                    # filter tokens for current sentence
                    included_tokens = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0:
                            included_tokens.append(t)

                    if len(included_tokens) > 0:
                        study_sent.append(" ".join(included_tokens))

                    cnt += len(included_tokens)

                if cnt >= 3:
                    sent_lens.append(cnt)
                    num_sents.append(len(study_sent))
                    path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        print("Example text caption: ", list(path2sent.values())[0])

        return path2sent

    def build_study2path(self):
        study2path = {}
        for path in self.filenames:
            sid = path.split("/")[-2]
            if sid not in study2path:
                study2path[sid] = []
            study2path[sid].append(path)
        return study2path

    def __len__(self):
        return len(self.filenames)

    def random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1] - 1):
            if masked_tokens[0][i] == 0:
                break

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = self.tokenizer.mask_token_id

        return masked_tokens

    def get_caption(self, path, series_sents=None):
        if series_sents is None:
            # Use augmented sentence
            if (
                self.extra_cap != None
                and "train" in self.split
                and random.random() < 0.5
            ):
                series_sents = self.extra_path2sent[path]
                print(series_sents)
            else:
                series_sents = self.path2sent[path]

        # Remove sensitive info directly from the sentence
        series_sents = [
            s.replace("___", self.tokenizer.mask_token) for s in series_sents
        ]

        if self.text_transform is not None:
            series_sents = [
                self.text_transform(series_sents[0]),
            ]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        masked_ids = self.random_mask(tokens["input_ids"])
        tokens["masked_ids"] = masked_ids

        return tokens, x_len

    def get_multi_hot_label(self, path):
        return torch.tensor(self.path2label[path])

    def __cls_getitem__(self, index):
        key = self.filenames[index]
        multi_hot_label = self.get_multi_hot_label(key)

        if self.zero_shot_caps is None or self.instance_test_cap:
            zero_shot_caps = []
            zero_shot_caps_len = []

            # translate the caption and then back to English
            if self.c3_cap_trans:
                # prepare the translation model
                device = "cpu"
                model_name = "Helsinki-NLP/opus-mt-en-it"
                forward_tokenizer = MarianTokenizer.from_pretrained(model_name)
                forward_model = MarianMTModel.from_pretrained(model_name).to(device)
                model_name = "Helsinki-NLP/opus-mt-it-en"
                backward_tokenizer = MarianTokenizer.from_pretrained(model_name)
                backward_model = MarianMTModel.from_pretrained(model_name).to(device)

            # Generate caption for each class
            for cls in self.findings:
                if self.gpt_cap:
                    # use caption generated from GPT
                    if self.instance_test_cap:
                        # get dicom from the path
                        dicom = key.split("/")[-1].split(".")[0].replace("_resized", "")
                        # use caption according to test time prediction information
                        captions = self.instance_cap_dict[dicom][cls][self.cap_idx]
                    else:
                        captions = MIMIC_CXR_LT_FINDINGS_C3_GPT_CAPTION[cls][
                            self.cap_idx
                        ]
                    captions.replace("..", ".")
                else:
                    if self.instance_test_cap:
                        if self.retrieve_cap_path != None:
                            base_caption = (
                                self.retrieve_cap_dict[key][self.retrieve_idx]
                                + MIMIC_CXR_LABEL_CAPTION[self.cap_idx]
                            )
                        else:
                            # utilize the test view information
                            view = self.path2view[key]
                            if view != None:
                                base_caption = MIMIC_CXR_VIEW_CAPTION_TEST.replace(
                                    "{{VIEW}}", view
                                )
                            else:
                                base_caption = MIMIC_CXR_LABEL_CAPTION[self.cap_idx]
                    else:
                        base_caption = MIMIC_CXR_LABEL_CAPTION[self.cap_idx]

                    captions = base_caption + cls
                    captions = captions.replace("\n", " ")
                    if self.c3_cap_aug and self.c3:
                        if self.c3_rand_cap and cls in ["Cardiomyopathy", "Osteopenia"]:
                            c3_extra_cap = random.choice(
                                MIMIC_CXR_LT_FINDINGS_C3_EXTRA_SENT[cls]
                            )
                            captions += ", " + c3_extra_cap + "."
                        else:
                            captions += ", such as "
                            extra_list = MIMIC_CXR_LT_FINDINGS_C3_EXTRA[cls]
                            for idx, extra in enumerate(extra_list):
                                if idx > 0:
                                    captions += " and "
                                captions += extra + ","
                    # if not self.instance_test_cap:
                    #     print(cls, ": ", captions)

                # apply translation back augmentation
                if self.c3_cap_trans:
                    forward_tokens = forward_tokenizer(
                        captions, return_tensors="pt", padding=True
                    ).to(device)
                    translated = forward_model.generate(**forward_tokens)[0]
                    translated = forward_tokenizer.decode(
                        translated, skip_special_tokens=True
                    )
                    backward_tokens = backward_tokenizer(
                        translated, return_tensors="pt", padding=True
                    ).to(device)
                    translated_back = backward_model.generate(**backward_tokens)[0]
                    translated_back = backward_tokenizer.decode(
                        translated_back, skip_special_tokens=True
                    )
                    # print("#### Translated: ", translated_back)
                    captions = translated_back

                # caption post processing
                if self.raw_caption:
                    captions = captions.replace("-", "")
                    if "." in captions:
                        captions = captions.replace(".", " " + self.tokenizer.sep_token)
                    else:
                        captions = captions + " " + self.tokenizer.sep_token
                    captions = (
                        self.tokenizer.cls_token
                        + " "
                        + captions
                        + " "
                        + self.tokenizer.eos_token
                    )
                else:
                    captions = captions.lower()
                cap, cap_len = self.get_caption(None, [captions])
                zero_shot_caps.append(cap)
                zero_shot_caps_len.append(cap_len)

            stacked_caps = {}
            for cap in zero_shot_caps:
                for k, v in cap.items():
                    if k not in stacked_caps:
                        stacked_caps[k] = v
                    else:
                        stacked_caps[k] = torch.concat([stacked_caps[k], v], dim=0)
            zero_shot_caps_len = torch.tensor(zero_shot_caps_len)
            self.zero_shot_caps = stacked_caps
            self.zero_shot_caps_len = zero_shot_caps_len
        if self.load_large:
            key = key.replace("mimic-cxr-jpg-resized", "mimic-cxr-jpg-resized-1024")
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, self.zero_shot_caps, self.zero_shot_caps_len, key, multi_hot_label

    def __getitem__(self, index):
        if self.cls_prompt:
            return self.__cls_getitem__(index)
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        if self.load_large:
            key = key.replace("mimic-cxr-jpg-resized", "mimic-cxr-jpg-resized-1024")
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key


class MergeDataset(data.Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        datasets_length = [len(dataset) for dataset in datasets]
        print(f"### Datasets length: {datasets_length}")

    def __len__(self):
        return sum([len(self.datasets[i]) for i in range(len(self.datasets))])

    def __getitem__(self, index):
        dataset_idx = 0
        for i in range(len(self.datasets)):
            if index < len(self.datasets[i]):
                dataset_idx = i
                break
            index -= len(self.datasets[i])
        return self.datasets[dataset_idx].__getitem__(index)


# def multimodal_collate_fn(batch):
#     """sort sequence"""
#     imgs, cap_len, ids, tokens, attention, masked_ids, labels = [], [], [], [], [], [], []
#     up_ids, up_labels, up_attention = [], [], []
#     ext_imgs = []
#     orig_imgs = []
#     path = []
#     tokens_type_id_exist = False
#     eval_mode = False
#     for b in batch:
#         img, cap, cap_l, p, label, up_cap, up_label, orig_img = b
#         if isinstance(img, list):
#             img, ext_img = img
#             ext_imgs.append(ext_img)
#         imgs.append(img)
#         cap_len.append(cap_l)
#         ids.append(cap["input_ids"])
#         up_ids.append(up_cap["input_ids"])
#         if "token_type_ids" in cap:
#             tokens.append(cap["token_type_ids"])
#             tokens_type_id_exist = True
#         labels.append(label)
#         up_labels.append(up_label)
#         attention.append(cap["attention_mask"])
#         up_attention.append(up_cap["attention_mask"])
#         masked_ids.append(cap["masked_ids"])
#         path.append(p)
#         orig_imgs.append(orig_img)

#     # stack
#     imgs = torch.stack(imgs)
#     ext_imgs = torch.stack(ext_imgs) if len(ext_imgs) > 0 else None
#     orig_imgs = torch.stack(orig_imgs)
#     # keep the batch dim
#     ids = torch.stack(ids).squeeze(1)
#     up_ids = torch.stack(up_ids).squeeze(1)
#     if tokens_type_id_exist:
#         tokens = torch.stack(tokens).squeeze(1)
#     labels = torch.stack(labels).squeeze(1)
#     up_labels = torch.stack(up_labels).squeeze(1)
#     attention = torch.stack(attention).squeeze(1)
#     up_attention = torch.stack(up_attention).squeeze(1)
#     masked_ids = torch.stack(masked_ids).squeeze(1)

#     # sort and add to dictionary
#     sorted_cap_indices = torch.arange(len(cap_len))
#     try:
#         sorted_cap_lens = torch.tensor(cap_len)
#     except TypeError:
#         sorted_cap_lens = torch.stack(cap_len, 0)

#     path = np.array(path)
#     if len(path) != 1:
#         path = path[sorted_cap_indices]
#     return_dict = {
#         "caption_ids": ids[sorted_cap_indices],
#         "token_type_ids": tokens[sorted_cap_indices] if tokens_type_id_exist else None,
#         "attention_mask": attention[sorted_cap_indices],
#         "imgs": imgs[sorted_cap_indices],
#         "cap_lens": sorted_cap_lens,
#         "path": path,
#         "masked_ids": masked_ids[sorted_cap_indices],
#         "multi_hot_label": labels[sorted_cap_indices],
#         "up_caption_ids": up_ids[sorted_cap_indices],
#         "up_multi_hot_label": up_labels[sorted_cap_indices],
#         "up_attention_mask": up_attention[sorted_cap_indices],
#         "orig_imgs": orig_imgs[sorted_cap_indices],
#     }
#     if ext_imgs is not None:
#         return_dict["ext_imgs"] = ext_imgs
#     return return_dict


if __name__ == "__main__":
    transform = DataTransforms(is_train=True)
    dataset = MIMICPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
