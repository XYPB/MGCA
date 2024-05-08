import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from mgca.constants import *
from mgca.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3, **kwargs):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)
        
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

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

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

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

        return tokens, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        return imgs, caps, cap_len, key


def check_element_type(element, str_pool = None):
    if str_pool is None:
        # either non-empty string or non-nan float
        return (isinstance(element, str) and element != '') or (isinstance(element, float) and not np.isnan(element))
    else:
        # either string in pool or non-nan float
        return (isinstance(element, str) and element in str_pool) or (isinstance(element, float) and not np.isnan(element))

class EmbedPretrainingDataset(data.Dataset):
    def __init__(self, split='train', transform=None, data_pct=1.0,
                 imsize=256, max_words=72, simple_cap=False,
                 train_sub_set=False, structural_cap=False,
                 natural_cap=False, balanced_test=False, 
                 pred_density=False, ten_pct=False, large_density=False, 
                 instance_test_cap=False, zero_shot=False,
                 **kwargs):
        super().__init__()
        if not os.path.exists(EMBED_DATA_DIR):
            raise RuntimeError(f"{EMBED_DATA_DIR} does not exist!")
        
        self.transform = transform
        self.imsize = imsize
        self.split = split
        self.max_words = max_words
        self.structural_cap = structural_cap
        self.simple_cap = simple_cap
        self.natural_cap = natural_cap
        self.balanced_test = balanced_test
        self.pred_density = pred_density
        self.instance_test_cap = instance_test_cap
        self.zero_shot = zero_shot
        self.zero_shot_caps = None
        self.zero_shot_caps_len = None
        if split == 'train':
            self.df = pd.read_csv(EMBED_TRAIN_META_CSV)
        elif split == 'valid':
            self.df = pd.read_csv(EMBED_VALID_META_CSV)
        elif split == 'test':
            self.df = pd.read_csv(EMBED_TEST_META_CSV)
            self.cls_prompt = True
        else:
            raise ValueError(f"split {split} not supported")
        self.df_anno = pd.read_csv(EMBED_ANNO_CSV_REDUCED)
        self.df_anno_full = pd.read_csv(EMBED_ANNO_CSV)
        df_legends = pd.read_csv(EMBED_LEGENDS_CSV)
        
        self.massshape_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'massshape'].iterrows()}
        self.massdensity_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'massdens'].iterrows()}
        self.calcfind_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'calcfind'].iterrows()}
        self.calcdistri_dict = {row['Code']: row['Meaning'] for _, row in df_legends[df_legends['Header in export'] == 'calcdistri'].iterrows()}
        
        # Only use 2D mammograms for now
        self.df = self.df[self.df[EMBED_IMAGE_TYPE_COL].isin(['2D'])]
        self.df[EMBED_PATH_COL] = self.df[EMBED_PATH_COL].apply(EMBED_PATH_TRANS_FUNC)
        
        if self.structural_cap or self.natural_cap:
            self.max_words = 144
        
        sub_train_set = split == 'train' or train_sub_set
        if data_pct != 1.0 and sub_train_set:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        if self.pred_density:
            if split == 'train':
                density_file = EMBED_TRAIN_PATH2DENSITY
            elif split == 'valid':
                density_file = EMBED_VALID_PATH2DENSITY
            elif split == 'test':
                density_file = EMBED_TEST_PATH2DENSITY
            else:
                raise ValueError(f"split {split} not supported")
            assert os.path.exists(density_file)
            self.path2density = pickle.load(open(density_file, "rb"))

        if self.balanced_test:
            if self.pred_density:
                if ten_pct:
                    assert os.path.exists(EMBED_10PCT_DEN_TEST_PATH)
                    print('### Using balanced test set with 10% test examples...')
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(open(EMBED_10PCT_DEN_TEST_PATH, "rb"))
                elif large_density:
                    assert os.path.exists(EMBED_BALANCED_LARGE_DEN_TEST_PATH)
                    print('### Using balanced test set with 4x2500 examples...')
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(open(EMBED_BALANCED_LARGE_DEN_TEST_PATH, "rb"))
                else:
                    assert os.path.exists(EMBED_BALANCED_DEN_TEST_PATH)
                    print('### Using balanced test set with 4x500 examples...')
                    # Note this also contains the density label
                    self.balanced_test_path = pickle.load(open(EMBED_BALANCED_DEN_TEST_PATH, "rb"))
            else:
                if ten_pct:
                    assert os.path.exists(EMBED_10PCT_TEST_PATH)
                    print('### Using balanced test set with 10% test examples...')
                    self.balanced_test_path = pickle.load(open(EMBED_10PCT_TEST_PATH, "rb"))
                else:
                    assert os.path.exists(EMBED_BALANCED_TEST_PATH)
                    print('### Using balanced test set with 7x200 examples...')
                    self.balanced_test_path = pickle.load(open(EMBED_BALANCED_TEST_PATH, "rb"))
        else:
            self.balanced_test_path = None
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        
        self.filenames, self.path2sent, self.path2label = self.load_text_data(split)

    def load_text_data(self, split):
        base_filename = f"{split}_mgca_captions.pickle"
        if self.structural_cap:
            base_filename = base_filename.replace(".pickle", "_structural.pickle")
        elif self.simple_cap:
            base_filename = base_filename.replace(".pickle", "_simple.pickle")
        elif self.natural_cap:
            base_filename = base_filename.replace(".pickle", "_natural.pickle")
        filepath = os.path.join(EMBED_DATA_DIR, base_filename)
        
        if not os.path.isfile(filepath):
            print(f"### Caption file {filepath} does not exist. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            print(f"### Loading captions from {filepath}...")
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)
        
        # Some of the paths in the dataframe are not in the captions
        filenames = []
        path2label = {}

        print("### extract label from captions...")
        for p, sentences in tqdm(path2sent.items()):
            # Only use the test image from balanced test set during test time
            if self.split == 'test' and self.balanced_test and p not in self.balanced_test_path.keys():
                continue
            # Extract BI-RAS label from the last sentence
            if self.pred_density:
                if p not in self.path2density.keys():
                    print(f"### {p} not in density map")
                    continue
                # Ignore male images
                label = self.path2density[p] - 1
                if label == 4:
                    continue
                path2label[p] = label
                filenames.append(p)
            else:
                if self.structural_cap:
                    sent = sentences[-2]
                elif self.natural_cap or self.simple_cap:
                    sent = sentences[-1]
                else:
                    sent = sentences[-1].lower().replace('-', '')
                sent = sent.replace('bi rads', 'birads')
                assert 'birads' in sent
                if self.structural_cap or self.natural_cap:
                    label = re.findall(r"\bbirads\s\bcategory\s(\d+)", sent)[0]
                else:
                    label = re.findall(r"\bbirads\s\bscore\s(\d+)", sent)[0]
                path2label[p] = int(label)
                filenames.append(p)
        print(np.unique(list(path2label.values()), return_counts=True))
        return filenames, path2sent, path2label
    
    def _create_captions_(self, row, meta_only=False):
        target_side = row[EMBED_SIDE_COL]
        anno_row = self.df_anno[self.df_anno[EMBED_SID_COL] == row[EMBED_SID_COL]]
        anno_full_row = self.df_anno_full[self.df_anno_full[EMBED_SID_COL] == row[EMBED_SID_COL]]
        # Pick the correct side
        if target_side in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == target_side]
            anno_full_row = anno_full_row[anno_full_row[EMBED_FINDING_SIDE_COL] == target_side]
        elif 'B' in anno_row[EMBED_FINDING_SIDE_COL].tolist():
            # Pick biliteral result otherwise
            anno_row = anno_row[anno_row[EMBED_FINDING_SIDE_COL] == 'B']
            anno_full_row = anno_full_row[anno_full_row[EMBED_FINDING_SIDE_COL] == 'B']
        try:
            # pick the case with highest BI-RADS
            all_asses = anno_row[EMBED_BIRADS_COL].to_list()
            all_birads = [EMBED_LETTER_TO_BIRADS[a] for a in all_asses if check_element_type(a, EMBED_LETTER_TO_BIRADS.keys())]
            # If all screening image, prefer 0 case
            if np.max(all_birads) <= 2 and np.min(all_birads) == 0:
                idx = np.argmin(all_birads)
            else:
                idx = np.argmax(all_birads)
            anno_row = anno_row.iloc[idx]
            anno_full_row = anno_full_row.iloc[idx]
        except:
            anno_row = anno_row.iloc[0]
            anno_full_row = anno_full_row.iloc[0]
        # use the first annotation
        
        label_cnt = 0
        # if critical information is missing
        missing_info = False
        
        if self.structural_cap:
            captions = ""
            procedure = row[EMBED_PROCEDURE_COL]
            if check_element_type(procedure):
                captions += EMBED_PROCEDURE + procedure
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions += EMBED_REASON + reason
                captions += "; "
                label_cnt += 1
            else:
                missing_info = True

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            # Replace the caption with information
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += EMBED_PATIENT + patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = EMBED_SIDES_DESC[side] if check_element_type(side, EMBED_SIDES_DESC.keys()) else "unknown"
            view = view if check_element_type(view) else "unknown"
            # Replace the caption with information
            image_cap = EMBED_IMAGE_INFO_CAPTION
            image_cap = image_cap.replace("{{IMAGE_TYPE}}", image_type)
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += EMBED_IMAGE + image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info
            
            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_DENSITY + EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            calc_find = False
            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(shape_code, self.massshape_dict.keys()) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace("{{SHAPE}}", self.massshape_dict[shape_code]).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDINGS + EMBED_FINDS_CAPTION + mass_info + " "
                
                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(calc_find_code, self.calcfind_dict.keys()) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace("{{SHAPE}}", self.calcfind_dict[calc_find_code]).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                    calc_find = True
                
                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                
                captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                label_cnt += 1
                
                assert '{{' not in captions
                # dev
                # if calc_find:
                #     print(captions)
            else:
                missing_info = True
        elif self.natural_cap:
            captions = EMBED_NATURE_BASE_CAPTION

            procedure = row[EMBED_PROCEDURE_COL]
            reason = EMBED_PROCEDURE2REASON_FUNC(procedure)
            if check_element_type(reason):
                captions = captions.replace("{{REASON}}", reason)
            else:
                captions = captions.replace("{{REASON}}", '')

            age = anno_row[EMBED_AGE_COL]
            race = anno_row[EMBED_RACE_COL]
            ethnic = anno_row[EMBED_ETHNIC_COL]
            ethnic = "Non-Hispanic or Latino" if ethnic != "Hispanic or Latino" else ethnic
            # Check element type
            age = str(int(age)) if check_element_type(age) else "unknown"
            race = race if check_element_type(race) else "unknown"
            ethnic = ethnic if check_element_type(ethnic) else "unknown"
            patient_cap = EMBED_PATIENT_INFO_CAPTION
            patient_cap = patient_cap.replace("{{RACE}}", race)
            patient_cap = patient_cap.replace("{{ETHNIC}}", ethnic)
            patient_cap = patient_cap.replace("{{AGE}}", age)
            captions += patient_cap + " "
            label_cnt += 1

            image_type = row[EMBED_IMAGE_TYPE_COL]
            side = row[EMBED_SIDE_COL]
            view = row[EMBED_VIEW_COL]
            # Check element type
            image_type = image_type if check_element_type(image_type) else "unknown"
            side = EMBED_SIDES_DESC[side] if check_element_type(side, EMBED_SIDES_DESC.keys()) else "unknown"
            view = view if check_element_type(view) else "unknown"
            image_cap = EMBED_NATURE_IMAGE_CAPTION
            image_cap = image_cap.replace("{{SIDE}}", side)
            image_cap = image_cap.replace("{{VIEW}}", view)
            captions += image_cap + " "
            label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info
            
            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                    captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                mass_info = EMBED_MASS_CAPTION[asses]
                shape_code = anno_full_row[EMBED_MASS_SHAPE_COL]
                density_code = anno_full_row[EMBED_MASS_DENSITY_COL]
                if check_element_type(shape_code, self.massshape_dict.keys()) and check_element_type(density_code, self.massdensity_dict.keys()):
                    mass_info += EMBED_MASS_EXTRA_CAPTION.replace("{{SHAPE}}", self.massshape_dict[shape_code]).replace("{{DENSITY}}", self.massdensity_dict[density_code])
                captions += EMBED_FINDS_CAPTION + mass_info + " "
                
                calc_find_code = anno_full_row[EMBED_CALC_FIND_COL]
                calc_distri_code = anno_full_row[EMBED_CALC_DIST_COL]
                if check_element_type(calc_find_code, self.calcfind_dict.keys()) and check_element_type(calc_distri_code, self.calcdistri_dict.keys()):
                    calc_info = EMBED_CALC_FINDS_CAPTION.replace("{{SHAPE}}", self.calcfind_dict[calc_find_code]).replace("{{DISTRI}}", self.calcdistri_dict[calc_distri_code])
                    captions += calc_info + " "
                
                birads = EMBED_LETTER_TO_BIRADS[asses]
                impression_desc = EMBED_BIRADS_DESC[asses]
                captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                
                assert '{{' not in captions
            else:
                missing_info = True
        else:
            # Start with base caption
            captions = BREAST_BASE_CAPTION
            
            if not self.simple_cap:
                # provide extra side, view, density information
                side = row[EMBED_SIDE_COL]
                if check_element_type(side, EMBED_SIDES_DESC.keys()):
                    captions += BREAST_SIDE_CAPTION + EMBED_SIDES_DESC[side]
                    captions += " "
                    label_cnt += 1
                
                view = row[EMBED_VIEW_COL]
                if check_element_type(view):
                    captions += BREAST_VIEW_CAPTION + view
                    captions += " "
                    label_cnt += 1
            if meta_only:
                return captions, label_cnt, missing_info

            density = anno_row[EMBED_DENSITY_COL]
            if check_element_type(density, EMBED_DENSITY_DESC.keys()):
                density_desc = EMBED_DENSITY_DESC[density]
                captions += BREAST_DENSITY_CAPTION + str(int(density)) + ":" + density_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

            asses = anno_row[EMBED_BIRADS_COL]
            if check_element_type(asses, EMBED_BIRADS_DESC.keys()):
                asses_desc = EMBED_BIRADS_DESC[asses]
                birads = EMBED_LETTER_TO_BIRADS[asses]
                captions += BREAST_BIRADS_CAPTION + str(birads) + ":" + asses_desc + "."
                captions += " "
                label_cnt += 1
            else:
                missing_info = True

        return captions, label_cnt, missing_info

    def create_path_2_sent_mapping(self):
        sent_lens = []
        path2sent = {}
        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # Find annotations for this image
            # Can be more than 1 annotations
            captions, label_cnt, missing_info = self._create_captions_(row)

            # Skip the image if there is no label
            if label_cnt == 0 or missing_info:
                continue

            # use space instead of newline
            captions = captions.replace("\n", " ")

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
                if len(tokens) <= 0:
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

            if cnt >= 1:
                sent_lens.append(cnt)
                path2sent[row[EMBED_PATH_COL]] = study_sent
        
        sent_lens = np.array(sent_lens)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}] {len(sent_lens)}"
        )
        
        return path2sent
    
    def __len__(self):
        return len(self.filenames)
    
    def random_mask(self, tokens, mask_ratio=0.1):
        # Unused
        return tokens
        masked_tokens = deepcopy(tokens)
        length = max(1, masked_tokens.shape[1]-5)
        for i in range(1, length):
            if masked_tokens[0][i] == self.tokenizer.eos_token_id:
                break

            prob = random.random()
            if prob < mask_ratio:
                masked_tokens[0][i] = self.tokenizer.mask_token_id
        return tokens
    
    def get_caption(self, path, series_sents=None):
        if series_sents is None:
            series_sents = self.path2sent[path]

        print(series_sents)

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
        masked_ids = self.random_mask(tokens['input_ids'])
        tokens['masked_ids'] = masked_ids

        return tokens, x_len
    
    def get_birads_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_LETTER_TO_BIRADS))
        key = self.filenames[index]
        asses = self.path2label[key]
        multi_hot_label[asses] = 1
        return multi_hot_label
    
    def get_density_one_hot_label(self, index, get_full=False):
        multi_hot_label = torch.zeros(len(EMBED_DENSITY_DESC) - 1)
        key = self.filenames[index]
        density = self.path2label[key]
        multi_hot_label[density] = 1
        return multi_hot_label
    
    def __cls_getitem__(self, index):
        key = self.filenames[index]
        if self.pred_density:
            one_hot_label = self.get_density_one_hot_label(index)
        else:
            one_hot_label = self.get_birads_one_hot_label(index)
        if self.zero_shot_caps is None or self.instance_test_cap:
            zero_shot_caps = []
            zero_shot_caps_len = []
            # get base caption
            if self.instance_test_cap:
                target_row = self.df[self.df[EMBED_PATH_COL] == key].iloc[0]
                base_captions = self._create_captions_(target_row, meta_only=True)[0]
            else:
                base_captions = ''
            # get zero-shot captions based on classes
            if self.pred_density:
                for density, density_desc in EMBED_DENSITY_DESC.items():
                    if density == 5:
                        continue
                    # build density caption following training format
                    if self.structural_cap:
                        density_desc = EMBED_DENSITY_DESC[density]
                        captions = base_captions + EMBED_DENSITY + EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                        if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                            captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                    elif self.natural_cap:
                        density_desc = EMBED_DENSITY_DESC[density]
                        captions = base_captions + EMBED_BREAST_COMPOSITION_CAPTION.replace("{{DENSITY}}",density_desc)
                        if density in EMBED_DENSITY_EXTRA_CAPTION.keys():
                            captions += EMBED_DENSITY_EXTRA_CAPTION[density]
                    else:
                        captions = base_captions + BREAST_BASE_CAPTION + BREAST_DENSITY_CAPTION + str(density) + ": " + density_desc + "."
                    captions = captions.replace("\n", " ").lower()
                    cap, cap_len = self.get_caption(None, [captions])
                    zero_shot_caps.append(cap)
                    zero_shot_caps_len.append(cap_len)
            else:
                for asses, birads_desc in EMBED_BIRADS_DESC.items():
                    birads = EMBED_LETTER_TO_BIRADS[asses]
                    # build density caption following training format
                    if self.structural_cap:
                        # findings
                        mass_info = EMBED_MASS_CAPTION[asses]
                        captions = base_captions + EMBED_FINDINGS + EMBED_FINDS_CAPTION + mass_info + " "
                        # impression
                        impression_desc = EMBED_BIRADS_DESC[asses]
                        captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                        # overall assesment
                        captions += EMBED_ASSESSMENT + EMBED_ASSESSMENT_CAPTION[asses]
                    elif self.natural_cap:
                        # findings
                        mass_info = EMBED_MASS_CAPTION[asses]
                        captions = base_captions + EMBED_FINDS_CAPTION + mass_info + " "
                        # impression
                        impression_desc = EMBED_BIRADS_DESC[asses]
                        captions += EMBED_IMPRESSIONS + EMBED_IMPRESSION_CAPTION.replace("{{BIRADS}}", str(birads)).replace("{{BIRADS_DESC}}", impression_desc)
                    else:
                        captions = base_captions + BREAST_BASE_CAPTION + BREAST_BIRADS_CAPTION + str(birads) + ": " + birads_desc + "."
                    # Update caption type if using raw style caption
                    captions = captions.replace("\n", " ").lower()
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
        key = GET_JPEG_PATH_FUNC(key)
        imgs = get_imgs(key, self.imsize, self.transform)
        return imgs, self.zero_shot_caps, self.zero_shot_caps_len, one_hot_label

    def __getitem__(self, index):
        if self.zero_shot:
            return self.__cls_getitem__(index)
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        key = GET_JPEG_PATH_FUNC(key)
        imgs = get_imgs(key, self.imsize, self.transform)
        return imgs, caps, cap_len, key



def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    try:
        sorted_cap_lens, sorted_cap_indices = torch.sort(
            torch.tensor(cap_len), 0, True)
    except Exception as e:
        sorted_cap_indices = torch.arange(len(cap_len))
        sorted_cap_lens = torch.stack(cap_len, 0)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict


if __name__ == "__main__":
    from mgca.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
