import os
from pathlib import Path


DATA_BASE_DIR = '~/project/PEMedCLIP/data' 
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
DATA_BASE_DIR = Path(DATA_BASE_DIR)
HF_CKPT_CACHE_DIR = "~/palmer_scratch/hugging-face-cache"
HF_CKPT_CACHE_DIR = os.path.expanduser(HF_CKPT_CACHE_DIR)
# #############################################
# CheXpert constants
# #############################################
CHEXPERT_DATA_DIR = DATA_BASE_DIR / "CheXpert-v1.0"
CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR / "train.csv"
CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR / \
    "train_split.csv"  # train split from train.csv
CHEXPERT_VALID_CSV = CHEXPERT_DATA_DIR / \
    "valid_split.csv"  # valid split from train.csv
CHEXPERT_TEST_CSV = (
    CHEXPERT_DATA_DIR / "valid.csv"
)  # using validation set as test set (test set label hidden)
CHEXPERT_MASTER_CSV = (
    CHEXPERT_DATA_DIR / "master_updated.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_TRAIN_DIR = CHEXPERT_DATA_DIR / "train"
CHEXPERT_TEST_DIR = CHEXPERT_DATA_DIR / "valid"
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_5x200.csv"
CHEXPERT_8x200_QUERY = CHEXPERT_DATA_DIR / "chexpert_8x200_query.csv"
CHEXPERT_8x200_CANDIDATES = CHEXPERT_DATA_DIR / "chexpert_8x200_candidates.csv"

CHEXPERT_VALID_NUM = 5000
CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"
CHEXPERT_REPORT_COL = "Report Impression"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

# CHEXPERT_CLASS_PROMPTS = {
#     "Atelectasis": "Platelike opacity likely represents atelectasis.",
#     "Cardiomegaly": "The cardiac silhouette is enlarged.",
#     "Edema": "The presence of hazy opacity suggests interstitial pulmonary edema.",
#     "Fracture": "A cortical step off indicates the presence of a fracture.",
#     "Pleural Effusion": "The pleural space is partially filled with fluid",
#     "Pneumonia": "A pulmonary opacity with ill defined borders likely represents pneumonia.",
#     "Pneumothorax": "A medial pneumothorax is present adjacent to the heart.",
#     "No Finding": "No clinically significant radiographic abnormalities."
# }

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


# #############################################
# MIMIC-CXR-JPG constants
# #############################################
MIMIC_CXR_DATA_DIR = DATA_BASE_DIR / "raw/physionet.org/files/mimic-cxr-jpg/2.0.0"
# MIMIC_CXR_TRAIN_TXT = MIMIC_CXR_DATA_DIR / "train.txt"
# MIMIC_CXR_VALID_TXT = MIMIC_CXR_DATA_DIR / "test.txt"
MIMIC_CXR_CHEXPERT_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-chexpert.csv"
MIMIC_CXR_META_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_TEXT_CSV = MIMIC_CXR_DATA_DIR / "mimic_cxr_sectioned.csv"
MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_DATA_DIR / "mimic-cxr-2.0.0-split.csv"
# Created csv
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR / "train.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR / "test.csv"
MIMIC_CXR_TEST_CSV = MIMIC_CXR_DATA_DIR / "test.csv"
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR / "master.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"

# #############################################
# RSNA constants
# #############################################
RSNA_DATA_DIR = DATA_BASE_DIR / "RSNA_Pneumonia"
RSNA_ORIGINAL_TRAIN_CSV = RSNA_DATA_DIR / "stage_2_train_labels.csv"
RSNA_CLASSINFO_CSV = RSNA_DATA_DIR / "stage_2_detailed_class_info.csv"
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_VALID_CSV = RSNA_DATA_DIR / "val.csv"
RSNA_TEST_CSV = RSNA_DATA_DIR / "test.csv"
RSNA_DETECTION_TRAIN_PKL = RSNA_DATA_DIR / "train.pkl"
RSNA_DETECTION_VALID_PKL = RSNA_DATA_DIR / "val.pkl"
RSNA_DETECTION_TEST_PKL = RSNA_DATA_DIR / "test.pkl"

RSNA_IMG_DIR = RSNA_DATA_DIR / "stage_2_train_images"
RSNA_TRAIN_PCT = 0.7


# #############################################
# SIIM constants
# #############################################
PNEUMOTHORAX_DATA_DIR = DATA_BASE_DIR / "SIIM_Pneumothorax"
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7


# #############################################
# tuberculosis constants
# #############################################
COVIDX_DATA_DIR = DATA_BASE_DIR / "COVIDx"
# COVIDX_ORIGINAL_TRAIN_TXT = COVIDX_DATA_DIR / "train.txt"
COVIDX_ORIGINAL_TRAIN_TXT = COVIDX_DATA_DIR / "train_COVIDx9A.txt"
# COVIDX_ORIGINAL_TEST_TXT = COVIDX_DATA_DIR / "test.txt"
COVIDX_ORIGINAL_TEST_TXT = COVIDX_DATA_DIR / "test_COVIDx9A.txt"
COVIDX_TRAIN_CSV = COVIDX_DATA_DIR / "train.csv"
COVIDX_VALID_CSV = COVIDX_DATA_DIR / "valid.csv"
COVIDX_TEST_CSV = COVIDX_DATA_DIR / "test.csv"

# #############################################
# COVIDx constants
# #############################################
TUBERCULOSIS_DATA_DIR = DATA_BASE_DIR / "tuberculosis"
TUBERCULOSIS_ORIGINAL_TRAIN_CSV = TUBERCULOSIS_DATA_DIR / "shenzhen_metadata.csv"
TUBERCULOSIS_TRAIN_CSV = TUBERCULOSIS_DATA_DIR / "train.csv"
TUBERCULOSIS_VALID_CSV = TUBERCULOSIS_DATA_DIR / "valid.csv"
TUBERCULOSIS_TEST_CSV = TUBERCULOSIS_DATA_DIR / "test.csv"

# #############################################
# Vinbigdata constants
# #############################################
VIN_DATA_DIR = DATA_BASE_DIR / "vinbigdata"
VIN_ORIGINAL_TRAIN_TXT = VIN_DATA_DIR / "train.csv"
VIN_TRAIN_CSV = VIN_DATA_DIR / "train_df.csv"
VIN_VALID_CSV = VIN_DATA_DIR / "valid_df.csv"
VIN_TEST_CSV = VIN_DATA_DIR / "test_df.csv"


# #############################################
# Object CXR constants
# #############################################
OBJ_DATA_DIR = DATA_BASE_DIR / "object-CXR"
OBJ_ORIGINAL_TRAIN_CSV = OBJ_DATA_DIR / "train.csv"
OBJ_ORIGINAL_DEV_CSV = OBJ_DATA_DIR / "dev.csv"
OBJ_TRAIN_PKL = OBJ_DATA_DIR / "train.pkl"
OBJ_VALID_PKL = OBJ_DATA_DIR / "valid.pkl"
OBJ_TEST_PKL = OBJ_DATA_DIR / "test.pkl"
OBJ_TRAIN_IMG_PATH = OBJ_DATA_DIR / "train"
OBJ_VALID_IMG_PATH = OBJ_DATA_DIR / "train"
OBJ_TEST_IMG_PATH = OBJ_DATA_DIR / "dev"


# #############################################
# EMBED constants
# #############################################
DATA_BASE_DIR = '~/project/PEMedCLIP/data' 
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
EMBED_DATA_DIR = DATA_BASE_DIR + "/Embed"
EMBED_DATA_PATH = EMBED_DATA_DIR + "/images"
EMBED_TRAIN_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_train.csv"
EMBED_TEST_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_test.csv"
EMBED_VALID_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_valid.csv"
# Read the full annotation for calcification information
EMBED_ANNO_CSV_REDUCED = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical_reduced.csv"
EMBED_ANNO_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical.csv"
EMBED_LEGENDS_CSV = EMBED_DATA_DIR + "/tables/AWS_Open_Data_Clinical_Legend.csv"
EMBED_INTER_VIEW_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_view.pkl"
EMBED_INTER_SIDE_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_side.pkl"
EMBED_BALANCED_TEST_PATH = EMBED_DATA_DIR + "/test_7x200_path2label.pickle"
EMBED_10PCT_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2label.pickle"
EMBED_BALANCED_TRAIN_PATH = EMBED_DATA_DIR + "/train_7x550_path2label.pickle"
EMBED_BALANCED_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x500_path2density.pickle"
EMBED_10PCT_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2density.pickle"
EMBED_BALANCED_LARGE_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x2500_path2density.pickle"
EMBED_BALANCED_DEN_TRAIN_PATH = EMBED_DATA_DIR + "/train_4x1000_path2density.pickle"
EMBED_TRAIN_PATH2DENSITY = EMBED_DATA_DIR + "/train_path2density.pickle"
EMBED_VALID_PATH2DENSITY = EMBED_DATA_DIR + "/valid_path2density.pickle"
EMBED_TEST_PATH2DENSITY = EMBED_DATA_DIR + "/test_path2density.pickle"
EMBED_TRAIN_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_train_resized.pickle"
EMBED_VALID_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_valid_resized.pickle"
EMBED_TEST_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_test_resized.pickle"

EMBED_IMAGE_TYPE_COL = "FinalImageType"
EMBED_PATH_COL = "anon_dicom_path"
EMBED_PID_COL = 'empi_anon'
EMBED_SID_COL = 'acc_anon'
EMBED_SIDE_COL = 'ImageLateralityFinal'
EMBED_FINDING_SIDE_COL = 'side'
EMBED_VIEW_COL = 'ViewPosition'
EMBED_DENSITY_COL = 'tissueden'
EMBED_BIRADS_COL = 'asses'
EMBED_PROCEDURE_COL = 'StudyDescription'
EMBED_MASS_SHAPE_COL = 'massshape'
EMBED_MASS_DENSITY_COL = 'massdens'
EMBED_CALC_FIND_COL = 'calcfind'
EMBED_CALC_DIST_COL = 'calcdistri'
EMBED_AGE_COL = 'age_at_study'
EMBED_ROI_COORD = 'ROI_coords'
EMBED_RACE_COL = 'RACE_DESC'
EMBED_ETHNIC_COL = 'ETHNIC_GROUP_DESC'
EMBED_PATH_TRANS_FUNC = lambda x: x.replace("/mnt/NAS2/mammo/anon_dicom", EMBED_DATA_PATH)
EMBED_PROCEDURE2REASON_FUNC = lambda x: "screening" if "screen" in x.lower() else "diagnostic" if "diag" in x.lower() else ""
# Normal caption constants
BREAST_BASE_CAPTION = "This is a breast 2D full-field digital mammogram of a patient "
BREAST_SIDE_CAPTION = "on side " # Make the caption more grammarly correct
BREAST_VIEW_CAPTION = "with view "
BREAST_DENSITY_CAPTION = "with breast tissue density "
BREAST_BIRADS_CAPTION = "with BIRADS score "
# TODO: Add more findings according to the EMBED dataset structure
# Natural Captions
EMBED_NATURE_BASE_CAPTION = "This is a breast 2D full-field digital {{REASON}} mammogram of a patient. "
EMBED_NATURE_IMAGE_CAPTION = "This mammogram is for {{SIDE}} breast with {{VIEW}} view. "
# Structural Captions
EMBED_PROCEDURE = 'Procedure reported: ' # EMBED_PROCEDURE_COL
EMBED_REASON = 'Reason for procedure: ' # Screening / Diagnostic, maybe add more details later
EMBED_PATIENT = 'Patient info: ' # AGE + RACE + ETHNIC
EMBED_IMAGE = 'Image info: ' # EMBED_IMAGE_TYPE_COL + EMBED_SIDE_COL + EMBED_VIEW_COL
EMBED_DENSITY = 'Breast composition: ' # EMBED_DENSITY_COL + extra description
EMBED_FINDINGS = 'Findings: ' # EMBED_MASS info + EMBED_CALC_FIND_COL + extra description
EMBED_IMPRESSIONS = 'Impressions: ' # EMBED_BIRADS_COL + extra description
EMBED_ASSESSMENT = 'Overall Assessment: ' # EMBED_BIRADS_COL number

EMBED_PATIENT_INFO_CAPTION = "This patient is {{RACE}}, {{ETHNIC}}, and {{AGE}} years old. "
EMBED_IMAGE_INFO_CAPTION = "This is a {{IMAGE_TYPE}} full-field digital mammogram of the {{SIDE}} breast with {{VIEW}} view. "
EMBED_BREAST_COMPOSITION_CAPTION = "The breast is {{DENSITY}}. "
EMBED_DENSITY_EXTRA_CAPTION = {
    3: "This may lower the sensitivity of mammography. ",
    4: "This may lower the sensitivity of mammography. ",
}
EMBED_FINDS_CAPTION = "The mammogram shows that "
EMBED_MASS_CAPTION = {
    'A': "an additional imaging is recommended. ",
    'N': "no significant masses, calcification, or other abnormalities are present. ",
    'B': "a benign finding is present. ",
    'P': "a probably benign finding is present. ",
    'S': "a suspicious abnormality is present. ",
    'M': "a highly suggestive of malignancy is present, a biopsy is recommended. ",
    'K': "a known biopsy-proven malignant mass is present. ",
}
EMBED_MASS_EXTRA_CAPTION = 'The mass is {{SHAPE}} and {{DENSITY}}. '
EMBED_CALC_FINDS_CAPTION = 'A {{DISTRI}} {{SHAPE}} calcification is present. '
EMBED_IMPRESSION_CAPTION = "BI-RADS Category {{BIRADS}}: {{BIRADS_DESC}}. "
EMBED_ASSESSMENT_CAPTION = {
    'A': "Additional imaging is recommended. ",
    'N': "Negative. ",
    'B': "Benign. ",
    'P': "Probably benign. ",
    'S': "Suspicious abnormality. ",
    'M': "Highly suggestive of malignancy. ",
    'K': "Known biopsy-proven malignancy. ",
}
EMBED_SIDES_DESC = {
    'L': 'left',
    'R': 'right',
    'B': 'bilateral',
}
EMBED_DENSITY_DESC = {
    1: "almost entirely fat",
    2: "scattered fibroglandular densities",
    3: "heterogeneously dense",
    4: "extremely dense",
    5: "normal male dense",
}
EMBED_LETTER_TO_BIRADS = {
    "A": 0,
    "N": 1,
    "B": 2,
    "P": 3,
    "S": 4,
    "M": 5,
    "K": 6,
}
EMBED_BIRADS_DESC = {
    'A': "additional imaging required",
    'N': "negative",
    'B': "benign finding",
    'P': "probably benign finding",
    'S': "suspicious abnormality",
    'M': "highly suggestive of malignancy",
    'K': "known biopsy-proven malignancy",
}
GET_JPEG_PATH_FUNC = lambda x: x.replace('Embed', 'EMBED_1080_JPG').replace(".dcm", "_resized.jpg")