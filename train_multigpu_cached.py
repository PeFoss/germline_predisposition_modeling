import os
import gc
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from model_classes import GermLinePredModel


# ============================================================
# PATHS / CONSTANTS
# ============================================================

MEDS_DATABASE_PATH = (
    "/home/jupyter/workspace/data_bucket/"
    "MEDS_DATA/MEDS_cohort_reader"
)

ENTRIES_PATH = (
    "/home/jupyter/workspace/data_bucket/"
    "v9_gen_data/entries_table_filt_v9.csv"
)

SURVEY_PATH = (
    "/home/jupyter/workspace/data_bucket/"
    "survey_data/survey_filt.parquet"
)

TOKEN_CACHE_DIR = (
    "/home/jupyter/workspace/data_bucket/"
    "tokenized_germline_cache"
)

OUTPUT_DIR = "/home/jupyter/saved_model"

EHR_MODEL_NAME = (
    "StanfordShahLab/mamba-tiny-4096-clmbr"
)

SURVEY_MODEL_NAME = (
    "jinaai/jina-embeddings-v2-small-en"
)

NUM_GENES = 73

XAI_LAYERS = [7, 15, 23]


# ============================================================
# TOKENIZED CACHE
# ============================================================

class TokenizedPatientCache:

    def __init__(
        self,
        cache_dir,
    ):

        self.cache_dir = cache_dir

        with open(
            os.path.join(
                cache_dir,
                "metadata.json",
            ),
            "r",
        ) as f:

            self.metadata = json.load(
                f
            )

        self.subject_ids = np.load(
            os.path.join(
                cache_dir,
                "subject_ids.npy",
            ),
            mmap_mode="r",
        )

        self.ehr_offsets = np.load(
            os.path.join(
                cache_dir,
                "ehr_offsets.npy",
            ),
            mmap_mode="r",
        )

        self.survey_offsets = np.load(
            os.path.join(
                cache_dir,
                "survey_offsets.npy",
            ),
            mmap_mode="r",
        )

        self.ehr_input_ids = np.memmap(
            os.path.join(
                cache_dir,
                "ehr_input_ids.int32.bin",
            ),
            dtype=np.int32,
            mode="r",
        )

        self.survey_input_ids = np.memmap(
            os.path.join(
                cache_dir,
                "survey_input_ids.int32.bin",
            ),
            dtype=np.int32,
            mode="r",
        )

        self.ehr_pad_token_id = int(
            self.metadata[
                "ehr_pad_token_id"
            ]
        )

        self.survey_pad_token_id = int(
            self.metadata[
                "survey_pad_token_id"
            ]
        )

    def __len__(
        self,
    ):

        return len(
            self.subject_ids
        )

    def get_ehr_ids(
        self,
        row_index,
    ):

        start = int(
            self.ehr_offsets[
                row_index
            ]
        )

        end = int(
            self.ehr_offsets[
                row_index + 1
            ]
        )

        return self.ehr_input_ids[
            start:end
        ]

    def get_survey_ids(
        self,
        row_index,
    ):

        start = int(
            self.survey_offsets[
                row_index
            ]
        )

        end = int(
            self.survey_offsets[
                row_index + 1
            ]
        )

        return self.survey_input_ids[
            start:end
        ]


# ============================================================
# DATASET
# ============================================================

class PatientEHRSurveyDataset(
    Dataset
):

    def __init__(
        self,
        token_cache,
        subject_ids,
        labels_by_subject,
    ):

        self.token_cache = (
            token_cache
        )

        self.subject_ids = np.asarray(
            [
                int(x)
                for x in subject_ids
            ],
            dtype=np.int64,
        )

        self.labels_by_subject = (
            labels_by_subject
        )

        self.cache_row_indices = (
            np.searchsorted(
                self.token_cache.subject_ids,
                self.subject_ids,
            )
        )

        if np.any(
            self.cache_row_indices
            >= len(
                self.token_cache.subject_ids
            )
        ):

            raise ValueError(
                "Some subject IDs are missing "
                "from the tokenized cache."
            )

        cached_subject_ids = (
            self.token_cache.subject_ids[
                self.cache_row_indices
            ]
        )

        if not np.array_equal(
            cached_subject_ids,
            self.subject_ids,
        ):

            raise ValueError(
                "Some subject IDs are missing "
                "from the tokenized cache."
            )

    def __len__(
        self,
    ):

        return len(
            self.subject_ids
        )

    def __getitem__(
        self,
        index,
    ):

        subject_id = int(
            self.subject_ids[
                index
            ]
        )

        cache_row_index = int(
            self.cache_row_indices[
                index
            ]
        )

        ehr_input_ids = (
            self.token_cache.get_ehr_ids(
                cache_row_index
            )
        )

        survey_input_ids = (
            self.token_cache.get_survey_ids(
                cache_row_index
            )
        )

        label = (
            self.labels_by_subject[
                subject_id
            ]
        )

        return {
            "ehr_input_ids":
                ehr_input_ids,

            "survey_input_ids":
                survey_input_ids,

            "label":
                label,
        }


# ============================================================
# COLLATOR
# ============================================================

@dataclass
class PatientBatchCollator:

    ehr_pad_token_id: int = 0
    survey_pad_token_id: int = 0

    def __call__(
        self,
        features,
    ):

        # ====================================================
        # EHR
        # ====================================================

        ehr_sequences = [
            torch.as_tensor(
                feature[
                    "ehr_input_ids"
                ],
                dtype=torch.long,
            )
            for feature
            in features
        ]

        ehr_input_ids = (
            pad_sequence(
                ehr_sequences,
                batch_first=True,
                padding_value=
                    self.ehr_pad_token_id,
            )
        )

        ehr_lengths = torch.tensor(
            [
                len(sequence)
                for sequence
                in ehr_sequences
            ],
            dtype=torch.long,
        )

        max_ehr_length = (
            ehr_input_ids.shape[1]
        )

        ehr_attention_mask = (
            (
                torch.arange(
                    max_ehr_length
                ).unsqueeze(0)
                <
                ehr_lengths.unsqueeze(
                    1
                )
            )
            .long()
        )

        # ====================================================
        # SURVEY
        # ====================================================

        survey_sequences = [
            torch.as_tensor(
                feature[
                    "survey_input_ids"
                ],
                dtype=torch.long,
            )
            for feature
            in features
        ]

        survey_input_ids = (
            pad_sequence(
                survey_sequences,
                batch_first=True,
                padding_value=
                    self.survey_pad_token_id,
            )
        )

        survey_lengths = torch.tensor(
            [
                len(sequence)
                for sequence
                in survey_sequences
            ],
            dtype=torch.long,
        )

        max_survey_length = (
            survey_input_ids.shape[1]
        )

        survey_attention_mask = (
            (
                torch.arange(
                    max_survey_length
                ).unsqueeze(0)
                <
                survey_lengths.unsqueeze(
                    1
                )
            )
            .long()
        )

        # ====================================================
        # LABELS
        # ====================================================

        labels = torch.stack(
            [
                torch.as_tensor(
                    feature[
                        "label"
                    ],
                    dtype=torch.float32,
                )
                for feature
                in features
            ]
        )

        return {
            "ehr_input_ids":
                ehr_input_ids,

            "ehr_attention_mask":
                ehr_attention_mask,

            "survey_input_ids":
                survey_input_ids,

            "survey_attention_mask":
                survey_attention_mask,

            "labels":
                labels,
        }


# ============================================================
# LABEL PARSING
# ============================================================

def to_float32_array(x):

    if isinstance(
        x,
        np.ndarray,
    ):

        return x.astype(
            np.float32
        )

    if isinstance(
        x,
        (list, tuple),
    ):

        return np.asarray(
            x,
            dtype=np.float32,
        )

    if isinstance(
        x,
        str,
    ):

        values = re.findall(
            (
                r"np\.float64\("
                r"([-+]?\d*\.?\d+"
                r"(?:[eE][-+]?\d+)?)"
                r"\)"
            ),
            x,
        )

        if values:

            return np.asarray(
                values,
                dtype=np.float32,
            )

        cleaned = (
            x.strip("[]")
        )

        return np.fromstring(
            cleaned.replace(
                ",",
                " ",
            ),
            sep=" ",
            dtype=np.float32,
        )

    return np.asarray(
        x,
        dtype=np.float32,
    )


def make_label(x):

    if x is None:

        return np.zeros(
            NUM_GENES,
            dtype=np.float32,
        )

    if (
        isinstance(
            x,
            (float, np.floating),
        )
        and np.isnan(x)
    ):

        return np.zeros(
            NUM_GENES,
            dtype=np.float32,
        )

    label = (
        to_float32_array(x)
    )

    if label.size == 0:

        return np.zeros(
            NUM_GENES,
            dtype=np.float32,
        )

    label = label.reshape(-1)

    if (
        label.shape[0]
        != NUM_GENES
    ):

        raise ValueError(
            f"Expected {NUM_GENES} "
            "gene labels, got "
            f"{label.shape[0]}."
        )

    return label.astype(
        np.float32
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    eval_pred,
):

    predictions = (
        eval_pred.predictions
    )

    labels = np.asarray(
        eval_pred.label_ids
    )

    # Robust to model outputs containing
    # additional tensors.
    if isinstance(
        predictions,
        (tuple, list),
    ):

        logits = (
            predictions[0]
        )

    else:

        logits = predictions

    logits = np.asarray(
        logits
    )

    probs = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -logits
            )
        )
    )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_labels = (
        labels[:, 0]
    )

    overall_probs = (
        probs[:, 0]
    )

    if (
        np.unique(
            overall_labels
        ).size == 2
    ):

        overall_roc_auc = (
            roc_auc_score(
                overall_labels,
                overall_probs,
            )
        )

    else:

        overall_roc_auc = (
            np.nan
        )

    if np.any(
        overall_labels == 1
    ):

        overall_pr_auc = (
            average_precision_score(
                overall_labels,
                overall_probs,
            )
        )

    else:

        overall_pr_auc = (
            np.nan
        )

    # ========================================================
    # GENES
    # ========================================================

    gene_labels = (
        labels[:, 1:]
    )

    gene_probs = (
        probs[:, 1:]
    )

    gene_roc_aucs = []

    gene_pr_aucs = []

    for i in range(
        gene_labels.shape[1]
    ):

        y_true = (
            gene_labels[:, i]
        )

        y_prob = (
            gene_probs[:, i]
        )

        if (
            np.unique(
                y_true
            ).size == 2
        ):

            gene_roc_aucs.append(
                roc_auc_score(
                    y_true,
                    y_prob,
                )
            )

        if np.any(
            y_true == 1
        ):

            gene_pr_aucs.append(
                average_precision_score(
                    y_true,
                    y_prob,
                )
            )

    gene_macro_roc_auc = (
        float(
            np.mean(
                gene_roc_aucs
            )
        )
        if gene_roc_aucs
        else np.nan
    )

    gene_macro_pr_auc = (
        float(
            np.mean(
                gene_pr_aucs
            )
        )
        if gene_pr_aucs
        else np.nan
    )

    return {
        "overall_roc_auc":
            overall_roc_auc,

        "overall_pr_auc":
            overall_pr_auc,

        "gene_macro_roc_auc":
            gene_macro_roc_auc,

        "gene_macro_pr_auc":
            gene_macro_pr_auc,
    }


# ============================================================
# MODEL / TOKENIZER LOADING + LORA
# ============================================================

def load_models_and_tokenizers():

    # ========================================================
    # EHR MODEL
    # ========================================================

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            EHR_MODEL_NAME,
            trust_remote_code=True,
        )
    )

    # ========================================================
    # SURVEY MODEL
    # ========================================================


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model_surv = AutoModel.from_pretrained(
        SURVEY_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        # quantization_config=bnb_config,
    )
    
    # model_surv.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={
    #         "use_reentrant": False
    #     }
    # )

    # ========================================================
    # EHR LORA
    # ========================================================

    ehr_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "in_proj"
        ],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(
        model,
        ehr_config,
    )

    # ========================================================
    # SURVEY LORA
    # ========================================================

    survey_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "dense",
            "query",
            "key",
            'value',
            'Wo',
            'gated_layers'
        ],
        lora_dropout=0.05,
        bias="none",
    )

    model_surv = (
        get_peft_model(
            model_surv,
            survey_config,
        )
    )

    return (
        model_surv,
        model,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    # ========================================================
    # TOKENIZED CACHE
    # ========================================================

    token_cache = (
        TokenizedPatientCache(
            TOKEN_CACHE_DIR
        )
    )

    total_subjects = (
        token_cache
        .subject_ids
        .astype(
            np.int64
        )
        .tolist()
    )

    if rank == 0:

        print(
            "Loaded tokenized cache: "
            f"{len(token_cache):,} subjects",
            flush=True,
        )

        print(
            "Cached max lengths: "
            f"EHR={token_cache.metadata['max_ehr_tokens']} | "
            f"Survey={token_cache.metadata['max_survey_tokens']}",
            flush=True,
        )

    # ========================================================
    # GENETIC LABEL DATA
    # ========================================================

    entries_v9 = (
        pd.read_csv(
            ENTRIES_PATH,
            usecols=[
                "s",
                "label",
            ],
        )
    )

    if rank == 0:

        print(
            "Loaded entries: "
            f"{len(entries_v9):,}",
            flush=True,
        )

    # ========================================================
    # SPLIT
    # ========================================================

    (
        train_ids,
        temp_ids,
    ) = train_test_split(
        total_subjects,
        test_size=0.3,
        random_state=97,
    )

    (
        test_ids,
        val_ids,
    ) = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=87,
    )

    if rank == 0:

        print(
            f"Train: {len(train_ids):,} | "
            f"Val: {len(val_ids):,} | "
            f"Test: {len(test_ids):,}",
            flush=True,
        )

    # ========================================================
    # LABELS
    #
    # Build labels only for actual EHR+survey subjects.
    # ========================================================

    labels_df = (
        pd.DataFrame(
            {
                "s":
                    total_subjects
            }
        )
        .merge(
            entries_v9,
            how="left",
            on="s",
        )
    )

    labels_df[
        "label"
    ] = (
        labels_df[
            "label"
        ].apply(
            make_label
        )
    )

    labels = (
        labels_df
        .set_index("s")[
            "label"
        ]
        .to_dict()
    )

    for (
        key,
        label,
    ) in labels.items():

        overall = int(
            np.sum(label)
            > 0
        )

        labels[
            key
        ] = (
            np.insert(
                label,
                0,
                overall,
            )
            .astype(
                np.float32
            )
        )

    if rank == 0:

        print(
            "Built labels: "
            f"{len(labels):,}",
            flush=True,
        )

    # ========================================================
    # FREE STARTUP OBJECTS
    # ========================================================

    del entries_v9
    del labels_df

    gc.collect()

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        PatientEHRSurveyDataset(
            token_cache,
            train_ids,
            labels,
        )
    )

    val_dataset = (
        PatientEHRSurveyDataset(
            token_cache,
            val_ids,
            labels,
        )
    )

    test_dataset = (
        PatientEHRSurveyDataset(
            token_cache,
            test_ids,
            labels,
        )
    )

    # ========================================================
    # MODELS
    # ========================================================

    if rank == 0:

        print(
            "Loading models/tokenizers...",
            flush=True,
        )

    (
        model_surv,
        model,
    ) = (
        load_models_and_tokenizers()
    )

    if rank == 0:

        print(
            "Loaded models/tokenizers",
            flush=True,
        )

    # ========================================================
    # FULL MODEL
    # ========================================================

    num_selected_layers = len(
        XAI_LAYERS
    )

    pos_weight = np.full(
        NUM_GENES,
        3000,
        dtype=np.float32,
    )

    pos_weight = np.insert(
        pos_weight,
        0,
        50,
    )

    model_pipeline = (
        GermLinePredModel(
            model_surv,
            model,
            XAI_LAYERS,
            num_selected_layers,
            NUM_GENES,
            gene_embedding_dim=32,
            attn_heads=8,
            dropout=0.1,
            pos_weight=list(
                pos_weight
            ),
        )
    )

    # ========================================================
    # COLLATOR
    # ========================================================

    collator = (
        PatientBatchCollator(
            ehr_pad_token_id=
                token_cache.ehr_pad_token_id,

            survey_pad_token_id=
                token_cache.survey_pad_token_id,
        )
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_config = (
        TrainingArguments(

            output_dir=
                OUTPUT_DIR,

            num_train_epochs=
                10,

            lr_scheduler_type=
                "constant_with_warmup",

            warmup_steps=
                20,

            optim=
                "adamw_torch_fused",

            fp16=True,
            bf16=False,

            # ================================================
            # PER GPU UNDER DDP
            # ================================================

            per_device_train_batch_size=
                4,
            
            per_device_eval_batch_size=
                4,

            # ================================================
            # REQUIRED BECAUSE TOKENIZATION HAPPENS
            # IN THE CUSTOM COLLATOR.
            # ================================================

            remove_unused_columns=
                False,

            # ================================================
            # KEEP 0 FOR NOW TO AVOID EXTRA PROCESSES
            # DUPLICATING DATASET STATE.
            # ================================================        
            dataloader_num_workers=
                4,
            
            dataloader_prefetch_factor=
                16,
            
            dataloader_persistent_workers=
                True,
            
            dataloader_pin_memory=
                True,

            # ================================================
            # VERBOSE PROGRESS
            # ================================================

            disable_tqdm=
                False,

            logging_strategy=
                "steps",

            logging_steps=
                10,

            logging_first_step=
                True,

            # ================================================
            # EVAL / CHECKPOINTING
            #
            # These match because
            # load_best_model_at_end=True.
            # ================================================

            eval_strategy=
                "epoch",

            save_strategy=
                "epoch",

            load_best_model_at_end=
                True,

            metric_for_best_model=
                "overall_pr_auc",

            greater_is_better=
                True,

            # ================================================
            # SPEEDUP ARGS
            # ================================================
        
            # Your DDP warning showed all params are used
            ddp_find_unused_parameters=True,
        
            # Optional: periodically release cached blocks
            torch_empty_cache_steps=50,
        

            # ================================================
            # MLFLOW ONLY
            # ================================================

            report_to=
                "mlflow",

            run_name=
                "germline_multigpu_v1",

            # ================================================
            # OTHER
            # ================================================

            seed=
                42,

            torch_compile=
                False,
        )
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model_pipeline,
        args=training_config,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    if rank == 0:
        print(
            "Starting Trainer.train()",
            flush=True,
        )
    
    trainer.train(
        ignore_keys_for_eval=[
            "overall_loss",
            "gene_loss",
        ]
    )
    
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
        ignore_keys=[
            "overall_loss",
            "gene_loss",
        ],
    )
    
    if rank == 0:
        print(
            test_metrics,
            flush=True,
        )

if __name__ == "__main__":
    main()
