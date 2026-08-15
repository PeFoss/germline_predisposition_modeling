import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import meds_reader

from hf_ehr.config import Event
from hf_ehr.data.tokenization import CLMBRTokenizer
from transformers import AutoTokenizer


# ============================================================
# PATHS / CONSTANTS
# ============================================================

MEDS_DATABASE_PATH = (
    "/home/jupyter/workspace/data_bucket/"
    "MEDS_DATA/MEDS_cohort_reader"
)

SURVEY_PATH = (
    "/home/jupyter/workspace/data_bucket/"
    "survey_data/survey_filt.parquet"
)

# Write locally while tokenizing.
CACHE_DIR = Path(
    "/home/jupyter/tokenized_germline_cache"
)

# Copy completed cache here afterward.
FINAL_CACHE_DIR = Path(
    "/home/jupyter/workspace/data_bucket/"
    "tokenized_germline_cache"
)

EHR_MODEL_NAME = (
    "StanfordShahLab/mamba-tiny-4096-clmbr"
)

SURVEY_MODEL_NAME = (
    "jinaai/jina-embeddings-v2-small-en"
)

MAX_EHR_TOKENS = 4096
MAX_SURVEY_TOKENS = 1500

SURVEY_BATCH_SIZE = 25600


# ============================================================
# MEDS -> HF EVENTS
# ============================================================

def meds_subject_to_hf_events(subject):

    patient_events = []

    for meds_event in subject.events:

        properties = dict(meds_event)

        numeric_value = properties.get(
            "numeric_value"
        )

        text_value = properties.get(
            "text_value"
        )

        value = (
            numeric_value
            if numeric_value is not None
            else text_value
        )

        patient_events.append(
            Event(
                code=str(meds_event.code),
                value=value,
                unit=properties.get("unit"),
                start=meds_event.time,
                end=properties.get("end"),
                omop_table=properties.get(
                    "omop_table"
                ),
            )
        )

    return patient_events


# ============================================================
# CACHE BUILD
# ============================================================

def main():

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Writing temporary cache locally to: "
        f"{CACHE_DIR}",
        flush=True,
    )

    # ========================================================
    # SURVEY
    # ========================================================

    print(
        "Loading survey data...",
        flush=True,
    )

    survey_df = pd.read_parquet(
        SURVEY_PATH,
        columns=[
            "person_id",
            "llm_event_text",
        ],
    )

    survey_data = (
        survey_df
        .set_index("person_id")[
            "llm_event_text"
        ]
        .fillna("")
        .to_dict()
    )

    del survey_df

    print(
        f"Loaded survey data: "
        f"{len(survey_data):,}",
        flush=True,
    )

    # ========================================================
    # MEDS
    # ========================================================

    print(
        "Opening MEDS database...",
        flush=True,
    )

    database = meds_reader.SubjectDatabase(
        MEDS_DATABASE_PATH
    )

    meds_subject_ids = list(
        database
    )

    total_subjects = sorted(
        {
            int(x)
            for x in survey_data.keys()
        }
        &
        {
            int(x)
            for x in meds_subject_ids
        }
    )

    del meds_subject_ids

    print(
        f"Subjects with EHR + survey: "
        f"{len(total_subjects):,}",
        flush=True,
    )

    # ========================================================
    # TOKENIZERS
    # ========================================================

    print(
        "Loading tokenizers...",
        flush=True,
    )

    ehr_tokenizer = (
        CLMBRTokenizer.from_pretrained(
            EHR_MODEL_NAME
        )
    )

    survey_tokenizer = (
        AutoTokenizer.from_pretrained(
            SURVEY_MODEL_NAME,
            trust_remote_code=True,
        )
    )

    # ========================================================
    # CACHE PATHS
    # ========================================================

    subject_ids_path = (
        CACHE_DIR / "subject_ids.npy"
    )

    ehr_offsets_path = (
        CACHE_DIR / "ehr_offsets.npy"
    )

    survey_offsets_path = (
        CACHE_DIR / "survey_offsets.npy"
    )

    ehr_tokens_path = (
        CACHE_DIR /
        "ehr_input_ids.int32.bin"
    )

    survey_tokens_path = (
        CACHE_DIR /
        "survey_input_ids.int32.bin"
    )

    # ========================================================
    # OFFSETS / IDS
    # ========================================================

    subject_ids = np.asarray(
        total_subjects,
        dtype=np.int64,
    )

    ehr_offsets = np.zeros(
        len(total_subjects) + 1,
        dtype=np.int64,
    )

    survey_offsets = np.zeros(
        len(total_subjects) + 1,
        dtype=np.int64,
    )

    np.save(
        subject_ids_path,
        subject_ids,
    )

    ehr_token_count = 0
    survey_token_count = 0

    overall_start = (
        time.perf_counter()
    )

    print(
        "Tokenizing and writing cache...",
        flush=True,
    )

    # ========================================================
    # CACHE LOOP
    # ========================================================

    with (
        open(
            ehr_tokens_path,
            "wb",
            buffering=1024 * 1024 * 16,
        ) as ehr_file,
        open(
            survey_tokens_path,
            "wb",
            buffering=1024 * 1024 * 16,
        ) as survey_file,
    ):

        for batch_num, batch_start in enumerate(
            tqdm(
                range(
                    0,
                    len(total_subjects),
                    SURVEY_BATCH_SIZE,
                ),
                desc="Tokenizing subjects",
            )
        ):

            batch_timer = (
                time.perf_counter()
            )

            batch_subject_ids = (
                total_subjects[
                    batch_start:
                    batch_start
                    + SURVEY_BATCH_SIZE
                ]
            )

            # ====================================================
            # SURVEY TOKENIZATION
            # ====================================================

            survey_texts = [
                survey_data[subject_id]
                for subject_id
                in batch_subject_ids
            ]

            t0 = time.perf_counter()

            survey_encoded = (
                survey_tokenizer(
                    survey_texts,
                    add_special_tokens=True,
                    padding=False,
                    truncation=True,
                    max_length=
                        MAX_SURVEY_TOKENS,
                    return_attention_mask=False,
                )
            )

            survey_batch_ids = (
                survey_encoded[
                    "input_ids"
                ]
            )

            survey_time = (
                time.perf_counter()
                - t0
            )

            # ====================================================
            # COLLECT ARRAYS FOR ONE LARGE WRITE
            # ====================================================

            ehr_batch_to_write = []
            survey_batch_to_write = []

            ehr_processing_start = (
                time.perf_counter()
            )

            for local_idx, subject_id in enumerate(
                batch_subject_ids
            ):

                global_idx = (
                    batch_start
                    + local_idx
                )

                # --------------------------------------------
                # MEDS -> HF events
                # --------------------------------------------

                subject = database[
                    subject_id
                ]

                ehr_events = (
                    meds_subject_to_hf_events(
                        subject
                    )
                )

                # --------------------------------------------
                # EHR tokenize
                # --------------------------------------------

                ehr_encoded = (
                    ehr_tokenizer(
                        [ehr_events],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=
                            MAX_EHR_TOKENS,
                        return_tensors="pt",
                    )
                )

                ehr_ids = (
                    ehr_encoded[
                        "input_ids"
                    ]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(
                        np.int32,
                        copy=False,
                    )
                )

                survey_ids = np.asarray(
                    survey_batch_ids[
                        local_idx
                    ],
                    dtype=np.int32,
                )

                # --------------------------------------------
                # DO NOT WRITE YET.
                # Accumulate this whole batch.
                # --------------------------------------------

                ehr_batch_to_write.append(
                    ehr_ids
                )

                survey_batch_to_write.append(
                    survey_ids
                )

                # --------------------------------------------
                # Update offsets
                # --------------------------------------------

                ehr_token_count += (
                    ehr_ids.size
                )

                survey_token_count += (
                    survey_ids.size
                )

                ehr_offsets[
                    global_idx + 1
                ] = ehr_token_count

                survey_offsets[
                    global_idx + 1
                ] = survey_token_count

            ehr_processing_time = (
                time.perf_counter()
                - ehr_processing_start
            )

            # ====================================================
            # CONCATENATE ENTIRE BATCH
            # ====================================================

            t0 = time.perf_counter()

            ehr_batch_flat = (
                np.concatenate(
                    ehr_batch_to_write
                )
            )

            survey_batch_flat = (
                np.concatenate(
                    survey_batch_to_write
                )
            )

            concat_time = (
                time.perf_counter()
                - t0
            )

            # Release lists.
            del ehr_batch_to_write
            del survey_batch_to_write

            # ====================================================
            # ONLY TWO DISK WRITES PER 256 PATIENTS
            # ====================================================

            t0 = time.perf_counter()

            ehr_batch_flat.tofile(
                ehr_file
            )

            survey_batch_flat.tofile(
                survey_file
            )

            disk_write_time = (
                time.perf_counter()
                - t0
            )

            del ehr_batch_flat
            del survey_batch_flat

            # ====================================================
            # PROGRESS
            # ====================================================

            batch_time = (
                time.perf_counter()
                - batch_timer
            )

            processed = min(
                batch_start
                + len(batch_subject_ids),
                len(total_subjects),
            )

            if (
                batch_num < 3
                or (batch_num + 1) % 10 == 0
            ):

                elapsed = (
                    time.perf_counter()
                    - overall_start
                )

                print(
                    "\n"
                    "====================================\n"
                    f"BATCH {batch_num + 1}\n"
                    "====================================\n"
                    f"Subjects:          "
                    f"{len(batch_subject_ids)}\n"
                    f"Survey tokenize:   "
                    f"{survey_time:.3f}s\n"
                    f"EHR processing:    "
                    f"{ehr_processing_time:.3f}s\n"
                    f"Concatenation:     "
                    f"{concat_time:.3f}s\n"
                    f"Batch disk write:  "
                    f"{disk_write_time:.3f}s\n"
                    f"Total batch:       "
                    f"{batch_time:.3f}s\n"
                    "\n"
                    f"Processed:         "
                    f"{processed:,}/"
                    f"{len(total_subjects):,}\n"
                    f"Subjects/sec:      "
                    f"{processed / elapsed:.2f}\n"
                    "====================================\n",
                    flush=True,
                )

    # ========================================================
    # SAVE OFFSETS
    # ========================================================

    np.save(
        ehr_offsets_path,
        ehr_offsets,
    )

    np.save(
        survey_offsets_path,
        survey_offsets,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "num_subjects":
            len(total_subjects),

        "ehr_model_name":
            EHR_MODEL_NAME,

        "survey_model_name":
            SURVEY_MODEL_NAME,

        "max_ehr_tokens":
            MAX_EHR_TOKENS,

        "max_survey_tokens":
            MAX_SURVEY_TOKENS,

        "ehr_token_count":
            int(ehr_token_count),

        "survey_token_count":
            int(survey_token_count),

        "token_dtype":
            "int32",

        "ehr_pad_token_id":
            int(
                getattr(
                    ehr_tokenizer,
                    "pad_token_id",
                    None,
                )
                or 0
            ),

        "survey_pad_token_id":
            int(
                getattr(
                    survey_tokenizer,
                    "pad_token_id",
                    None,
                )
                or 0
            ),
    }

    with open(
        CACHE_DIR / "metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # LOCAL CACHE COMPLETE
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print(
        "\n"
        "============================================\n"
        "LOCAL TOKENIZATION CACHE COMPLETE\n"
        "============================================",
        flush=True,
    )

    print(
        f"Subjects: "
        f"{len(total_subjects):,}",
        flush=True,
    )

    print(
        f"Time: "
        f"{total_elapsed / 60:.2f} minutes",
        flush=True,
    )

    print(
        f"Throughput: "
        f"{len(total_subjects) / total_elapsed:.2f} "
        f"subjects/sec",
        flush=True,
    )

    print(
        f"Local cache: {CACHE_DIR}",
        flush=True,
    )

    # ========================================================
    # COPY FINISHED CACHE TO DATA BUCKET
    # ========================================================

    print(
        "\nCopying completed cache to data_bucket...",
        flush=True,
    )

    FINAL_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_start = (
        time.perf_counter()
    )

    for source_path in CACHE_DIR.iterdir():

        if not source_path.is_file():
            continue

        destination_path = (
            FINAL_CACHE_DIR
            / source_path.name
        )

        print(
            f"Copying {source_path.name}...",
            flush=True,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

    copy_time = (
        time.perf_counter()
        - copy_start
    )

    print(
        f"\nCopy complete in "
        f"{copy_time / 60:.2f} minutes.",
        flush=True,
    )

    print(
        f"Final cache: "
        f"{FINAL_CACHE_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()