import logging
import os
import re
from typing import List

import numpy as np
import pandas as pd
import nibabel as nib  # Import nibabel for loading .nii.gz files


class DataReader:
    """
    Parses paths to files, saves ids of patients and their diagnoses.
    """

    def __init__(self, caps_directories: List[str], info_data: List[str], diagnoses_info: List[str],
                 quality_check: bool, valid_dataset_names: list, info_data_cols: list) -> None:
        """
        Initialize with all required attributes.
        :param caps_directories: CAPS directory produced by the clinica library
        :param info_data: tabular data containing information about patients and their diagnosis
        :param diagnoses_info: dict about existing and valid diagnoses/labels
        :param quality_check: if True then samples will filtered based on MMSE values
        :param valid_dataset_names: valid names of the datasets
        :param info_data_cols: columns of the info data that will be considered
        """
        self.columns = info_data_cols
        self.valid_dataset_names = valid_dataset_names
        self.quality_check = quality_check
        self.diagnoses_info = diagnoses_info
        self.data = self.get_files_and_labels(caps_directories, info_data)

    @staticmethod
    def search_files(caps_directories: List[str]) -> pd.DataFrame:
        """
        Searches for all .nii.gz files containing information about an MRI scan.
        :param caps_directories: a list of the paths to the CAPS directories.
        :return: the pandas DataFrame containing the ID of a patient, the ID of a session, and the path to a .nii.gz file.
        """
        subjects_list = []
        sessions_list = []
        path_file_names_list = []
        for caps_dir in caps_directories:
            for root, dirs, files in os.walk(caps_dir):
                for name in files:
                    if name.endswith(".nii"):  # Adjusted to look for .nii.gz files
                        path_file_name = os.path.join(os.path.abspath(root), name)
                        # print(f"Found file: {path_file_name}")  # Log each found file

                        # Extract subject and session IDs based on filename structure
                        file_name_split = name.split('_')
                        subject_id = file_name_split[0]
                        assert subject_id.startswith("sub-")

                        session_id = file_name_split[1]
                        assert session_id.startswith("ses-")

                        subjects_list.append(subject_id)
                        sessions_list.append(session_id)
                        path_file_names_list.append(path_file_name)

        d = {'participant_id': subjects_list, 'session_id': sessions_list, 'file': path_file_names_list}
        return pd.DataFrame(data=d)

    @staticmethod
    def calculate_statistics(data: pd.DataFrame) -> None:
        """
        Logs some statistics, e.g., age, MMSE, unique patients
        :param data: info data as DataFrame
        """
        for dataset in data["dataset"].unique():
            logging.info("Calculating statistics for {} ...".format(dataset))
            data_ = data[data["dataset"] == dataset].copy()
            logging.info("Number of total samples is {}".format(data_.shape[0]))

            means = data_[["diagnosis", "mmse", "age"]].groupby('diagnosis').transform("mean")
            means.rename(columns={"mmse": "mean_mmse", "age": "mean_age"}, inplace=True)

            stds = data_[["diagnosis", "mmse", "age"]].groupby('diagnosis').transform("std")
            stds.rename(columns={"mmse": "std_mmse", "age": "std_age"}, inplace=True)

            stats = pd.concat([data_["diagnosis"].drop_duplicates(), means.drop_duplicates(), stds.drop_duplicates()],
                              axis=1)
            logging.info("\n{}".format(stats))

            data_["counter"] = 1
            gender_data = data_[["participant_id", "diagnosis", "sex", "counter"]].drop_duplicates().groupby(
                ['diagnosis', 'sex']).sum()
            logging.info("\n{}".format(gender_data))

            logging.info("Unique patients: {}".format(data_['participant_id'].nunique()))

    @staticmethod
    def _apply_filter(data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes samples that are 1 SD above/below of the MMSE's mean.
        :param data: info data as DataFrame
        :return: filtered DataFrame
        """
        data["mean"] = data[["diagnosis", "mmse"]].groupby('diagnosis').transform("mean")
        data["std"] = data[["diagnosis", "mmse"]].groupby('diagnosis').transform("std")
        data["threshold_pos"] = data["mean"] + data["std"]
        data["threshold_neg"] = data["mean"] - data["std"]
        data = data.dropna(subset=['mmse'])

        # Assure that age is less than 110 if some values are extremely large:
        data = data[(data["age"] < 110)]

        if 'NIFD' in data['participant_id'][0]:
            logging.info("Applying MMSE quality filter on NIFD ...")
            data = data[~((data["diagnosis"] == 'CN') & (data["mmse"] < data["threshold_neg"]))]
            data = data[~((data["diagnosis"] != 'CN') & (data["mmse"] > data["threshold_pos"]))]
        elif 'ADNI' in data['participant_id'][0] or 'AIBL' in data['participant_id'][0]:
            logging.info("Applying MMSE quality filter on ADNI/AIBL ...")
            data = data[~((data["diagnosis"] == 'CN') & (data["mmse"] < data["threshold_neg"]))]
            data = data[~((data["diagnosis"] == 'MCI') & (data["mmse"] > data["threshold_pos"]))]
            data = data[~((data["diagnosis"] == 'MCI') & (data["mmse"] < data["threshold_neg"]))]
            data = data[~((data["diagnosis"] == 'AD') & (data["mmse"] > data["threshold_pos"]))]
        elif 'OAS' in data['participant_id'][0]:
            logging.info("Applying MMSE quality filter on OASIS ...")
            data = data[~((data["diagnosis"] == 'CN') & (data["mmse"] < data["threshold_neg"]))]
            data = data[~((data["diagnosis"] == 'AD') & (data["mmse"] > data["threshold_pos"]))]
        else:
            logging.warning("Dataset {} is not relevant for data quality.".format(
                " ".join(re.findall("[a-zA-Z]+", data['participant_id'][0])).split(' ')[1]))

        return data

    def select_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Selects only specific columns.
        :param data: DataFrame object
        :return: filtered DataFrame object
        """
        for col in self.columns:
            if col not in data.columns:
                data[col] = None

        return data[self.columns]

    def filter_on_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters samples based on MMSE values.
        :param data: DataFrame object
        :return: filtered DataFrame object
        """

        if 'MMS' in data.columns:
            data.rename(columns={"MMS": "mmse"}, errors="raise", inplace=True)
        elif 'MMSE' in data.columns:
            data.rename(columns={"MMSE": "mmse"}, errors="raise", inplace=True)
        elif 'mmse' in data.columns:
            pass
        else:
            logging.warning("Data quality will not be performed for the dataset {}.".format(
                " ".join(re.findall("[a-zA-Z]+", data['participant_id'][0])).split(' ')[1]))
            return data

        return self._apply_filter(data)

    def read_info_data(self, info_data_list: List[str]) -> pd.DataFrame:
        df_list = []
        for info_data_path in info_data_list:
            info_data = pd.read_csv(info_data_path, sep=",", low_memory=False)
            info_data.columns = info_data.columns.str.strip()
            # print("Column names after stripping:", info_data.columns)
            info_data = info_data[self.columns]
            df_list.append(info_data)

        data = pd.concat(df_list)
        return data

    def get_files_and_labels(self, caps_directories: List[str], info_data: List[str]) -> pd.DataFrame:
        files_df = self.search_files(caps_directories)
        info_data_df = self.read_info_data(info_data)

        # Reformat participant_id to match files_df format
        info_data_df['participant_id'] = info_data_df['participant_id'].apply(
            lambda x: f"sub-{str(x).replace('_', '')}" if pd.notnull(x) else x
        )

        # print("files_df:")
        # print(files_df.head())
        # print("info_data_df after reformatting participant_id:")
        # print(info_data_df.head())

        # Filter out rows where participant_id or session_id is missing in info_data_df
        info_data_df = info_data_df.dropna(subset=["participant_id", "session_id"])

        # Print counts to help debug
        # print(f"Number of rows in files_df: {files_df.shape[0]}")
        # print(f"Number of rows in info_data_df before merging: {info_data_df.shape[0]}")

        files = []
        patients = []
        diagnoses = []

        for idx, row in info_data_df.iterrows():
            patient_id_search = row["participant_id"]
            session_id_search = row["session_id"]

            found_data = files_df[(files_df["participant_id"] == patient_id_search) &
                                (files_df["session_id"] == session_id_search)]

            if found_data.empty:
                # Exclude unmatched rows and log for troubleshooting
                # print(f"Skipping unmatched row: participant_id={patient_id_search}, session_id={session_id_search}")
                continue

            files.append(found_data["file"].values[0])
            patients.append(row["participant_id"])
            diagnoses.append(row["diagnosis"])

        # Print final row counts
        # print("Number of rows in info_data_df after filtering:", info_data_df.shape[0])
        # print("Length of diagnoses list:", len(diagnoses))

        # Check the assertion after logging details
        # if info_data_df.shape[0] != len(diagnoses):
        #     print(f"Warning: Mismatch in row counts between info_data_df and diagnoses list.")
        #     print(f"Rows in info_data_df: {info_data_df.shape[0]}, Diagnoses entries: {len(diagnoses)}")

        d = {'file': files, 'patient': patients, 'diagnosis': diagnoses}
        d['target'] = pd.factorize(d['diagnosis'])[0].astype(np.uint16)
        return pd.DataFrame(data=d)
