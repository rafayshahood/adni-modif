import logging
import os
import re
from typing import List

import numpy as np
import pandas as pd


class DataReader:
    """
    Parses paths to files, saves ids of patients and their diagnoses.
    """

    def __init__(self, caps_directories: List[str], info_data: List[str], diagnoses_info: List[str],
                 quality_check: bool) -> None:
        """
        Initialize with all required attributes.
        :param caps_directories: CAPS directory produced by the clinica library
        :param info_data: tabular data containing information about patients and their diagnosis
        :param quality_check: if True then samples will filtered based on MMSE values
        """
        self.quality_check = quality_check
        self.diagnoses_info = diagnoses_info
        self.data = self.get_files_and_labels(caps_directories, info_data)

    @staticmethod
    def search_files(caps_directories: List[str]) -> pd.DataFrame:
        """
        Search for all PyTorch tensors containing information about an MRI scan.
        :param caps_directories: a list of the paths to the CAPS directories.
        :return: the pandas data frame containing the ID of a patient, the ID of a session, and the path to a PyTorch.
        tensor.
        """
        subjects_list = []
        sessions_list = []
        path_file_names_list = []
        for caps_dir in caps_directories:
            for root, dirs, files in os.walk(caps_dir):
                for name in files:
                    if name.endswith(".pt"):
                        path_file_name = os.path.join(os.path.abspath(root), name)

                        # Get the file name
                        path_file_name_split = name.split('/')
                        file_name = path_file_name_split[len(path_file_name_split) - 1]
                        file_name_split = file_name.split('_')

                        # Subject ID
                        subject_id = file_name_split[0]
                        assert subject_id.startswith("sub-")

                        # Session ID
                        session_id = file_name_split[1]
                        assert session_id.startswith("ses-")

                        subjects_list.append(subject_id)
                        sessions_list.append(session_id)
                        path_file_names_list.append(path_file_name)

        d = {'participant_id': subjects_list, 'session_id': sessions_list, 'file': path_file_names_list}
        return pd.DataFrame(data=d)

    def _apply_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes samples that are above/below of 1 SD of the mean of MMSE.
        :param data: pd.DataFrame
        :return: filtered pd.DataFrame
        """
        data["mean"] = data[["diagnosis", "mmse"]].groupby('diagnosis').transform("mean")
        data["std"] = data[["diagnosis", "mmse"]].groupby('diagnosis').transform("std")
        data["threshold_pos"] = data["mean"] + data["std"]
        data["threshold_neg"] = data["mean"] - data["std"]
        data = data.dropna(subset=['mmse'])

        if 'NIFD' in data['participant_id'][0]:
            logging.info("Processing NIFD ...")
            # drop 'CN' samples where MMSE < (mean + 1 std)
            data = data[~((data["diagnosis"] == 'CN') & (data["mmse"] < data["threshold_neg"]))]

            # drop 'FTD' samples where MMSE > (mean + 1 std)
            data = data[~((data["diagnosis"] != 'CN') & (data["mmse"] > data["threshold_pos"]))]

        elif 'ADNI' in data['participant_id'][0] or 'AIBL' in data['participant_id'][0]:
            logging.info("Processing ADNI/AIBL ...")
            # drop 'CN' samples where MMSE < (mean + 1 std)
            data = data[~((data["diagnosis"] == 'CN') & (data["mmse"] < data["threshold_neg"]))]

            # drop 'MCI' samples where MMSE > (mean + 1 std)
            data = data[~((data["diagnosis"] == 'MCI') & (data["mmse"] > data["threshold_pos"]))]
            data = data[~((data["diagnosis"] == 'MCI') & (data["mmse"] < data["threshold_neg"]))]

            # drop 'AD' samples where MMSE > (mean + 1 std)
            data = data[~((data["diagnosis"] == 'AD') & (data["mmse"] > data["threshold_pos"]))]
        else:
            logging.warning("Dataset {} is not relevant for data quality.".format(
                " ".join(re.findall("[a-zA-Z]+", data['participant_id'][0])).split(' ')[1]))

        return data

    def filter_on_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters samples based on MMSE values.
        :param data: pd.DataFrame
        :return: filtered pd.DataFrame
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
        """
        Read the information about available MRI scans in the TSV files.
        :param info_data_list: a list of the paths to the TSV files containing targets/labels.
        :return: the pandas data frame containing the ID of a patient, the ID of a session, and the corresponding
        target/label.
        """
        cols_to_read = ["participant_id", "session_id", "diagnosis"]
        df_list = []
        for info_data in info_data_list:
            info_data = pd.read_csv(info_data, sep="\t")
            info_data.loc[info_data['diagnosis'].isin(self.diagnoses_info['control_labels']), 'diagnosis'] = "CN"
            info_data.loc[info_data['diagnosis'].isin(self.diagnoses_info['ad_labels']), 'diagnosis'] = "AD"
            info_data = info_data[info_data['diagnosis'].isin(self.diagnoses_info['valid_diagnoses'])]
            if self.quality_check:
                info_data = self.filter_on_quality(info_data)
                info_data = info_data[cols_to_read]
            df_list.append(info_data)
        data = pd.concat(df_list)
        data = data.dropna()
        if self.diagnoses_info['merge_ftd']:
            data.loc[data['diagnosis'].isin(self.diagnoses_info['ftd_labels']), 'diagnosis'] = "FTD"
        assert ((data.groupby(['participant_id', 'session_id']).count().reset_index()["diagnosis"] == 1) == True).all()
        return data

    def get_files_and_labels(self, caps_directories: List[str], info_data: List[str]) -> pd.DataFrame:
        """
        Search for PyTorch tensors in the CAPS directories and for the corresponding targets/labels in TSV files.
        :param caps_directories: a list of the paths to the CAPS directories.
        :param info_data: a list of the paths to the TSV files containing targets/labels.
        """
        files_df = DataReader.search_files(caps_directories)
        info_data_df = self.read_info_data(info_data)

        files = []
        patients = []
        diagnoses = []
        for idx, row in info_data_df.iterrows():
            patient_id_search = row["participant_id"]
            session_id_search = row["session_id"]
            found_data = files_df[(files_df["participant_id"] == patient_id_search) &
                                  (files_df["session_id"] == session_id_search)]
            if found_data.empty:
                logging.warning(
                    "No data are found for patient {} and session {}".format(patient_id_search, session_id_search))
                continue
            files.append(found_data["file"].values[0])
            patients.append(row["participant_id"])
            diagnoses.append(row["diagnosis"])
        logging.info("Total number of samples: {}".format(len(diagnoses)))
        logging.info("Counts: {}".format(dict(
            zip(list(diagnoses), [list(diagnoses).count(i) for i in list(diagnoses)]))))

        d = {'file': files, 'patient': patients, 'diagnosis': diagnoses}
        d['target'] = pd.factorize(d['diagnosis'])[0].astype(np.uint16)
        return pd.DataFrame(data=d)
