import os
import os.path as osp
import shutil

import numpy as np
from mne import set_config
from moabb.datasets.download import (
    fs_get_file_hash,
    fs_get_file_id,
    fs_get_file_list,
    get_dataset_path,
)
from moabb.utils import set_download_dir
from pooch import HTTPDownloader, Unzip, retrieve

BEETL_URL = "https://ndownloader.figshare.com/files/"


class BeetlDataset:
    def __init__(self, figshare_id, code, subject_list):
        self.figshare_id = figshare_id
        self.code = code
        self.subject_list = subject_list

    def data_path(self, subject):
        pass

    def download(self, path=None, subjects=None):
        pass

    def get_data(self, subjects=None):
        pass


class BeetlSleepDataset(BeetlDataset):
    def __init__(self):
        super().__init__(
            figshare_id=14779407,
            code="beetlsleep",
            subject_list=range(10),
        )

    def data_path(self, subject):
        sign = self.code
        key_dest = "MNE-{:s}-data".format(sign.lower())
        path = osp.join(get_dataset_path(sign, None), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        spath = []
        for f in fsn.keys():
            if not osp.exists(osp.join(path, "s{}r1X.npy".format(subject))):
                retrieve(
                    BEETL_URL + fsn[f],
                    reg[fsn[f]],
                    fsn[f],
                    path,
                    processor=Unzip(),
                    downloader=HTTPDownloader(progressbar=True),
                )
                zpath = osp.join(path, fsn[f] + ".unzip")
                for i in range(10):
                    fx, fy = "s{}r1X.npy".format(i), "s{}r1y.npy".format(i)
                    shutil.move(osp.join(zpath, fx), osp.join(path, fx))
                    shutil.move(osp.join(zpath, fy), osp.join(path, fy))
                shutil.move(
                    osp.join(zpath, "headerInfo.npy"), osp.join(path, "headerInfo.npy")
                )
                os.rmdir(osp.join(path, fsn[f] + ".unzip"))
        spath.append(osp.join(path, "s{}r1X.npy".format(subject)))
        spath.append(osp.join(path, "s{}r1y.npy".format(subject)))
        spath.append(osp.join(path, "headerInfo.npy"))
        return spath

    def download(self, path=None, subjects=None):
        """Download datasets for sleep task

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects
        """
        if path:
            set_download_dir(path)
            set_config("MNE_DATASETS_{}_PATH".format(self.code.upper()), path)
            # TODO: Fix FileExistsError: [Errno 17] File exists: '/Users/sylchev/test_compet in
            # moabb/utils.py in set_download_dir(path), l. 54

        subjects = self.subject_list if subjects is None else subjects
        # Competition files
        spath = []
        for s in subjects:
            spath.append(self.data_path(s))
        return osp.dirname(spath[-1][0])

    def get_data(self, path=None, subjects=None):
        """Get data as list of numpy array, labels and metadata

        Parameters
        ----------
        path: str | None
            Path to download the data, store in ~/mne_data if None
        subjects: list | None
            list of subject, default=None to select all subjects

        Returns
        --------
        X_domain: list of ndarray, shape (n_trials, n_electrodes, n_samples)
            one ndarray for each dataset, with data
        y_domain: list of ndarray, shape (n_trials)
            label for a dataset
        metadata_domain: list of DataFrame
            list of metadata per dataset
        """
        subjects = self.subject_list if subjects is None else subjects
        spath = []
        for s in subjects:
            files = self.data_path(s)
            for f in files:
                if osp.basename(f) != "headerInfo.npy":
                    spath.append(f)
                else:
                    hd = f
        spath.append(hd)
        X_domain, y_domain, meta_domain = [], [], []
        for p in spath:
            d = np.load(p, allow_pickle=True)
            if osp.basename(p)[4] == "X":
                X_domain.append(d)
            elif osp.basename(p)[4] == "y":
                y_domain.append(d)
            elif osp.basename(p) == "headerInfo.npy":
                meta_domain = d
        X_domain = np.concatenate(X_domain)
        y_domain = np.concatenate(y_domain)

        return X_domain, y_domain, meta_domain
