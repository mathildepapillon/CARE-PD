"""H36M data reader — loads preprocessed Human3.6M data in CARE-PD format.

Expects the output of ``scripts/prepare_h36m_dataset.py``:

*  An NPZ file where each key is ``S{subj}__{action}_{trial}`` and each value
   is an ``(T, 17, 3)`` float32 array (3-D joint positions in metres).
*  A pickle file containing ``{"labels": {seq→int}, "participant": {seq→str},
   "action_to_label": {str→int}, "n_classes": int}``.

Provides the same public attributes as ``BMCLabReader`` so the existing
``DataPreprocessor`` / ``dataset_factory`` pipeline works unchanged.
"""

from __future__ import annotations

import pickle

import numpy as np
from tqdm import tqdm


class H36MReader:
    """Read preprocessed Human3.6M data (output of prepare_h36m_dataset.py)."""

    def __init__(self, joints_path_list, labels_path, params):
        self.joints_path_list = joints_path_list
        self.labels_path = labels_path
        self.params = params

        with open(labels_path, "rb") as f:
            self._label_bundle = pickle.load(f)

        (
            self.pose_dict,
            self.labels_dict,
            self.video_names,
            self.participant_ID,
            self.metadata_dict,
            self.medication_dict,
            self.FoG_labels_dict,
        ) = self._read_all()

        print(f"[H36MReader] {len(self.pose_dict)} sequences loaded.")
        subjects = sorted(set(self.participant_ID))
        print(f"[H36MReader] {len(subjects)} subjects: {subjects}")
        unique, counts = np.unique(
            list(self.labels_dict.values()), return_counts=True
        )
        print("[H36MReader] Label distribution:")
        print(np.column_stack((unique, counts)))

    def _read_all(self):
        pose_dict: dict[str, np.ndarray] = {}
        labels_dict: dict[str, int] = {}
        metadata_dict: dict[str, np.ndarray] = {}
        medication_dict: dict[str, str] = {}
        fog_dict: dict[str, str] = {}
        video_names: list[str] = []
        participant_ids: list[str] = []

        seq_labels = self._label_bundle["labels"]
        seq_participants = self._label_bundle["participant"]

        dummy_meta = np.zeros((1, 5), dtype=np.float32)

        for joints_path in self.joints_path_list:
            seqs = np.load(joints_path, allow_pickle=True)
            for seq_name in tqdm(seqs.keys(), desc="H36M"):
                joints = seqs[seq_name]
                label = seq_labels[seq_name]
                subj = seq_participants[seq_name]

                # Append _view0 to match BMCLabReader convention.
                dict_name = f"{seq_name}_view0"
                pose_dict[dict_name] = joints
                labels_dict[dict_name] = label
                metadata_dict[dict_name] = dummy_meta
                medication_dict[dict_name] = "N/A"
                fog_dict[dict_name] = "N/A"
                video_names.append(dict_name)
                participant_ids.append(subj)

        return (
            pose_dict,
            labels_dict,
            video_names,
            participant_ids,
            metadata_dict,
            medication_dict,
            fog_dict,
        )
