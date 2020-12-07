import os
import torch
import logging

import numpy as np
from torch.utils.data import Dataset, DataLoader

from compute_face_reconstruction_diff import load_latest_gan, compute_diff, \
    compute_reconstruction
from utils import init_logger


def validate_data_in_files(face_crop_path):
    return os.path.getsize(face_crop_path)


class Sample:
    def __init__(self, video_name, frame_id, face_crop_path,
                 face_reconstruction_diff, reconstruction_diff_path):
        self.video_name = video_name
        self.frame_id = frame_id
        self.face_crop_path = face_crop_path
        self.face_reconstruction_diff = face_reconstruction_diff
        self.reconstruction_diff_path = reconstruction_diff_path


class FaceReconstructionDataset(Dataset):
    """Face reconstruction datasets."""

    PRISTINE_LABEL = 1
    FAKE_LABEL = 1 - PRISTINE_LABEL

    def __init__(self, root, reconstruct_again=False):
        """
        Args:
            root (string): Path to the dataset root directory
        Assuming that the test/trin directory has the two following
        sub-directories:
        1. face_crops (string): Path to the root directory of the
        face crops images.
        So, we need to count the number of frames in all videos.
        """
        self.cache = dict()
        self.logger = init_logger('dataset')
        self.reconstruct_again = reconstruct_again

        self.root = root
        self.face_crops_root_dir = os.path.join(root, "face_crops")
        self.reconstruction_diff_root_dir = os.path.join(root,
                                                         "reconstruction_diff")

        self.num_of_pristine_samples = 0
        self.num_of_fake_samples = 0

        self._generator, _ = load_latest_gan()

        self.index_to_video_and_frame, self.video_and_frame_to_sample = \
            self.build_index_to_video_and_frame()

    def calculate_full_paths(self, video, frame):
        # calculate the path to the face crop image
        face_crop_path = os.path.join(
            self.face_crops_root_dir, video, f"{frame}")
        reconstruction_diff_path = os.path.join(
            self.reconstruction_diff_root_dir, video,
            f"{frame[len('frame'):-len('.png')]}.npy")
        return face_crop_path, reconstruction_diff_path

    def calculate_diff(self, face_crop_path, reconstruction_diff_path):
        transformed_image, input_tensor, filled_samples, reconstructed_face, \
            original_face = compute_reconstruction(face_crop_path,
                                                   generator=self._generator)

        return compute_diff(reconstructed_face,
                            original_face.cuda(),
                            reconstruction_diff_path)

    @staticmethod
    def load_diff_from_path( reconstruction_diff_path):

        return np.load(reconstruction_diff_path)

    def build_index_to_video_and_frame(self):
        index_to_video_and_frame = dict()
        video_and_frame_to_sample = dict()

        all_videos = os.listdir(self.face_crops_root_dir)
        all_videos.sort(key=lambda vid: int(vid))

        index = 0
        for video in all_videos:
            all_frames_in_video = os.listdir(os.path.join(
                self.face_crops_root_dir, video))
            all_frames_in_video.sort(
                key=lambda frame_id: int(frame_id[len("frame"):-len(".png")]))

            for frame in all_frames_in_video:
                face_crop_path, reconstruction_diff_path = \
                    self.calculate_full_paths(video, frame)
                if validate_data_in_files(face_crop_path):
                    # compute or load reconstruction difference
                    if os.path.exists(reconstruction_diff_path) and not \
                            self.reconstruct_again:
                        diff = self.load_diff_from_path(
                            reconstruction_diff_path)
                    else:
                        try:
                            diff = self.calculate_diff(face_crop_path,
                                                       reconstruction_diff_path)
                        except IndexError:
                            msg = f"Face not found: vid={video}, frame={frame}"
                            self.logger.warning(msg)
                            continue
                        except Exception as e:
                            msg = f"Got error {e} for vid={video}, frame={frame}"
                            self.logger.warning(msg)
                            continue

                    # populate the index to video and frame mapping
                    index_to_video_and_frame[index] = (video, frame)
                    index += 1

                    # log the kind of sample - real / fake
                    if "_" in video:
                        self.num_of_fake_samples += 1
                    else:
                        self.num_of_pristine_samples += 1

                    # create a sample object from the kernels and face crops
                    # paths.
                    video_and_frame_to_sample[(video, frame)] = Sample(
                        video_name=video,
                        frame_id=frame,
                        face_crop_path=face_crop_path,
                        face_reconstruction_diff=diff,
                        reconstruction_diff_path=reconstruction_diff_path)
                else:
                        message = f"face crop does not exist: {face_crop_path}"
                        self.logger.debug(message)

        return index_to_video_and_frame, video_and_frame_to_sample

    def __len__(self):
        """The number of samples in the dataset  is the size of the video,
        frame to sample dictionary."""
        return len(self.video_and_frame_to_sample)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # deduce the sample from index
        video_and_frame = self.index_to_video_and_frame[idx]
        # check cache to improve performance
        if video_and_frame in self.cache:
            return self.cache[video_and_frame]

        video = video_and_frame[0]
        # label is 1 for pristine videos (not containing under-score), 0 fake
        label = self.PRISTINE_LABEL if "_" not in video else self.FAKE_LABEL
        sample = self.video_and_frame_to_sample[video_and_frame]

        sample = {'reconstruction diff': sample.face_reconstruction_diff,
                  'label': label}
        self.cache[video_and_frame] = sample
        return sample


def print_dataset_statistics(dataset, name):
    count_pristine_labels = 0
    count_fake_labels = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample['label'] == FaceReconstructionDataset.PRISTINE_LABEL:
            count_pristine_labels += 1
        elif sample['label'] == FaceReconstructionDataset.FAKE_LABEL:
            count_fake_labels += 1
        else:
            print(f"Something went wrong with: sample in "
                  f"index {idx},label: {sample['label']}")

    print(f"The dataset {name} has {len(dataset)} samples")
    print(f"The dataset {name} has {count_pristine_labels} pristine "
          f"samples")
    print(f"The dataset {name} has {count_fake_labels} fake samples")


if __name__ == "__main__":
    train_dataset = FaceReconstructionDataset(
        '/mnt/data/deepfakes/context_encoder_dataset/train/',
        reconstruct_again=True)
    print(f"len(train_dataset) = {len(train_dataset)}")
    test_dataset = FaceReconstructionDataset(
        '/mnt/data/deepfakes/context_encoder_dataset/test/',
        reconstruct_again=True)
    print(f"len(test_dataset) = {len(test_dataset)}")

    print_dataset_statistics(train_dataset, name='train')
    print_dataset_statistics(test_dataset, name='test')
