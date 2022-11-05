import numpy as np
import torch

from torch.utils.data import Dataset


class PixelNeRFDataset(Dataset):
    def __init__(
        self,
        data_path,
        num_iters,
        test_source_pose_idx,
        test_target_pose_idx,
    ):
        self.data_path = data_path
        self.N = num_iters


        self.test_source_pose_idx = test_source_pose_idx
        self.test_target_pose_idx = test_target_pose_idx
    
        data = np.load(data_path)
    
        poses = data["poses"]
        images =  data["images"]

        self.poses = torch.Tensor(poses)

        self.channel_means = torch.Tensor([0.485, 0.456, 0.406])
        self.channel_stds = torch.Tensor([0.229, 0.224, 0.225])

        samp_img = images[0].copy()

        img_size = samp_img.shape[0]
        self.pix_idxs = np.arange(img_size ** 2)
        xs = torch.arange(img_size) - (img_size / 2 - 0.5)
        ys = torch.arange(img_size) - (img_size / 2 - 0.5)

        (xs, ys) = torch.Tensor(np.meshgrid(xs, -ys, indexing="xy"))
        focal = float(data["focal"])

        pixel_coords = torch.stack(
            [xs, ys, torch.full_like(xs, -focal)], dim=-1)
        camera_coords = pixel_coords / focal
        self.init_ds = camera_coords
        self.camera_distance = camera_distance = float(data["camera_distance"])
        self.init_o = torch.Tensor(np.array([0, 0, camera_distance]))
        # tan(theta) = opposite / adjacent.
        self.scale = (img_size / 2) / focal

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        obj_idx = np.random.randint(self.poses.shape[0])
        obj = self.objs[obj_idx]
        obj_dir = f"{self.data_path}/{obj}"

        source_pose_idx = np.random.randint(self.poses.shape[1])
        if obj_idx == self.test_obj_idx:
            while source_pose_idx == self.test_source_pose_idx:
                source_pose_idx = np.random.randint(self.poses.shape[1])

        source_img_f = f"{obj_dir}/{str(source_pose_idx).zfill(self.z_len)}.npy"
        source_image = torch.Tensor(np.load(source_img_f) / 255)
        source_image = (source_image - self.channel_means) / self.channel_stds
        source_pose = self.poses[obj_idx, source_pose_idx]
        source_R = source_pose[:3, :3]

        target_pose_idx = np.random.randint(self.poses.shape[1])
        if obj_idx == self.test_obj_idx:
            while (target_pose_idx == self.test_source_pose_idx) or (
                target_pose_idx == self.test_target_pose_idx
            ):
                target_pose_idx = np.random.randint(self.poses.shape[1])

        target_img_f = f"{obj_dir}/{str(target_pose_idx).zfill(self.z_len)}.npy"
        target_image = np.load(target_img_f)
        not_gray_pix = np.argwhere((target_image == 128).sum(-1) != 3)
        top_row = not_gray_pix[:, 0].min()
        bottom_row = not_gray_pix[:, 0].max()
        left_col = not_gray_pix[:, 1].min()
        right_col = not_gray_pix[:, 1].max()
        bbox = (top_row, left_col, bottom_row, right_col)

        target_image = np.load(target_img_f) / 255
        target_pose = self.poses[obj_idx, target_pose_idx]
        target_R = target_pose[:3, :3]

        R = source_R.T @ target_R

        return (source_image, torch.Tensor(R), torch.Tensor(target_image), bbox)

##### class for nerf dataset usage ###

# class PixelNeRFDataset(Dataset):
#     def __init__(
#         self,
#         data_path,
#         num_iters,
#         test_source_pose_idx,
#         test_target_pose_idx,
#     ):
#         self.data_path = data_path
#         self.N = num_iters


#         self.test_source_pose_idx = test_source_pose_idx
#         self.test_target_pose_idx = test_target_pose_idx
    
#         data = np.load(data_path)
    
#         poses = data["poses"]
#         images =  data["images"]

#         self.poses = torch.Tensor(poses)

#         self.channel_means = torch.Tensor([0.485, 0.456, 0.406])
#         self.channel_stds = torch.Tensor([0.229, 0.224, 0.225])

#         samp_img = images[0].copy()

#         img_size = samp_img.shape[0]
#         self.pix_idxs = np.arange(img_size ** 2)
#         xs = torch.arange(img_size) - (img_size / 2 - 0.5)
#         ys = torch.arange(img_size) - (img_size / 2 - 0.5)

#         (xs, ys) = torch.Tensor(np.meshgrid(xs, -ys, indexing="xy"))
#         focal = float(data["focal"])

#         pixel_coords = torch.stack(
#             [xs, ys, torch.full_like(xs, -focal)], dim=-1)
#         camera_coords = pixel_coords / focal
#         self.init_ds = camera_coords
#         self.camera_distance = camera_distance = float(data["camera_distance"])
#         self.init_o = torch.Tensor(np.array([0, 0, camera_distance]))
#         # tan(theta) = opposite / adjacent.
#         self.scale = (img_size / 2) / focal

#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         obj_idx = np.random.randint(self.poses.shape[0])
#         obj = self.objs[obj_idx]
#         obj_dir = f"{self.data_path}/{obj}"

#         source_pose_idx = np.random.randint(self.poses.shape[1])
#         if obj_idx == self.test_obj_idx:
#             while source_pose_idx == self.test_source_pose_idx:
#                 source_pose_idx = np.random.randint(self.poses.shape[1])

#         source_img_f = f"{obj_dir}/{str(source_pose_idx).zfill(self.z_len)}.npy"
#         source_image = torch.Tensor(np.load(source_img_f) / 255)
#         source_image = (source_image - self.channel_means) / self.channel_stds
#         source_pose = self.poses[obj_idx, source_pose_idx]
#         source_R = source_pose[:3, :3]

#         target_pose_idx = np.random.randint(self.poses.shape[1])
#         if obj_idx == self.test_obj_idx:
#             while (target_pose_idx == self.test_source_pose_idx) or (
#                 target_pose_idx == self.test_target_pose_idx
#             ):
#                 target_pose_idx = np.random.randint(self.poses.shape[1])

#         target_img_f = f"{obj_dir}/{str(target_pose_idx).zfill(self.z_len)}.npy"
#         target_image = np.load(target_img_f)
#         not_gray_pix = np.argwhere((target_image == 128).sum(-1) != 3)
#         top_row = not_gray_pix[:, 0].min()
#         bottom_row = not_gray_pix[:, 0].max()
#         left_col = not_gray_pix[:, 1].min()
#         right_col = not_gray_pix[:, 1].max()
#         bbox = (top_row, left_col, bottom_row, right_col)

#         target_image = np.load(target_img_f) / 255
#         target_pose = self.poses[obj_idx, target_pose_idx]
#         target_R = target_pose[:3, :3]

#         R = source_R.T @ target_R

#         return (source_image, torch.Tensor(R), torch.Tensor(target_image), bbox)
