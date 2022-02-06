import matplotlib.pyplot as plt
import numpy as np
import torch

from image_encoder import ImageEncoder
from pixelnerf_dataset import PixelNeRFDataset
from torch import nn, optim


def seed_worker(worker_id):
    # See: https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os):
    u_is_c = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds)
    t_is_c = t_i_c_bin_edges + u_is_c * t_i_c_gap
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
    return (r_ts_c, t_is_c)


def get_image_features_for_query_points(r_ts, camera_distance, scale, W_i):
    # Get the projected image coordinates (pi_x_is) for each point along the rays
    # (r_ts). This is just geometry. See: http://www.songho.ca/opengl/gl_projectionmatrix.html.
    pi_x_is = r_ts[..., :2] / (camera_distance - r_ts[..., 2].unsqueeze(-1))
    pi_x_is = pi_x_is / scale
    pi_x_is = pi_x_is.clamp(-1, 1)
    # PyTorch's grid_sample function assumes (-1, -1) is the left-top pixel, but we want
    # (-1, -1) to be the left-bottom pixel, so we negate the y-coordinates.
    # pi_x_is[..., 1] = -1 * pi_x_is[..., 1]
    # PyTorch's grid_sample function expects the grid to have shape
    # (N, H_out, W_out, 2).
    pi_x_is = pi_x_is.permute(2, 0, 1, 3)
    # PyTorch's grid_sample function expects the input to have shape (N, C, H_in, W_in).
    W_i = W_i.repeat(pi_x_is.shape[0], 1, 1, 1)
    # Get the image features (z_is) associated with the projected image coordinates
    # (pi_x_is) from the encoded image features (W_i). See Section 4.2.
    z_is = nn.functional.grid_sample(
        W_i, pi_x_is, align_corners=True, padding_mode="border"
    )
    # Convert shape back to match rays.
    z_is = z_is.permute(2, 3, 0, 1)
    return z_is


def render_radiance_volume(r_ts, ds, z_is, chunk_size, F, t_is):
    r_ts_flat = r_ts.reshape((-1, 3))
    ds_rep = ds.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
    ds_flat = ds_rep.reshape((-1, 3))
    z_is_flat = z_is.reshape((ds_flat.shape[0], -1))
    c_is = []
    sigma_is = []
    for chunk_start in range(0, r_ts_flat.shape[0], chunk_size):
        r_ts_batch = r_ts_flat[chunk_start : chunk_start + chunk_size]
        ds_batch = ds_flat[chunk_start : chunk_start + chunk_size]
        w_is_batch = z_is_flat[chunk_start : chunk_start + chunk_size]
        preds = F(r_ts_batch, ds_batch, w_is_batch)
        c_is.append(preds["c_is"])
        sigma_is.append(preds["sigma_is"])

    c_is = torch.cat(c_is)
    c_is = torch.reshape(c_is, r_ts.shape)
    sigma_is = torch.cat(sigma_is)
    sigma_is = torch.reshape(sigma_is, r_ts.shape[:-1])

    delta_is = t_is[..., 1:] - t_is[..., :-1]
    one_e_10 = torch.Tensor([1e10]).expand(delta_is[..., :1].shape)
    delta_is = torch.cat([delta_is, one_e_10.to(delta_is)], dim=-1)
    delta_is = delta_is * ds.norm(dim=-1).unsqueeze(-1)

    alpha_is = 1.0 - torch.exp(-sigma_is * delta_is)

    T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
    T_is = torch.roll(T_is, 1, -1)
    T_is[..., 0] = 1.0

    w_is = T_is * alpha_is

    C_rs = (w_is[..., None] * c_is).sum(dim=-2)

    return C_rs


def run_one_iter_of_pixelnerf(
    ds,
    N_c,
    t_i_c_bin_edges,
    t_i_c_gap,
    os,
    camera_distance,
    scale,
    W_i,
    chunk_size,
    F_c,
):
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    z_is = get_image_features_for_query_points(r_ts_c, camera_distance, scale, W_i)
    C_rs_c = render_radiance_volume(r_ts_c, ds, z_is, chunk_size, F_c, t_is_c)
    return C_rs_c


class PixelNeRFModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Number of encoding functions for positions. See Section B.1 in the
        # Supplementary Materials.
        self.L_pos = 6
        # Number of encoding functions for viewing directions.
        self.L_dir = 0
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        # Set up ResNet MLP. See Section B.1 and Figure 18 in the Supplementary
        # Materials.
        net_width = 512
        self.first_layer = nn.Sequential(
            nn.Linear(pos_enc_feats + dir_enc_feats, net_width)
        )
        self.n_resnet_blocks = 5
        z_linears = []
        mlps = []
        for resnet_block in range(self.n_resnet_blocks):
            z_linears.append(nn.Linear(net_width, net_width))
            mlps.append(
                nn.Sequential(
                    nn.Linear(net_width, net_width),
                    nn.ReLU(),
                    nn.Linear(net_width, net_width),
                    nn.ReLU(),
                )
            )

        self.z_linears = nn.ModuleList(z_linears)
        self.mlps = nn.ModuleList(mlps)
        self.final_layer = nn.Linear(net_width, 4)

    def forward(self, xs, ds, zs):
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2 ** l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2 ** l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2 ** l_dir * torch.pi * ds))
            ds_encoded.append(torch.cos(2 ** l_dir * torch.pi * ds))

        ds_encoded = torch.cat(ds_encoded, dim=-1)

        # Use the network to predict colors (c_is) and volume densities (sigma_is) for
        # 3D points (xs) along rays given the viewing directions (ds) of the rays
        # and the associated input image features (zs). See Section B.1 and Figure 18 in
        # the Supplementary Materials and:
        # https://github.com/sxyu/pixel-nerf/blob/master/src/model/resnetfc.py.
        outputs = self.first_layer(torch.cat([xs_encoded, ds_encoded], dim=-1))
        for block_idx in range(self.n_resnet_blocks):
            resnet_zs = self.z_linears[block_idx](zs)
            outputs = outputs + resnet_zs
            outputs = self.mlps[block_idx](outputs) + outputs

        outputs = self.final_layer(outputs)
        sigma_is = torch.relu(outputs[:, 0])
        c_is = torch.sigmoid(outputs[:, 1:])
        return {"c_is": c_is, "sigma_is": sigma_is}


def main():
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda:0"
    F_c = PixelNeRFModel().to(device)
    E = ImageEncoder().to(device)

    chunk_size = 1024 * 32
    # See Section B.2 in the Supplementary Materials.
    batch_img_size = 12
    n_batch_pix = batch_img_size ** 2
    n_objs = 4

    # Initialize dataset and test object/poses.
    data_dir = "data"
    num_iters = 20000
    test_obj_idx = 5
    test_source_pose_idx = 11
    test_target_pose_idx = 33
    train_dataset = PixelNeRFDataset(
        data_dir, num_iters, test_obj_idx, test_source_pose_idx, test_target_pose_idx
    )

    init_o = train_dataset.init_o.to(device)
    init_ds = train_dataset.init_ds.to(device)
    camera_distance = train_dataset.camera_distance
    scale = train_dataset.scale
    z_len = train_dataset.z_len

    test_obj = train_dataset.objs[test_obj_idx]
    test_obj_dir = f"{data_dir}/{test_obj}"

    test_source_img_f = f"{test_obj_dir}/{str(test_source_pose_idx).zfill(z_len)}.npy"
    test_source_image = np.load(test_source_img_f) / 255
    test_source_pose = train_dataset.poses[test_obj_idx, test_source_pose_idx]
    test_source_R = test_source_pose[:3, :3]

    test_target_img_f = f"{test_obj_dir}/{str(test_target_pose_idx).zfill(z_len)}.npy"
    test_target_image = np.load(test_target_img_f) / 255
    test_target_pose = train_dataset.poses[test_obj_idx, test_target_pose_idx]
    test_target_R = test_target_pose[:3, :3]

    test_R = torch.Tensor(test_target_R @ test_source_R.T).to(device)

    plt.imshow(test_source_image)
    plt.show()
    test_source_image = torch.Tensor(test_source_image)
    test_source_image = (
        test_source_image - train_dataset.channel_means
    ) / train_dataset.channel_stds
    test_source_image = test_source_image.to(device).unsqueeze(0).permute(0, 3, 1, 2)
    plt.imshow(test_target_image)
    plt.show()
    test_target_image = torch.Tensor(test_target_image).to(device)

    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    # See Section B.2 in the Supplementary Materials.
    lr = 1e-4
    train_params = list(F_c.parameters())
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()

    t_n = float(1)
    t_f = float(4)
    N_c = 32
    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    psnrs = []
    iternums = []
    display_every = 100
    F_c.train()
    E.eval()
    for i in range(num_iters):
        loss = 0
        for obj in range(n_objs):
            try:
                (source_image, R, target_image, bbox) = train_dataset[0]
            except ValueError:
                continue

            R = R.to(device)
            ds = torch.einsum("ij,hwj->hwi", R, init_ds)
            os = (R @ init_o).expand(ds.shape)

            pix_rows = np.arange(bbox[0], bbox[2])
            pix_cols = np.arange(bbox[1], bbox[3])
            pix_row_cols = np.meshgrid(pix_rows, pix_cols, indexing="ij")
            pix_row_cols = np.stack(pix_row_cols).transpose(1, 2, 0).reshape(-1, 2)
            choices = np.arange(len(pix_row_cols))
            try:
                selected_pix = np.random.choice(choices, n_batch_pix, False)
            except ValueError:
                continue

            pix_idx_rows = pix_row_cols[selected_pix, 0]
            pix_idx_cols = pix_row_cols[selected_pix, 1]
            ds_batch = ds[pix_idx_rows, pix_idx_cols].reshape(
                batch_img_size, batch_img_size, -1
            )
            os_batch = os[pix_idx_rows, pix_idx_cols].reshape(
                batch_img_size, batch_img_size, -1
            )

            # Extract feature pyramid from image. See Section 4.1., Section B.1 in the
            # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
            with torch.no_grad():
                W_i = E(source_image.unsqueeze(0).permute(0, 3, 1, 2).to(device))

            C_rs_c = run_one_iter_of_pixelnerf(
                ds_batch,
                N_c,
                t_i_c_bin_edges,
                t_i_c_gap,
                os_batch,
                camera_distance,
                scale,
                W_i,
                chunk_size,
                F_c,
            )
            target_img = target_image.to(device)
            target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(
                C_rs_c.shape
            )
            loss += criterion(C_rs_c, target_img_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % display_every == 0:
            F_c.eval()
            with torch.no_grad():
                test_W_i = E(test_source_image)

                C_rs_c = run_one_iter_of_pixelnerf(
                    test_ds,
                    N_c,
                    t_i_c_bin_edges,
                    t_i_c_gap,
                    test_os,
                    camera_distance,
                    scale,
                    test_W_i,
                    chunk_size,
                    F_c,
                )

            loss = criterion(C_rs_c, test_target_image)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(C_rs_c.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            F_c.train()

    print("Done!")


if __name__ == "__main__":
    main()
