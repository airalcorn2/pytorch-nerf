import matplotlib.pyplot as plt
import numpy as np
import torch

from image_encoder import ImageEncoder
from pixelnerf_dataset import PixelNeRFDataset
from torch import nn, optim


def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os):
    u_is_c = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds)
    t_is_c = t_i_c_bin_edges + u_is_c * t_i_c_gap
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
    return (r_ts_c, t_is_c)


def get_fine_query_points(w_is_c, N_f, t_is_c, t_f, os, ds, r_ts_c, N_d, d_std, t_n):
    w_is_c = w_is_c + 1e-5
    pdfs = w_is_c / torch.sum(w_is_c, dim=-1, keepdim=True)
    cdfs = torch.cumsum(pdfs, dim=-1)
    cdfs = torch.cat([torch.zeros_like(cdfs[..., :1]), cdfs[..., :-1]], dim=-1)

    us = torch.rand(list(cdfs.shape[:-1]) + [N_f]).to(w_is_c)

    idxs = torch.searchsorted(cdfs.detach(), us.detach(), right=True)
    t_i_f_bottom_edges = torch.gather(t_is_c, 2, idxs - 1)
    idxs_capped = idxs.clone()
    max_ind = cdfs.shape[-1]
    idxs_capped[idxs_capped == max_ind] = max_ind - 1
    t_i_f_top_edges = torch.gather(t_is_c, 2, idxs_capped)
    t_i_f_top_edges[idxs == max_ind] = t_f
    t_i_f_gaps = t_i_f_top_edges - t_i_f_bottom_edges
    u_is_f = torch.rand_like(t_i_f_gaps).to(os)
    t_is_f = t_i_f_bottom_edges + u_is_f * t_i_f_gaps

    # See Section B.1 in the Supplementary Materials and:
    # https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/render/nerf.py#L150.
    t_is_d = (w_is_c * r_ts_c[..., 2]).sum(dim=-1)
    t_is_d = t_is_d.unsqueeze(2).repeat((1, 1, N_d))
    t_is_d = t_is_d + torch.normal(0, d_std, size=t_is_d.shape).to(t_is_d)
    t_is_d = torch.clamp(t_is_d, t_n, t_f)

    t_is_f = torch.cat([t_is_c, t_is_f.detach(), t_is_d], dim=-1)
    (t_is_f, _) = torch.sort(t_is_f, dim=-1)
    r_ts_f = os[..., None, :] + t_is_f[..., :, None] * ds[..., None, :]

    return (r_ts_f, t_is_f)


def get_image_features_for_query_points(r_ts, camera_distance, scale, W_i):
    # Get the projected image coordinates (pi_x_is) for each point along the rays
    # (r_ts). This is just geometry. See: http://www.songho.ca/opengl/gl_projectionmatrix.html.
    pi_x_is = r_ts[..., :2] / (camera_distance - r_ts[..., 2].unsqueeze(-1))
    pi_x_is = pi_x_is / scale
    # PyTorch's grid_sample function assumes (-1, -1) is the left-top pixel, but we want
    # (-1, -1) to be the left-bottom pixel, so we negate the y-coordinates.
    pi_x_is[..., 1] = -1 * pi_x_is[..., 1]
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
        r_ts_batch = r_ts_flat[chunk_start: chunk_start + chunk_size]
        ds_batch = ds_flat[chunk_start: chunk_start + chunk_size]
        w_is_batch = z_is_flat[chunk_start: chunk_start + chunk_size]
        preds = F(r_ts_batch, ds_batch, w_is_batch)
        c_is.append(preds["c_is"])
        sigma_is.append(preds["sigma_is"])

    c_is = torch.cat(c_is).reshape(r_ts.shape)
    sigma_is = torch.cat(sigma_is).reshape(r_ts.shape[:-1])

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

    return (C_rs, w_is)


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
    N_f,
    t_f,
    N_d,
    d_std,
    t_n,
    F_f,
):
    (r_ts_c, t_is_c) = get_coarse_query_points(
        ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    z_is_c = get_image_features_for_query_points(
        r_ts_c, camera_distance, scale, W_i)
    (C_rs_c, w_is_c) = render_radiance_volume(
        r_ts_c, ds, z_is_c, chunk_size, F_c, t_is_c
    )

    (r_ts_f, t_is_f) = get_fine_query_points(
        w_is_c, N_f, t_is_c, t_f, os, ds, r_ts_c, N_d, d_std, t_n
    )
    z_is_f = get_image_features_for_query_points(
        r_ts_f, camera_distance, scale, W_i)
    (C_rs_f, _) = render_radiance_volume(
        r_ts_f, ds, z_is_f, chunk_size, F_f, t_is_f)
    return (C_rs_c, C_rs_f)


class PixelNeRFFCResNet(nn.Module):
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
            xs_encoded.append(torch.sin(2**l_pos * np.pi * xs))
            xs_encoded.append(torch.cos(2**l_pos * np.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2**l_dir * np.pi * ds))
            ds_encoded.append(torch.cos(2**l_dir * np.pi * ds))

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


def load_data():
    # Initialize dataset and test object/poses.
    data_dir = "data"
    # See Section B.2.1 in the Supplementary Materials.
    num_iters = 10000
    test_obj_idx = 5
    test_source_pose_idx = 11
    test_target_pose_idx = 33
    train_dataset = PixelNeRFDataset(
        data_dir, num_iters, test_obj_idx, test_source_pose_idx, test_target_pose_idx
    )
    return train_dataset


def set_up_test_data(train_dataset, device):
    obj_idx = train_dataset.test_obj_idx
    obj = train_dataset.objs[obj_idx]
    data_dir = train_dataset.data_dir
    obj_dir = f"{data_dir}/{obj}"

    z_len = train_dataset.z_len
    source_pose_idx = train_dataset.test_source_pose_idx
    source_img_f = f"{obj_dir}/{str(source_pose_idx).zfill(z_len)}.npy"
    source_image = np.load(source_img_f) / 255
    source_pose = train_dataset.poses[obj_idx, source_pose_idx]
    source_R = source_pose[:3, :3]

    target_pose_idx = train_dataset.test_target_pose_idx
    target_img_f = f"{obj_dir}/{str(target_pose_idx).zfill(z_len)}.npy"
    target_image = np.load(target_img_f) / 255
    target_pose = train_dataset.poses[obj_idx, target_pose_idx]
    target_R = target_pose[:3, :3]

    R = torch.Tensor(source_R.T @ target_R).to(device)

    # plt.imshow(source_image)
    plt.imsave("results/src.png", source_image)
    # plt.show()
    source_image = torch.Tensor(source_image)
    source_image = (
        source_image - train_dataset.channel_means
    ) / train_dataset.channel_stds
    source_image = source_image.to(device).unsqueeze(0).permute(0, 3, 1, 2)
    # plt.imshow(target_image)
    plt.imsave("results/target.png", target_image)
    # plt.show()
    target_image = torch.Tensor(target_image).to(device)

    return (source_image, R, target_image)


def main():
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda:0"
    F_c = PixelNeRFFCResNet().to(device)
    F_f = PixelNeRFFCResNet().to(device)

    E = ImageEncoder().to(device)
    chunk_size = 1024 * 32
    # See Section B.2 in the Supplementary Materials.
    batch_img_size = 4
    n_batch_pix = batch_img_size**2
    n_objs = 4

    # See Section B.2 in the Supplementary Materials.
    lr = 1e-4
    optimizer = optim.Adam(list(F_c.parameters()) +
                           list(F_f.parameters()), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = load_data()

    camera_distance = train_dataset.camera_distance
    scale = train_dataset.scale
    t_n = 1.0
    t_f = 4.0
    img_size = train_dataset[0][2].shape[0]
    # See Section B.1 in the Supplementary Materials,
    # and: https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/conf/default.conf#L50,
    # and: https://github.com/sxyu/pixel-nerf/blob/a5a514224272a91e3ec590f215567032e1f1c260/src/render/nerf.py#L150.
    N_c = 64
    N_f = 16
    N_d = 16
    d_std = 0.01

    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    init_o = train_dataset.init_o.to(device)
    init_ds = train_dataset.init_ds.to(device)

    (test_source_image, test_R, test_target_image) = set_up_test_data(
        train_dataset, device
    )
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)

    psnrs = []
    iternums = []
    num_iters = train_dataset.N
    use_bbox = True
    num_bbox_iters = 300000
    display_every = 100
    F_c.train()
    F_f.train()
    E.eval()
    for i in range(num_iters):
        if i == num_bbox_iters:
            use_bbox = False

        loss = 0
        for obj in range(n_objs):
            try:
                (source_image, R, target_image, bbox) = train_dataset[0]
            except ValueError:
                continue

            R = R.to(device)
            ds = torch.einsum("ij,hwj->hwi", R, init_ds)
            os = (R @ init_o).expand(ds.shape)

            if use_bbox:
                pix_rows = np.arange(bbox[0], bbox[2])
                pix_cols = np.arange(bbox[1], bbox[3])
            else:
                pix_rows = np.arange(0, img_size)
                pix_cols = np.arange(0, img_size)

            pix_row_cols = np.meshgrid(pix_rows, pix_cols, indexing="ij")
            pix_row_cols = np.stack(pix_row_cols).transpose(
                1, 2, 0).reshape(-1, 2)
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

            # Extract feature pyramid from image. See Section 4.1, Section B.1 in the
            # Supplementary Materials, and: https://github.com/sxyu/pixel-nerf/blob/master/src/model/encoder.py.
            with torch.no_grad():
                W_i = E(source_image.unsqueeze(
                    0).permute(0, 3, 1, 2).to(device))

            (C_rs_c, C_rs_f) = run_one_iter_of_pixelnerf(
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
                N_f,
                t_f,
                N_d,
                d_std,
                t_n,
                F_f,
            )
            target_img = target_image.to(device)
            target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(
                C_rs_c.shape
            )
            loss += criterion(C_rs_c, target_img_batch)
            loss += criterion(C_rs_f, target_img_batch)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except AttributeError:
            continue

        if i % display_every == 0:
            F_c.eval()
            F_f.eval()

            with torch.no_grad():
                test_W_i = E(test_source_image)

                (_, C_rs_f) = run_one_iter_of_pixelnerf(
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
                    N_f,
                    t_f,
                    N_d,
                    d_std,
                    t_n,
                    F_f,
                )

            loss = criterion(C_rs_f, test_target_image)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            iternums.append(i)

            # plt.figure(figsize=(10, 4))
            # plt.subplot(121)
            plt.imsave(f"results/{i}_img.png", C_rs_f.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            # plt.plot(iternums, psnrs)
            # plt.title("PSNR")
            # plt.show()

            F_c.train()
            F_f.train()

    print("Done!")


if __name__ == "__main__":
    main()
