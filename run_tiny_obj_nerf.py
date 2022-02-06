import io
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch

from torch import nn, optim


def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os):
    u_is_c = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds)
    t_is_c = t_i_c_bin_edges + u_is_c * t_i_c_gap
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :]
    return (r_ts_c, t_is_c)


def render_radiance_volume(r_ts, ns, chunk_size, F, t_is, ds):
    r_ts_flat = r_ts.reshape((-1, 3))
    ns_rep = ns.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
    ns_flat = ns_rep.reshape((-1, 3))
    c_is = []
    sigma_is = []
    for chunk_start in range(0, r_ts_flat.shape[0], chunk_size):
        r_ts_batch = r_ts_flat[chunk_start : chunk_start + chunk_size]
        ds_batch = ns_flat[chunk_start : chunk_start + chunk_size]
        preds = F(r_ts_batch, ds_batch)
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


def run_one_iter_of_tiny_nerf(
    ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, ns, chunk_size, F_c
):
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    C_rs_c = render_radiance_volume(r_ts_c, ns, chunk_size, F_c, t_is_c, ds)
    return C_rs_c


class VeryTinyObjNeRFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_pos = 6
        self.L_dir = 4
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        net_width = 256
        self.early_mlp = nn.Sequential(
            nn.Linear(pos_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width + 1),
            nn.ReLU(),
        )
        self.late_mlp = nn.Sequential(
            nn.Linear(net_width + dir_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3),
            nn.Sigmoid(),
        )

    def forward(self, xs, ns):
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2 ** l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2 ** l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ns_encoded = [ns]
        for l_dir in range(self.L_dir):
            ns_encoded.append(torch.sin(2 ** l_dir * torch.pi * ns))
            ns_encoded.append(torch.cos(2 ** l_dir * torch.pi * ns))

        ns_encoded = torch.cat(ns_encoded, dim=-1)

        outputs = self.early_mlp(xs_encoded)
        sigma_is = outputs[:, 0]
        c_is = self.late_mlp(torch.cat([outputs[:, 1:], ns_encoded], dim=-1))
        return {"c_is": c_is, "sigma_is": sigma_is}


def main():
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda:0"
    F_c = VeryTinyObjNeRFModel().to(device)
    chunk_size = 16384

    lr = 5e-3
    optimizer = optim.Adam(F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    data_f = "66bdbc812bd0a196e194052f3f12cb2e_obj.npz"
    try:
        data = np.load(data_f)
    except FileNotFoundError:
        url = (
            f"https://github.com/airalcorn2/pytorch-nerf/blob/master/{data_f}?raw=True"
        )
        response = requests.get(url)
        data = np.load(io.BytesIO(response.content))
        np.savez(
            data_f,
            images=data["images"],
            poses=data["poses"],
            focal=float(data["focal"]),
            camera_distance=float(data["camera_distance"]),
        )

    images = data["images"] / 255
    img_size = images.shape[1]
    xs = torch.arange(img_size) - img_size / 2
    ys = torch.arange(img_size) - img_size / 2
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")
    focal = float(data["focal"])
    pixel_coords = torch.stack([xs, ys, -focal * torch.ones_like(xs)], dim=-1)
    camera_coords = pixel_coords / focal
    init_ds = camera_coords.to(device)
    init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

    test_idx = 13
    plt.imshow(images[test_idx])
    plt.show()
    test_img = torch.Tensor(images[test_idx]).to(device)
    poses = data["poses"]
    test_R = torch.Tensor(poses[test_idx, :, :3]).to(device)
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)
    test_os = (test_R @ init_o).expand(test_ds.shape)
    test_ns = torch.Tensor(poses[test_idx, :, 3]).expand(test_ds.shape).to(device)

    t_n = float(1)
    t_f = float(4)
    N_c = 32
    t_i_c_gap = (t_f - t_n) / N_c
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

    train_idxs = np.arange(len(images)) != test_idx
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])
    psnrs = []
    iternums = []
    num_iters = 20000
    display_every = 100
    F_c.train()
    for i in range(num_iters):
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        os = (R @ init_o).expand(ds.shape)
        ns = (R @ init_o).expand(ds.shape)

        C_rs_c = run_one_iter_of_tiny_nerf(
            ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, ns, chunk_size, F_c
        )
        loss = criterion(C_rs_c, images[target_img_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % display_every == 0:
            F_c.eval()
            with torch.no_grad():
                C_rs_c = run_one_iter_of_tiny_nerf(
                    test_ds,
                    N_c,
                    t_i_c_bin_edges,
                    t_i_c_gap,
                    test_os,
                    test_ns,
                    chunk_size,
                    F_c,
                )

            loss = criterion(C_rs_c, test_img)
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
