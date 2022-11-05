import numpy as np
import os
import sys

from pyrr import Matrix44
from renderer import gen_rotation_matrix_from_azim_elev_in_plane, Renderer
from renderer_settings import *

SHAPENET_DIR = "data/ShapeNetCore.v2"


def main():
    # Set up the renderer.
    renderer = Renderer(
        camera_distance=CAMERA_DISTANCE,
        angle_of_view=ANGLE_OF_VIEW,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    # See Section 5.1.1.
    img_size = 128
    # Calculate focal length in pixel units. This is just geometry. See:
    # https://en.wikipedia.org/wiki/Angle_of_view#Derivation_of_the_angle-of-view_formula.
    focal = (img_size / 2) / np.tan(np.radians(ANGLE_OF_VIEW) / 2)

    # Generate car renders using random camera locations.
    init_cam_pos = np.array([0, 0, CAMERA_DISTANCE])
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    # See Section 5.1.1.
    samps = 50
    z_len = len(str(samps - 1))
    data_dir = "rendered"
    poses = []
    os.mkdir(data_dir)

    # Car category.
    cat = "02773838"
    objs = os.listdir(f"{SHAPENET_DIR}/{cat}")
    used_objs = []
    for obj in objs:
        # Load the ShapeNet object.
        obj_mtl_path = f"{SHAPENET_DIR}/{cat}/{obj}/models/model_normalized"
        try:
            renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
            sys.stderr.flush()
        except OSError:
            print(f"{SHAPENET_DIR}/{cat}/{obj} is empty.", flush=True)
            continue

        except FloatingPointError:
            print(f"{SHAPENET_DIR}/{cat}/{obj} divides by zero.", flush=True)

        obj_dir = f"{data_dir}/{obj}"
        os.mkdir(obj_dir)
        obj_poses = []
        for samp_idx in range(samps):
            angles = {
                "azimuth": np.random.uniform(-np.pi, np.pi),
                "elevation": np.random.uniform(-np.pi, np.pi),
            }
            R = gen_rotation_matrix_from_azim_elev_in_plane(**angles)
            eye = tuple((R @ init_cam_pos).flatten())
            look_at = Matrix44.look_at(eye, target, up)
            renderer.prog["VP"].write(
                (look_at @ renderer.perspective).astype("f4").tobytes()
            )
            renderer.prog["cam_pos"].value = eye

            image = renderer.render(0.5, 0.5, 0.5).resize((img_size, img_size))
            np.save(f"{obj_dir}/{str(samp_idx).zfill(z_len)}.npy",
                    np.array(image))

            pose = np.eye(4)
            pose[:3, :3] = np.array(look_at[:3, :3])
            pose[:3, 3] = -look_at[:3, :3] @ look_at[3, :3]
            obj_poses.append(pose)

        obj_poses = np.stack(obj_poses)
        poses.append(obj_poses)
        renderer.release_obj()
        used_objs.append(obj)

    poses = np.stack(poses)
    np.savez(
        f"{data_dir}/poses.npz",
        poses=poses,
        focal=focal,
        camera_distance=CAMERA_DISTANCE,
    )
    with open(f"{data_dir}/objs.txt", "w") as f:
        print("\n".join(used_objs), file=f)


if __name__ == "__main__":
    main()
