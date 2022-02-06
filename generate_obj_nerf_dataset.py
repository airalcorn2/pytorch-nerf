import numpy as np

from renderer import Renderer
from renderer_settings import *
from rotation_utils import gen_rotation_matrix


def main():
    # Set up the renderer.
    camera_distance = 2.25
    aov = 53.962828459664856
    renderer = Renderer(
        camera_distance=camera_distance,
        angle_of_view=aov,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    img_size = 100
    # Calculate focal length in pixel units. This is just geometry. See:
    # https://en.wikipedia.org/wiki/Angle_of_view#Derivation_of_the_angle-of-view_formula.
    focal = (img_size / 2) / np.tan(np.radians(aov) / 2)

    # Load the ShapeNet car object.
    SHAPENET_DIR = "/run/media/airalcorn2/MiQ BIG/ShapeNetCore.v2"
    obj = "66bdbc812bd0a196e194052f3f12cb2e"
    cat = "02958343"
    obj_mtl_path = f"{SHAPENET_DIR}/{cat}/{obj}/models/model_normalized"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")

    # Generate car renders using random camera locations.
    init_norm_dir = np.array([0, 0, 1])
    samps = 800
    imgs = []
    poses = []
    for idx in range(samps):
        angles = {
            "yaw": np.random.uniform(-np.pi, np.pi),
            "pitch": np.random.uniform(-np.pi, np.pi),
            "roll": np.random.uniform(-np.pi, np.pi),
        }
        renderer.set_params(angles)
        R = gen_rotation_matrix(**angles)
        image = renderer.render(0.5, 0.5, 0.5).resize((img_size, img_size))
        imgs.append(np.array(image))

        pose = np.zeros((3, 4))
        pose[:3, :3] = R.T
        pose[:3, 3] = R @ init_norm_dir
        poses.append(pose)

    imgs = np.stack(imgs)
    poses = np.stack(poses)
    np.savez(
        f"{obj}_obj.npz",
        images=imgs,
        poses=poses,
        focal=focal,
        camera_distance=camera_distance,
    )


if __name__ == "__main__":
    main()
