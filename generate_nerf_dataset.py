import numpy as np

from pyrr import Matrix44
from renderer import gen_rotation_matrix_from_azim_elev_in_plane, Renderer
from renderer_settings import *

SHAPENET_DIR = "/run/media/airalcorn2/MiQ BIG/ShapeNetCore.v2"


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
    img_size = 100
    # Calculate focal length in pixel units. This is just geometry. See:
    # https://en.wikipedia.org/wiki/Angle_of_view#Derivation_of_the_angle-of-view_formula.
    focal = (img_size / 2) / np.tan(np.radians(ANGLE_OF_VIEW) / 2)

    # Load the ShapeNet car object.
    obj = "66bdbc812bd0a196e194052f3f12cb2e"
    cat = "02958343"
    obj_mtl_path = f"{SHAPENET_DIR}/{cat}/{obj}/models/model_normalized"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")

    # Generate car renders using random camera locations.
    init_cam_pos = np.array([0, 0, CAMERA_DISTANCE])
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    samps = 800
    imgs = []
    poses = []
    for idx in range(samps):
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
        imgs.append(np.array(image))

        pose = np.eye(4)
        pose[:3, :3] = np.array(look_at[:3, :3])
        pose[:3, 3] = -look_at[:3, :3] @ look_at[3, :3]
        poses.append(pose)

    imgs = np.stack(imgs)
    poses = np.stack(poses)
    np.savez(
        f"{obj}.npz",
        images=imgs,
        poses=poses,
        focal=focal,
        camera_distance=CAMERA_DISTANCE,
    )


if __name__ == "__main__":
    main()
