import logging
import moderngl
import numpy as np

from PIL import Image, ImageOps
from pyrr import Matrix44
from scipy.spatial.transform import Rotation

YAW_PITCH_ROLL = {"yaw", "pitch", "roll"}
AZIM_ELEV_IN_PLANE = {"azimuth", "elevation", "in_plane"}


def gen_rotation_matrix_from_azim_elev_in_plane(
    azimuth=0.0, elevation=0.0, in_plane=0.0
):
    # See: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function.
    y = np.sin(elevation)
    radius = np.cos(elevation)
    x = radius * np.sin(azimuth)
    z = radius * np.cos(azimuth)

    cam_from = np.array([x, y, z])
    cam_to = np.zeros(3)
    tmp = np.array([0.0, 1.0, 0.0])

    diff = cam_from - cam_to
    forward = diff / np.linalg.norm(diff)
    crossed = np.cross(tmp, forward)
    right = crossed / np.linalg.norm(crossed)
    up = np.cross(forward, right)

    R = np.stack([right, up, forward])
    R_in_plane = Rotation.from_euler("Z", in_plane).as_matrix()
    return R_in_plane @ R


def parse_obj_file(input_obj):
    """Parse wavefront .obj file.

    :param input_obj:
    :return: Dictionary of NumPy arrays with shape (3 * num_faces, 8). Each row contains
    (1) the coordinates of a vertex of a face, (2) the vertex's normal vector, and (3)
    the texture coordinates for the vertex.
    """
    data = {"v": [], "vn": [], "vt": []}
    packed_arrays = {}
    obj_f = open(input_obj)
    current_mtl = None
    min_vec = np.full(3, np.inf)
    max_vec = np.full(3, -np.inf)
    empty_vt = np.array([0.0, 0.0, 0.0])
    for line in obj_f:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()
        elem_type = parts[0]
        if elem_type in data:
            vals = np.array(parts[1:4], dtype=np.float)
            if elem_type == "v":
                min_vec = np.minimum(min_vec, vals)
                max_vec = np.maximum(max_vec, vals)
            elif elem_type == "vn":
                vals /= np.linalg.norm(vals)
            elif elem_type == "vt":
                if len(vals) < 3:
                    vals = np.array(list(vals) + [0.0], dtype=np.float)

            data[elem_type].append(vals)
        elif elem_type == "f":
            f = parts[1:4]
            for fv in f:
                (v, vt, vn) = fv.split("/")

                # Convert to zero-based indexing.
                v = int(v) - 1
                vn = int(vn) - 1
                vt = int(vt) - 1 if vt else -1

                if vt == -1:
                    row = np.concatenate(
                        (data["v"][v], data["vn"][vn], empty_vt))
                else:
                    row = np.concatenate(
                        (data["v"][v], data["vn"][vn], data["vt"][vt]))

                packed_arrays[current_mtl].append(row)
        elif elem_type == "usemtl":
            current_mtl = parts[1]
            if current_mtl not in packed_arrays:
                packed_arrays[current_mtl] = []
        elif elem_type == "l":
            if current_mtl in packed_arrays:
                packed_arrays.pop(current_mtl)

    max_pos_vec = max_vec - min_vec
    max_pos_val = max(max_pos_vec)
    max_pos_vec_norm = max_pos_vec / max_pos_val
    for (sub_obj, packed_array) in packed_arrays.items():
        # z-coordinate of texture is always zero (if present).
        packed_array = np.stack(packed_array)[:, :8]
        original_vertices = packed_array[:, :3].copy()

        # All coordinates greater than or equal to zero.
        original_vertices -= min_vec
        # All coordinates between zero and one.
        original_vertices /= max_pos_val
        # All coordinates between zero and two.
        original_vertices *= 2
        # All coordinates between negative one and positive one with the center of object
        # at (0, 0, 0).
        original_vertices -= max_pos_vec_norm

        packed_array[:, :3] = original_vertices
        packed_arrays[sub_obj] = packed_array

    all_vertices = np.stack(data["v"])
    all_vertices -= min_vec
    all_vertices /= max_pos_val
    all_vertices *= 2
    all_vertices -= max_pos_vec_norm
    return (packed_arrays, all_vertices)


def parse_mtl_file(input_mtl):
    vector_elems = {"Ka", "Kd", "Ks"}
    float_elems = {"Ns", "Ni", "d"}
    int_elems = {"illum"}
    current_mtl = None
    mtl_infos = {}
    mtl_f = open(input_mtl)
    sub_objs = []
    for line in mtl_f:
        line = line.strip()
        if line == "":
            continue

        parts = line.split()
        elem_type = parts[0]
        if elem_type in vector_elems:
            vals = np.array(parts[1:4], dtype=np.float)
            mtl_infos[current_mtl][elem_type] = tuple(vals)
        elif elem_type in float_elems:
            mtl_infos[current_mtl][elem_type] = float(parts[1])
        elif elem_type in int_elems:
            mtl_infos[current_mtl][elem_type] = int(parts[1])
        elif elem_type == "newmtl":
            current_mtl = parts[1]
            sub_objs.append(current_mtl)
            mtl_infos[current_mtl] = {"d": 1.0}
        elif elem_type == "map_Kd":
            mtl_infos[current_mtl]["map_Kd"] = parts[1]

    sub_objs.sort()
    sub_objs.reverse()
    non_trans = [
        sub_obj for sub_obj in sub_objs if mtl_infos[sub_obj]["d"] == 1.0]
    trans = [
        (sub_obj, mtl_infos[sub_obj]["d"])
        for sub_obj in sub_objs
        if mtl_infos[sub_obj]["d"] < 1.0
    ]
    trans.sort(key=lambda x: x[1], reverse=True)
    sub_objs = non_trans + [sub_obj for (sub_obj, d) in trans]
    return (mtl_infos, sub_objs)


def get_texture_data(sub_objs, packed_arrays, mtl_infos, obj_f):
    texture_data = {}
    texture_path_list = obj_f.split("/")
    img_str_len = len("images/")
    for sub_obj in sub_objs:
        if sub_obj not in packed_arrays:
            continue

        if "map_Kd" in mtl_infos[sub_obj]:
            texture_f = mtl_infos[sub_obj]["map_Kd"]
            img_str_idx = texture_f.find("images/")
            if img_str_idx != -1:
                texture_path = "/".join(texture_path_list[:-2] + ["images"])
                texture_f = texture_f[img_str_idx + img_str_len:]
            else:
                texture_path = "/".join(texture_path_list[:-1])

            try:
                texture_img = (
                    Image.open(texture_path + "/" + texture_f)
                    .transpose(Image.FLIP_TOP_BOTTOM)
                    .convert("RGBA")
                )
            except FileNotFoundError:
                texture_f_parts = texture_f.split(".")
                ext = texture_f_parts[-1]
                if ext.isupper():
                    texture_f_parts[-1] = ext.lower()
                elif ext.islower():
                    texture_f_parts[-1] = ext.upper()

                texture_f = ".".join(texture_f_parts)
                texture_img = (
                    Image.open(texture_path + "/" + texture_f)
                    .transpose(Image.FLIP_TOP_BOTTOM)
                    .convert("RGBA")
                )

            texture_data[sub_obj] = {
                "size": texture_img.size,
                "bytes": texture_img.tobytes(),
            }

    return texture_data


class Renderer:
    def __init__(
        self,
        background_f=None,
        camera_distance=2.0,
        angle_of_view=16.426,
        dir_light=(0, 1 / np.sqrt(2), np.sqrt(2)),
        dif_int=0.7,
        amb_int=0.7,
        default_width=128,
        default_height=128,
        cull_faces=True,
    ):
        # Initialize OpenGL context.
        self.ctx = moderngl.create_standalone_context()
        # Render depth appropriately.
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Setting for rendering transparent objects.
        # See: https://learnopengl.com/Advanced-OpenGL/Blending
        # and: https://github.com/cprogrammer1994/ModernGL/blob/master/moderngl/context.py#L129.
        self.ctx.enable(moderngl.BLEND)

        # Define OpenGL program.
        prog = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform float x;
                uniform float y;
                uniform float z;

                uniform mat3 R_obj;
                uniform mat3 R_light;
                uniform vec3 DirLight;
                uniform mat4 VP;
                uniform int mode;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;

                out vec3 v_pos;
                out vec3 v_norm;
                out vec2 v_text;
                out vec3 v_light;

                void main() {
                    if (mode == 0) {
                        v_pos = R_obj * in_vert + vec3(x, y, z);
                        gl_Position = VP * vec4(v_pos, 1.0);
                        v_norm = R_obj * in_norm;
                        v_text = in_text;
                        v_light = R_light * DirLight;
                    } else {
                        gl_Position = vec4(in_vert, 1.0);
                        v_text = in_text;
                    }
                }
            """,
            fragment_shader="""
                #version 330

                uniform float amb_int;
                uniform float dif_int;
                uniform vec3 cam_pos;

                uniform sampler2D Texture;
                uniform int mode;
                uniform bool use_texture;
                uniform bool has_image;

                uniform vec3 box_rgb;

                uniform vec3 amb_rgb;
                uniform vec3 dif_rgb;
                uniform vec3 spc_rgb;
                uniform float spec_exp;
                uniform float trans;

                in vec3 v_pos;
                in vec3 v_norm;
                in vec2 v_text;
                in vec3 v_light;

                out vec4 f_color;

                void main() {
                    if (mode == 0) {
                        float dif = clamp(dot(v_light, v_norm), 0.0, 1.0) * dif_int;
                        if (use_texture) {
                            vec3 surface_rgb = dif_rgb;
                            vec3 diffuse = dif * surface_rgb;
                            if (has_image) {
                                surface_rgb = texture(Texture, v_text).rgb;
                                diffuse = dif * dif_rgb * surface_rgb;
                            }
                            vec3 ambient = amb_int * amb_rgb * surface_rgb;
                            float spec = 0.0;
                            if (dif > 0.0) {
                                vec3 reflected = reflect(-v_light, v_norm);
                                vec3 surface_to_camera = normalize(cam_pos - v_pos);
                                spec = pow(clamp(dot(surface_to_camera, reflected), 0.0, 1.0), spec_exp);
                            }
                            vec3 specular = spec * spc_rgb * surface_rgb;
                            vec3 linear = ambient + diffuse + specular;
                            f_color = vec4(linear, trans);
                        } else {
                            f_color = vec4(vec3(1.0, 1.0, 1.0) * dif + amb_int, 1.0);
                        }
                    } else if (mode == 1) {
                        f_color = vec4(texture(Texture, v_text).rgba);
                    } else {
                        f_color = vec4(box_rgb, 1.0);
                    }
                }
            """,
        )

        # Lighting uniform variables.
        prog["R_light"].write(np.eye(3).astype("f4").tobytes())
        dir_light = np.array(dir_light)
        prog["DirLight"].value = tuple(dir_light / np.linalg.norm(dir_light))
        prog["dif_int"].value = dif_int
        prog["amb_int"].value = amb_int
        prog["amb_rgb"].value = (1.0, 1.0, 1.0)
        prog["dif_rgb"].value = (1.0, 1.0, 1.0)
        prog["spc_rgb"].value = (1.0, 1.0, 1.0)
        prog["spec_exp"].value = 0.0
        self.use_spec = True

        # Mode uniform variables.
        prog["mode"].value = 0
        prog["use_texture"].value = True
        prog["has_image"].value = False

        # Model transformation uniform variables.
        prog["R_obj"].write(np.eye(3).astype("f4").tobytes())
        prog["x"].value = 0
        prog["y"].value = 0
        prog["z"].value = 0

        # Set up background.
        self.prog = prog
        (self.default_width, self.default_height) = (
            default_width, default_height)
        self.background = None
        (window_width, window_height) = self.set_up_background(background_f)

        # Look at origin matrix.
        eye = np.array([0.0, 0.0, camera_distance])
        prog["cam_pos"].value = tuple(eye)
        target = np.zeros(3)
        up = np.array([0.0, 1.0, 0.0])
        self.look_at = Matrix44.look_at(eye, target, up)

        # Perspective projection matrix.
        self.ratio = window_width / window_height
        self.angle_of_view = angle_of_view
        self.perspective = Matrix44.perspective_projection(
            angle_of_view, self.ratio, 0.1, 1000.0
        )

        # View-Projection uniform variable.
        self.prog["VP"].write(
            (self.look_at @ self.perspective).astype("f4").tobytes())

        # Set up object.
        self.mtl_infos = None
        self.cull_faces = cull_faces
        self.render_objs = []
        self.vbos = {}
        self.vaos = {}
        self.textures = {}

        # Initialize frame buffer.
        size = (window_width, window_height)
        self.window_size = size

        # Set up multisample anti-aliasing.
        self.ctx.multisample = True
        color_rbo = self.ctx.renderbuffer(size, samples=self.ctx.max_samples)
        depth_rbo = self.ctx.depth_renderbuffer(
            size, samples=self.ctx.max_samples)
        self.fbo = self.ctx.framebuffer(color_rbo, depth_rbo)

        color_rbo2 = self.ctx.renderbuffer(size)
        depth_rbo2 = self.ctx.depth_renderbuffer(size)
        self.fbo2 = self.ctx.framebuffer(color_rbo2, depth_rbo2)

        self.fbo.use()

    def set_up_obj(self, obj_f, mtl_f):
        (packed_arrays, vertices) = parse_obj_file(obj_f)
        packed_arrays = {
            sub_obj: packed_array.flatten().astype("f4").tobytes()
            for (sub_obj, packed_array) in packed_arrays.items()
        }
        (mtl_infos, sub_objs) = parse_mtl_file(mtl_f)
        texture_data = get_texture_data(
            sub_objs, packed_arrays, mtl_infos, obj_f)
        self.load_obj(packed_arrays, vertices,
                      mtl_infos, sub_objs, texture_data)

    def load_obj(self, packed_arrays, vertices, mtl_infos, sub_objs, texture_data):
        self.hom_vertices = np.hstack(
            [vertices, np.ones(len(vertices))[:, None]])
        render_objs = []
        vbos = {}
        vaos = {}
        textures = {}
        for sub_obj in sub_objs:
            if sub_obj not in packed_arrays:
                logging.info(f"Skipping {sub_obj}.")
                continue

            render_objs.append(sub_obj)
            packed_array = packed_arrays[sub_obj]
            vbo = self.ctx.buffer(packed_array)
            vbos[sub_obj] = vbo
            # Recall that "in_vert", "in_norm", and "in_text" are the inputs to the
            # vertex shader.
            vao = self.ctx.simple_vertex_array(
                self.prog, vbo, "in_vert", "in_norm", "in_text"
            )
            vaos[sub_obj] = vao

            if "map_Kd" in mtl_infos[sub_obj]:
                # Initialize texture from image.
                texture = self.ctx.texture(
                    texture_data[sub_obj]["size"], 4, texture_data[sub_obj]["bytes"]
                )
                texture.build_mipmaps()
                textures[sub_obj] = texture

        self.mtl_infos = mtl_infos
        self.render_objs = render_objs
        self.vbos = vbos
        self.vaos = vaos
        self.textures = textures

    def set_up_background(self, background_f=None):
        if background_f:
            background_img = (
                Image.open(background_f)
                .transpose(Image.FLIP_TOP_BOTTOM)
                .convert("RGBA")
            )

            # Initialize background from image.
            background = self.ctx.texture(
                background_img.size, 4, background_img.tobytes()
            )
            background.build_mipmaps()
            self.background = background

            # Create a square plane from two triangles (two sets of three points).
            vertices = np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            )
            # These arrays are not used by the renderer, but the vertex shader expects
            # them as input.
            normals = np.repeat([[0.0, 0.0, 1.0]], len(vertices), axis=0)
            # The texture (UV) coordinates corresponding to the above triangle points.
            texture_coords = np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [
                    0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
            )

            background_array = np.hstack((vertices, normals, texture_coords))
            self.background_vbo = self.ctx.buffer(
                background_array.flatten().astype("f4").tobytes()
            )
            self.background_vao = self.ctx.simple_vertex_array(
                self.prog, self.background_vbo, "in_vert", "in_norm", "in_text"
            )

            return (background_img.width, background_img.height)
        else:
            return (self.default_width, self.default_height)

    def render(self, r=0.485, g=0.456, b=0.406, with_alpha=False):
        if self.background is not None:
            # See: https://computergraphics.stackexchange.com/a/4007.
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog["mode"].value = 1
            self.background.use()
            self.fbo.clear()
            self.background_vao.render()

            self.ctx.enable(moderngl.DEPTH_TEST)
            self.prog["mode"].value = 0
        else:
            self.fbo.clear(r, g, b)

        if self.cull_faces:
            self.ctx.enable(moderngl.CULL_FACE)

        for render_obj in self.render_objs:
            if self.prog["use_texture"].value:
                self.prog["amb_rgb"].value = self.mtl_infos[render_obj]["Ka"]
                self.prog["dif_rgb"].value = self.mtl_infos[render_obj]["Kd"]
                if self.use_spec:
                    self.prog["spc_rgb"].value = self.mtl_infos[render_obj]["Ks"]
                    self.prog["spec_exp"].value = self.mtl_infos[render_obj]["Ns"]
                else:
                    self.prog["spc_rgb"].value = (0.0, 0.0, 0.0)

                self.prog["trans"].value = self.mtl_infos[render_obj]["d"]
                if render_obj in self.textures:
                    self.prog["has_image"].value = True
                    self.textures[render_obj].use()

            self.vaos[render_obj].render()
            self.prog["has_image"].value = False

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.copy_framebuffer(self.fbo2, self.fbo)
        if with_alpha:
            return Image.frombytes(
                "RGBA",
                self.fbo.size,
                self.fbo2.read(components=4),
                "raw",
                "RGBA",
                0,
                -1,
            )
        else:
            return Image.frombytes(
                "RGB", self.fbo.size, self.fbo2.read(), "raw", "RGB", 0, -1
            )

    def get_vertex_screen_coordinates(self):
        world = np.eye(4)
        world[:3, :3] = np.array(self.prog["R_obj"].value).reshape((3, 3)).T
        world[:3, 3] = (
            self.prog["x"].value,
            self.prog["y"].value,
            self.prog["z"].value,
        )
        PV = np.array(self.prog["VP"].value).reshape((4, 4)).T
        pre_screen_coords = PV @ world @ self.hom_vertices.T

        (window_width, window_height) = self.window_size
        screen_xs = (
            window_width
            * (np.array(pre_screen_coords[0]) / np.array(pre_screen_coords[3]) + 1)
            / 2
        )
        screen_ys = (
            window_height
            * (np.array(pre_screen_coords[1]) / np.array(pre_screen_coords[3]) + 1)
            / 2
        )
        screen_coords = np.hstack((screen_xs, screen_ys))

        screen = np.zeros((window_height, window_width))
        for i in range(len(screen_xs)):
            col = x = int(screen_xs[i])
            row = y = int(screen_ys[i])
            if x < window_width and y < window_height:
                screen[window_height - row - 1, col] = 1

        screen_mat = np.uint8(255 * screen)
        screen_img = Image.fromarray(screen_mat, mode="L")
        return (screen_coords, screen_img)

    def __del__(self):
        self.release()

    def release_obj(self):
        for sub_obj in self.vbos:
            self.vbos[sub_obj].release()
            self.vaos[sub_obj].release()
            if sub_obj in self.textures:
                self.textures[sub_obj].release()

        self.vbos = {}
        self.vaos = {}
        self.textures = {}

    def release_background(self):
        if self.background is not None:
            self.background.release()
            self.background_vbo.release()
            self.background_vao.release()
            self.background = None

    def release(self):
        self.release_obj()
        self.release_background()

        self.fbo.release()
        self.fbo2.release()
        self.ctx.release()

    def adjust_angle_of_view(self, angle_of_view):
        self.angle_of_view = angle_of_view
        perspective = Matrix44.perspective_projection(
            self.angle_of_view, self.ratio, 0.1, 1000.0
        )
        self.prog["VP"].write(
            (perspective * self.look_at).astype("f4").tobytes())

    def set_params(self, params):
        ypr_params = {}
        ae_params = {}
        for (param, value) in params.items():
            if param in self.prog:
                self.prog[param].value = value
            elif param == "aov":
                self.adjust_angle_of_view(value)
            elif param in YAW_PITCH_ROLL:
                ypr_params[param] = value
            elif param in AZIM_ELEV_IN_PLANE:
                ae_params[param] = value

        if len(ypr_params) > 0:
            yaw = ypr_params.get("yaw", 0)
            pitch = ypr_params.get("pitch", 0)
            roll = ypr_params.get("roll", 0)
            R_obj = Rotation.from_euler("YXZ", [yaw, pitch, roll]).as_matrix()
            self.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
        elif len(ae_params) > 0:
            R_obj = gen_rotation_matrix_from_azim_elev_in_plane(**ae_params)
            self.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())

    def get_depth_arrays(self):
        depth = np.frombuffer(
            self.fbo2.read(attachment=-1, dtype="f4"), dtype=np.dtype("f4")
        )
        depth = 1 - depth.reshape(self.window_size)
        min_pos = depth[depth > 0].min()
        depth[depth > 0] = depth[depth > 0] - min_pos
        depth_normed = depth / depth.max()
        return (depth, depth_normed)

    def get_depth_map(self):
        (depth, depth_normed) = self.get_depth_arrays()
        depth_map = np.uint8(255 * depth_normed)
        return ImageOps.flip(Image.fromarray(depth_map, "L"))

    def get_normal_map(self):
        # See: https://stackoverflow.com/questions/5281261/generating-a-normal-map-from-a-height-map
        # and: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc
        # and: https://en.wikipedia.org/wiki/Normal_mapping#How_it_works.
        (depth, depth_normed) = self.get_depth_arrays()
        depth_pad = np.pad(depth_normed, 1, "constant")
        (dx, dy) = (1 / depth.shape[1], 1 / depth.shape[0])
        dz_dx = (depth_pad[1:-1, 2:] - depth_pad[1:-1, :-2]) / (2 * dx)
        dz_dy = (depth_pad[2:, 1:-1] - depth_pad[:-2, 1:-1]) / (2 * dy)
        norms = np.stack(
            [-dz_dx.flatten(), -dz_dy.flatten(), np.ones(dz_dx.size)])
        magnitudes = np.linalg.norm(norms, axis=0)
        norms /= magnitudes
        norms = norms.T
        norms[:, :2] = 255 * (norms[:, :2] + 1) / 2
        norms[:, 2] = 127 * norms[:, 2] + 128
        norms = np.uint8(norms).reshape((*depth.shape, 3))
        return ImageOps.flip(Image.fromarray(norms))
