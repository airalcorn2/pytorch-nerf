# Renderer settings.
WINDOW_SIZE = 256
IMG_SIZE = 128
CULL_FACES = True

CAMERA_DISTANCE = 6.8
TOO_CLOSE = -1
TOO_FAR = -10
MIN_Z = -5.8
MAX_Z = -3.8
DEFAULT_Z = (MAX_Z + MIN_Z) / 2

DEFAULT_ANGLE_OF_VIEW = 16.426
# See: https://en.wikipedia.org/wiki/Angle_of_view#Common_lens_angles_of_view.
MIN_AOV = DEFAULT_ANGLE_OF_VIEW
MAX_AOV = 60.0

# Lighting.
DIR_LIGHT = (0, 1 / (2 ** 0.5), 2 ** 0.5)
DIF_INT = 0.7
AMB_INT = 0.7
