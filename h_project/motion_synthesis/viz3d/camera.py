import pyrr
import numpy as np
from . import default

class Camera:

    def __init__(self,
                 view_width,
                 view_height,
                 fov_degrees=default.CAMERA_FOV_DEGREES,
                 near_field=default.CAMERA_NEAR_FIELD,
                 far_field=default.CAMERA_FAR_FIELD):

        self.position = default.CAMERA_POSITION
        self.up_vector = default.CAMERA_UP
        self.world_up_vector = default.CAMERA_WORLD_UP
        self.front_vector = default.CAMERA_FRONT
        self.right_vector = np.zeros((3, ), dtype=np.float32)
        self.right_vector = np.zeros((3, ), dtype=np.float32)
        self.yaw_radians = default.CAMERA_YAW_RADIANS
        self.pitch_radians = default.CAMERA_PITCH_RADIANS
        self.zoom = default.CAMERA_ZOOM
        self.movement_speed = default.CAMERA_MOVEMENT_SPEED
        self.mouse_sensitivity = default.CAMERA_MOUSE_SENSITIVITY

        # Matrices
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.view_projection_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(fovy=fov_degrees,
                                                                                    aspect=view_width/view_height,
                                                                                    near=near_field,
                                                                                    far=far_field)

        # Command states
        self.move_forward = False
        self.move_back = False
        self.move_up = False
        self.move_down = False
        self.move_left = False
        self.move_right = False

        self._update_right_up_vectors()

    def update(self, elapsed_time):

        delta = elapsed_time * self.movement_speed

        if self.move_left:
            self.position -= self.right_vector * delta
        if self.move_right:
            self.position += self.right_vector * delta
        if self.move_up:
            self.position += default.CAMERA_UP * delta
        if self.move_down:
            self.position -= default.CAMERA_UP * delta
        if self.move_forward:
            self.position += self.front_vector * delta
        if self.move_back:
            self.position -= self.front_vector * delta

        #self.look_at(self.position + self.front_vector)
        #view_matrix = np.eye(4, dtype=np.float32)
        #view_matrix[:3, 0] = self.right_vector
        #view_matrix[:3, 1] = self.up_vector
        #view_matrix[:3, 2] = self.front_vector
        #view_matrix[:3, 3] = self.position
        #self.view_projection_matrix = np.linalg.inv(self.view_matrix) @ self.projection_matrix

        self.view_matrix = pyrr.matrix44.create_look_at(eye=self.position,
                                                        target=self.position + self.front_vector,
                                                        up=default.CAMERA_UP)

        self.view_projection_matrix = self.view_matrix @ self.projection_matrix

    def process_mouse_movement(self, delta_x, delta_y):

        self.yaw_radians += delta_x * self.mouse_sensitivity
        self.pitch_radians += delta_y * self.mouse_sensitivity
        self.pitch_radians = np.clip(self.pitch_radians,
                                     -default.CAMERA_MAX_PITCH_RADIANS,
                                     default.CAMERA_MAX_PITCH_RADIANS)

        #print(self.pitch_radians, self.yaw_radians)

        #rot_x = pyrr.matrix44.create_from_x_rotation(delta_y * self.mouse_sensitivity*0.1)
        #rot_y = pyrr.matrix44.create_from_y_rotation(-delta_x * self.mouse_sensitivity*0.1)
        #self.front_vector = np.matmul((rot_x @ rot_y)[0:3, 0:3], self.front_vector)


        self.front_vector[0] = np.cos(self.yaw_radians) * np.cos(self.pitch_radians)
        self.front_vector[1] = np.sin(self.pitch_radians)
        self.front_vector[2] = np.sin(self.yaw_radians) * np.cos(self.pitch_radians)
        self.front_vector /= np.linalg.norm(self.front_vector)
        self._update_right_up_vectors()

    def look_at(self, target):

        self.front_vector = target - self.position
        self.front_vector /= np.linalg.norm(self.front_vector)
        self._update_right_up_vectors()

    def _update_right_up_vectors(self):

        self.right_vector = np.cross(self.front_vector, self.world_up_vector)
        self.right_vector /= np.linalg.norm(self.right_vector)
        self.up_vector = np.cross(self.right_vector, self.front_vector)
        self.up_vector /= np.linalg.norm(self.up_vector)

