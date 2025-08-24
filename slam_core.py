import csv
from datetime import datetime
import sys
import os
import cv2
import numpy as np


from pathlib import Path


class PoseGraphSLAM:
    def __init__(self, fx=700, fy=700, cx=320, cy=240):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.name = Path(__file__).resolve().parent.name

        self.orb_detector = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        self.keyframe_poses = []
        self.relative_transformations = []

        self.previous_keyframe_image = None
        self.previous_keyframe_keypoints = None
        self.previous_keyframe_descriptors = None
        self.previous_keyframe_pose = np.eye(4)

        self.frame_counter = 0
        self.min_frame_gap = 5
        self.min_keyframe_translation = 0.05

        # Métricas para evaluación
        self.total_successful_frames = 0
        self.total_tracked_matches = 0
        self.total_translation_magnitude = 0.0
        self.total_pose_estimations = 0

    def filter_matches_lowe_ratio(self, descriptors1, descriptors2, ratio=0.75):
        knn_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches

    def process_frame(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb_detector.detectAndCompute(grayscale_frame, None)

        if self.previous_keyframe_descriptors is not None and descriptors is not None and len(keypoints) > 0:
            matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, descriptors)

            if len(matches) > 30:
                points_prev = np.float32([self.previous_keyframe_keypoints[m.queryIdx].pt for m in matches])
                points_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

                essential_matrix, mask = cv2.findEssentialMat(points_prev, points_curr, self.camera_matrix, method=cv2.RANSAC, threshold=1.0)

                if essential_matrix is not None:
                    _, rotation, translation, _ = cv2.recoverPose(essential_matrix, points_prev, points_curr, self.camera_matrix)

                    relative_pose = np.eye(4)
                    relative_pose[:3, :3] = rotation
                    relative_pose[:3, 3] = translation.ravel()

                    current_pose = self.previous_keyframe_pose @ relative_pose
                    translation_magnitude = np.linalg.norm(relative_pose[:3, 3])

                    if self.frame_counter >= self.min_frame_gap or translation_magnitude > self.min_keyframe_translation:
                        self.keyframe_poses.append(current_pose.copy())
                        self.relative_transformations.append(relative_pose.copy())
                        self.previous_keyframe_keypoints = keypoints
                        self.previous_keyframe_descriptors = descriptors
                        self.previous_keyframe_pose = current_pose
                        self.frame_counter = 0
                    else:
                        self.frame_counter += 1

                    # Actualizar métricas
                    self.total_successful_frames += 1
                    self.total_tracked_matches += len(matches)
                    self.total_translation_magnitude += translation_magnitude
                    self.total_pose_estimations += 1
        else:
            self.keyframe_poses.append(np.eye(4))
            self.previous_keyframe_keypoints = keypoints
            self.previous_keyframe_descriptors = descriptors
            self.previous_keyframe_pose = np.eye(4)

    def optimize_pose_graph(self):
        
        if not self.relative_transformations:
            return np.zeros((1, 2), dtype=np.float32)

        
        positions = [np.array([0.0, 0.0, 0.0])]
        for T in self.relative_transformations:
            dx, dy, dz = T[:3, 3]
            last = positions[-1]
            positions.append(last + np.array([dx, dy, dz]))

        positions = np.vstack(positions)

        # Proyectar a plano XZ
        traj_xz = positions[:, [0, 2]]

        # Suavizado básico con ventana 5
        # if len(traj_xz) >= 5:
        #     k = 5
        #     kernel = np.ones(k) / k
        #     x = np.convolve(traj_xz[:, 0], kernel, mode='same')
        #     z = np.convolve(traj_xz[:, 1], kernel, mode='same')
        #     traj_xz = np.column_stack([x, z]).astype(np.float32)

        return traj_xz

    
    def _normalize_traj_for_canvas(self, traj_xy, W, H, margin=60, y_up=True, center=True):
        import numpy as np
        if traj_xy is None or len(traj_xy) == 0:
            return None
        pts = np.asarray(traj_xy, dtype=float).copy()
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        scale = 0.9 * min((W - 2*margin) / span[0], (H - 2*margin) / span[1])
        if center:
            center_world = (mins + maxs) / 2.0
            pts -= center_world
            cx, cy = W / 2.0, H / 2.0
            xs = cx + scale * pts[:, 0]
            ys = cy + (-scale * pts[:, 1] if y_up else scale * pts[:, 1])
        else:
            pts -= mins
            xs = margin + scale * pts[:, 0]
            ys = (H - margin - scale * pts[:, 1]) if y_up else (margin + scale * pts[:, 1])
        return np.stack([xs, ys], axis=1).astype(np.int32)

    
    def _save_plot_cv_matplotlibish(self, traj_xy, out_png, bg=(255,255,255), info_lines=None):
        import cv2, numpy as np, os
        H, W = 720, 1280
        img = np.full((H, W, 3), bg, np.uint8)

        # grid suave
        for x in range(0, W, 100):
            cv2.line(img, (x, 0), (x, H), (230, 230, 230), 1)
        for y in range(0, H, 100):
            cv2.line(img, (0, y), (W, y), (230, 230, 230), 1)

        if traj_xy is not None and len(traj_xy) >= 2:
            pts_img = self._normalize_traj_for_canvas(traj_xy, W, H, margin=60, y_up=True, center=True)
            cv2.polylines(img, [pts_img.reshape(-1,1,2)], False, (50, 50, 200), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts_img[0]), 6, (0, 180, 0), -1)
            cv2.circle(img, tuple(pts_img[-1]), 6, (0, 0, 200), -1)

            # barra de escala (1m/5m)
            mins = np.min(traj_xy, axis=0); maxs = np.max(traj_xy, axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            scale = 0.9 * min((W - 120) / span[0], (H - 120) / span[1])
            pix_per_meter = scale
            meters = 1 if pix_per_meter >= 80 else 5
            bar = int(round(pix_per_meter * meters))
            x0, y0 = W - 180, H - 80
            cv2.line(img, (x0, y0), (x0 + bar, y0), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"{meters} m", (x0 + bar + 10, y0 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # ejes
        cv2.arrowedLine(img, (80, H-80), (200, H-80), (0,0,0), 2, tipLength=0.03)  # +X
        cv2.arrowedLine(img, (80, H-80), (80, H-200), (0,0,0), 2, tipLength=0.03)  # +Z (Y up)
        cv2.putText(img, "X (m)", (205, H-75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, "Z (m)", (60, H-205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        
        if info_lines:
            y0, dy = 30, 28
            for i, line in enumerate(info_lines):
                cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, img)

    
    def save_trajectory_outputs(self, trajectory, input_video_path):
        tipo_lms = self.name
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", tipo_lms, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{self.name}")

        # CSV
        with open(output_base + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        # Métricas
        num_keyframes = len(self.keyframe_poses)
        avg_translation = self.total_translation_magnitude / max(1, self.total_pose_estimations)
        avg_matches = self.total_tracked_matches / max(1, self.total_pose_estimations)
        triangulation_success_rate = self.total_successful_frames / max(1, self.total_pose_estimations)

        info = [
            f"Keyframes: {num_keyframes}",
            f"Prom. matches/pose: {avg_matches:.1f}",
            f"Exito triangulacion: {triangulation_success_rate:.2%}",
            f"Mov. medio entre keyframes: {avg_translation:.2f} m",
            f"Video: {os.path.basename(input_video_path)}",
        ]

        
        self._save_plot_cv_matplotlibish(trajectory, output_base + ".png", info_lines=info)


    def process_video_input(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            self.process_frame(frame)
        video_capture.release()

        traj_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses], dtype=float)
        self.save_trajectory_outputs(traj_2d, video_path)

