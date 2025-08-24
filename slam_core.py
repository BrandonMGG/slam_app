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
        # Si no hay keyframes, trayectoria vacía
        if not self.relative_transformations:
            return np.zeros((1, 2), dtype=np.float32)

        # Integrar desplazamientos relativos (simple “odometría” acumulada)
        positions = [np.array([0.0, 0.0, 0.0])]
        for T in self.relative_transformations:
            dx, dy, dz = T[:3, 3]
            last = positions[-1]
            positions.append(last + np.array([dx, dy, dz]))

        positions = np.vstack(positions)

        # Proyectar a plano XZ
        traj_xz = positions[:, [0, 2]]

        # Suavizado básico con ventana 5
        if len(traj_xz) >= 5:
            k = 5
            kernel = np.ones(k) / k
            x = np.convolve(traj_xz[:, 0], kernel, mode='same')
            z = np.convolve(traj_xz[:, 1], kernel, mode='same')
            traj_xz = np.column_stack([x, z]).astype(np.float32)

        return traj_xz


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

        # --- Render PNG con OpenCV ---
        h, w = 720, 1280
        margin = 40
        img = np.full((h, w, 3), 255, np.uint8)

        if len(trajectory) >= 2:
            # Normalizar a canvas
            mins = trajectory.min(axis=0)
            maxs = trajectory.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            # Dejar margen y mantener aspecto
            scale = 0.9 * min((w - 2*margin) / span[0], (h - 2*margin) / span[1])

            pts = ( (trajectory - mins) * scale )
            # Invertir eje Z->Y para pantalla y centrar con margen
            pts_img = np.zeros_like(pts)
            pts_img[:, 0] = margin + pts[:, 0]
            pts_img[:, 1] = h - margin - pts[:, 1]

            pts_img = pts_img.astype(np.int32).reshape(-1, 1, 2)

            # Polilínea
            cv2.polylines(img, [pts_img], False, (200, 0, 0), 2)

            # Start/End
            cv2.circle(img, tuple(pts_img[0, 0]), 6, (0, 180, 0), -1)
            cv2.circle(img, tuple(pts_img[-1, 0]), 6, (0, 0, 200), -1)

        # Texto de métricas
        y0 = 30
        dy = 28
        info = [
            f"Keyframes: {num_keyframes}",
            f"Prom. matches/pose: {avg_matches:.1f}",
            f"Exito triangulacion: {triangulation_success_rate:.2%}",
            f"Mov. medio entre keyframes: {avg_translation:.2f} m",
            f"Video: {os.path.basename(input_video_path)}",
        ]
        for i, line in enumerate(info):
            cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)

        cv2.imwrite(output_base + ".png", img)


    def process_video_input(self, video_path):
        video_capture = cv2.VideoCapture(video_path)

        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            self.process_frame(frame)

        video_capture.release()

        optimized_trajectory = self.optimize_pose_graph()
        self.save_trajectory_outputs(optimized_trajectory, video_path)

