import argparse
import math
import os
import csv
import json
import subprocess
from typing import Dict, List, Optional, Tuple
import cv2
from tqdm import tqdm

import numpy as np

from landmark import LandmarkExtractor
from insightface.app import FaceAnalysis


class FaceTrack:
    """Simple face track that stores a sequence of continuously matched keypoints."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.landmarks_history: List[np.ndarray] = []  # Each item is (106, 2)
        self.last_centroid: Optional[np.ndarray] = None  # (2,)
        self.last_embedding: Optional[np.ndarray] = None  # (d,)
        # New: stable square bbox (x1, y1, x2, y2)
        self.stable_bbox: Optional[Tuple[float, float, float, float]] = None

    def add(self, landmarks: np.ndarray, embedding: Optional[np.ndarray]) -> None:
        self.landmarks_history.append(landmarks)
        self.last_centroid = landmarks.mean(axis=0)
        if embedding is not None:
            self.last_embedding = embedding


def make_square_bbox(x1: float, y1: float, x2: float, y2: float, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
    """
    Convert rectangular bbox to square bbox while maintaining face aspect ratio.
    Reference implementation from get_face_bbox.py
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    square_size = max(width, height)
    half_size = square_size / 2
    new_x1 = center_x - half_size
    new_y1 = center_y - half_size
    new_x2 = center_x + half_size
    new_y2 = center_y + half_size
    
    # Handle boundary cases
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = square_size
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = square_size
    if new_x2 > image_width:
        new_x2 = image_width
        new_x1 = image_width - square_size
    if new_y2 > image_height:
        new_y2 = image_height
        new_y1 = image_height - square_size
    
    # Ensure coordinates are within valid range
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)
    
    return new_x1, new_y1, new_x2, new_y2


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("inf")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("inf")
    cos_sim = float(np.dot(a, b) / denom)
    # Convert similarity to "distance"
    return 1.0 - cos_sim


def match_faces(prev_tracks: Dict[int, FaceTrack], current_faces: List[Tuple[np.ndarray, Optional[np.ndarray]]], distance_threshold: float, embed_threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match faces using centroid nearest neighbor.
    Returns:
    - matches: list of (track_id, face_idx)
    - unmatched_tracks: list of unmatched track_ids
    - unmatched_faces: list of unmatched current frame face indices
    """
    if len(prev_tracks) == 0 or len(current_faces) == 0:
        return [], list(prev_tracks.keys()), list(range(len(current_faces)))

    # Precompute centroids
    current_centroids = [lm.mean(axis=0) for lm, _ in current_faces]

    # Compute distance matrix: tracks x faces
    track_ids = list(prev_tracks.keys())
    distances = np.full((len(track_ids), len(current_faces)), np.inf, dtype=np.float32)
    embed_distances = np.full((len(track_ids), len(current_faces)), np.inf, dtype=np.float32)
    for ti, tid in enumerate(track_ids):
        tcent = prev_tracks[tid].last_centroid
        temb = prev_tracks[tid].last_embedding
        if tcent is None:
            continue
        for fi, (fcent, (_, emb)) in enumerate(zip(current_centroids, current_faces)):
            d = float(np.linalg.norm(tcent - fcent))
            distances[ti, fi] = d
            if temb is not None and emb is not None:
                embed_distances[ti, fi] = float(cosine_distance(temb, emb))

    # Greedy matching: select by distance from small to large, skip if exceeds threshold
    pairs: List[Tuple[int, int]] = []
    used_tracks = set()
    used_faces = set()

    # Sort all candidates by distance
    candidates: List[Tuple[float, int, int]] = []
    # First filter strong matches by embedding distance, then fall back to spatial distance
    for ti in range(len(track_ids)):
        for fi in range(len(current_faces)):
            ed = embed_distances[ti, fi]
            sd = distances[ti, fi]
            if math.isfinite(ed) and ed <= embed_threshold:
                # Strong match candidate, highest priority
                candidates.append((0.0 + ed, ti, fi))
            elif math.isfinite(sd):
                candidates.append((1e3 + sd, ti, fi))  # Secondary priority: spatial distance
    candidates.sort(key=lambda x: x[0])

    for d, ti, fi in candidates:
        # For spatial distance candidates, filter by threshold; embedding candidates already passed threshold
        if d >= 1e3 and (d - 1e3) > distance_threshold:
            continue
        if ti in used_tracks or fi in used_faces:
            continue
        used_tracks.add(ti)
        used_faces.add(fi)
        pairs.append((track_ids[ti], fi))

    unmatched_tracks = [track_ids[ti] for ti in range(len(track_ids)) if ti not in used_tracks]
    unmatched_faces = [fi for fi in range(len(current_faces)) if fi not in used_faces]
    return pairs, unmatched_tracks, unmatched_faces


def compute_video_average_motion_distance(video_path: str, ctx_id: int = -1, distance_threshold: float = 60.0, embed_threshold: float = 0.3, extractor: Optional[LandmarkExtractor] = None, show_progress: bool = True, speaker_intervals: Optional[Dict[str, List[List[float]]]] = None, save_debug_videos: bool = False, output_dir: Optional[str] = None, target_size: int = 224, face_analysis: Optional[FaceAnalysis] = None) -> Tuple[List[Tuple[str, float]], float]:
    """
    Calculate the overall average of adjacent frame offset distances for face keypoints across the entire video:
    - For each stable track, compute the mean of keypoint displacement magnitudes frame by frame (average of 106 points);
    - Take the overall average of these means across all tracks and all frames;
    - If track breaks or face count inconsistency causes strict checks to fail, return -1.0.
    
    New features:
    - Use insightface to detect faces and generate stable square bbox as reference space
    - Calculate keypoint displacement within the reference space
    - Optionally save cropped face videos and keypoint motion videos
    """
    # Reuse externally provided model, create if not provided
    if extractor is None:
        extractor = LandmarkExtractor(ctx_id=ctx_id)
    
    # Reuse externally provided face detector, create if not provided
    if face_analysis is None:
        face_analysis = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        face_analysis.prepare(ctx_id=ctx_id, det_thresh=0.2, det_size=(224, 224))

    # ===== First pass: Find the maximum bbox size for each position (left/right) across the entire video =====
    print(f"First pass: Detecting all face bboxes to determine maximum size...")
    max_bbox_sizes = {"left": 0, "right": 0}  # Record maximum bbox side length for each position
    
    cap_scan = cv2.VideoCapture(video_path)
    frame_idx_scan = 0
    while True:
        ret, frame = cap_scan.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = face_analysis.get(frame)
        
        # Filter: keep only the two largest faces
        if len(detected_faces) > 2:
            faces_with_area = []
            for face in detected_faces:
                if isinstance(face, dict) and 'bbox' in face:
                    bbox = face['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    faces_with_area.append((face, area))
            faces_with_area.sort(key=lambda x: x[1], reverse=True)
            detected_faces = [face for face, _ in faces_with_area[:2]]
        
        # Sort by x coordinate, assign left and right
        if len(detected_faces) >= 2:
            face_items_sorted = []
            for face in detected_faces:
                if isinstance(face, dict) and 'bbox' in face:
                    bbox = face['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    face_items_sorted.append((x_center, bbox))
            face_items_sorted.sort(key=lambda x: x[0])
            
            for idx, (x_center, bbox) in enumerate(face_items_sorted[:2]):
                position = "left" if idx == 0 else "right"
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                
                # Convert to square bbox
                frame_height, frame_width = frame.shape[:2]
                square_bbox = make_square_bbox(x1, y1, x2, y2, frame_width, frame_height)
                sx1, sy1, sx2, sy2 = square_bbox
                bbox_size = int(sx2 - sx1)
                
                # Update maximum size
                if bbox_size > max_bbox_sizes[position]:
                    max_bbox_sizes[position] = bbox_size
        
        frame_idx_scan += 1
    
    cap_scan.release()
    
    # Take the maximum of both positions as unified crop_size (maximum size from first pass)
    max_crop_size = max(max_bbox_sizes["left"], max_bbox_sizes["right"], target_size)
    print(f"First pass completed: left_max={max_bbox_sizes['left']}, right_max={max_bbox_sizes['right']}, unified crop_size={max_crop_size}")
    
    # ===== Second pass: Process all frames using unified max_crop_size =====
    print(f"Second pass: Cropping and calculating motion with crop_size={max_crop_size}...")
    
    next_track_id = 0
    active_tracks: Dict[int, FaceTrack] = {}

    # Strict consistency check: fixed face count and stable track matching set
    expected_face_count: Optional[int] = None
    prev_matched_track_ids: Optional[set] = None
    failure = False

    # Motion distance accumulation (overall and per-track) - weighted average
    total_motion_sum: float = 0.0
    total_weight_sum: float = 0.0
    track_motion_sum: Dict[int, float] = {}
    track_weight_sum: Dict[int, float] = {}

    # Progress: read total frame count
    total_frames = 0
    fps = 0.0
    try:
        cap_tmp = cv2.VideoCapture(video_path)
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap_tmp.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        try:
            cap_tmp.release()
        except Exception:
            pass

    pbar = None
    if show_progress and total_frames > 0:
        pbar = tqdm(total=total_frames, desc=os.path.basename(video_path), unit="frame")

    # Speaking intervals -> silent frame determination
    left_speaks: List[Tuple[float, float]] = []
    right_speaks: List[Tuple[float, float]] = []
    if speaker_intervals is not None and fps > 0:
        # Data format: {"left": [[s, e], ...], "right": [[s, e], ...]}
        for k in ("left", "right"):
            if k not in speaker_intervals:
                speaker_intervals[k] = []
        left_speaks = [(float(s), float(e)) for s, e in speaker_intervals.get("left", [])]
        right_speaks = [(float(s), float(e)) for s, e in speaker_intervals.get("right", [])]

        def is_speaking_at(side: str, t: float) -> bool:
            segs = left_speaks if side == "left" else right_speaks
            for s, e in segs:
                if s <= t <= e:
                    return True
            return False
    else:
        def is_speaking_at(side: str, t: float) -> bool:
            return False

    # Focus only on eye and eyebrow keypoints
    eye_brow_indices = [
        43, 48, 49, 51, 50,        # Left eyebrow 5 points
        102, 103, 104, 105, 101,   # Right eyebrow 5 points
        35, 41, 42, 39, 37, 36,    # Left eye 6 points
        89, 95, 96, 93, 91, 90     # Right eye 6 points
    ]

    # Video writer (if saving debug videos)
    # Use fixed "left" and "right" identifiers instead of track IDs
    cropped_frames_buffer: Dict[str, List[np.ndarray]] = {}
    motion_frames_buffer: Dict[str, List[np.ndarray]] = {}
    
    # Store bbox center point and previous frame keypoints for each position
    bbox_centers: Dict[str, Optional[Tuple[float, float]]] = {"left": None, "right": None}
    prev_landmarks: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
    
    # Store stable values for each position (for outlier detection)
    stable_bbox_centers: Dict[str, Optional[Tuple[float, float]]] = {"left": None, "right": None}
    stable_landmarks: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
    outlier_threshold = 10.0  # Outlier threshold: displacement exceeding 20px is considered outlier
    
    if save_debug_videos and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cropped_frames_buffer["left"] = []
        cropped_frames_buffer["right"] = []
        motion_frames_buffer["left"] = []
        motion_frames_buffer["right"] = []

    # Open video to read original frames
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while True:
        # Read current frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB and extract keypoints
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_items = extractor.extract_landmarks_from_image(frame_rgb)
        
        # Use insightface to detect face bbox
        detected_faces = face_analysis.get(frame)
        
        # If more than 2 faces detected, keep only the 2 largest by area
        if len(detected_faces) > 2:
            faces_with_area = []
            for face in detected_faces:
                if isinstance(face, dict) and 'bbox' in face:
                    bbox = face['bbox']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    faces_with_area.append((face, area))
            # Sort by area from large to small, take top 2
            faces_with_area.sort(key=lambda x: x[1], reverse=True)
            detected_faces = [face for face, _ in faces_with_area[:2]]
        
        # Face count consistency check (only for score calculation, does not affect video saving)
        current_face_count = len(face_items)
        if expected_face_count is None and current_face_count > 0:
            expected_face_count = current_face_count
        
        # Check if deviates from expectation (used to mark failure, but don't break)
        if expected_face_count is not None:
            if current_face_count != expected_face_count:
                failure = True
            if expected_face_count != 2:
                failure = True
        
        # Sort landmark faces by x coordinate (left to right)
        face_items_sorted = sorted(enumerate(face_items), key=lambda x: x[1][0].mean(axis=0)[0]) if face_items else []
        
        # Sort detected bboxes by x coordinate (left to right)
        detected_faces_sorted = sorted(detected_faces, key=lambda x: x['bbox'][0]) if detected_faces else []
        
        # Build bbox mapping: associate by sorted position
        face_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
        for i, det_face in enumerate(detected_faces_sorted):
            if i < len(face_items_sorted):
                bbox_raw = det_face['bbox']
                x1, y1, x2, y2 = float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3])
                # Convert to square bbox
                square_bbox = make_square_bbox(x1, y1, x2, y2, frame_width, frame_height)
                face_bboxes.append(square_bbox)
            else:
                face_bboxes.append(None)
        
        # Pad with None to match face count
        while len(face_bboxes) < len(face_items_sorted):
            face_bboxes.append(None)
        
        # Prepare current frame data: left and right positions
        frame_data = {"left": None, "right": None}
        for sorted_idx, (orig_idx, (lm, emb)) in enumerate(face_items_sorted):
            if sorted_idx < 2:  # Only process first two faces
                position = "left" if sorted_idx == 0 else "right"
                current_bbox = face_bboxes[sorted_idx] if sorted_idx < len(face_bboxes) else None
                frame_data[position] = (lm, emb, current_bbox)
        
        # Process left and right positions (process even if a position is not detected)
        for position in ["left", "right"]:
            data = frame_data[position]
            
            # If current frame has data at this position
            if data is not None:
                lm, emb, current_bbox = data
                
                # Update bbox center point (using smoothing strategy: EMA) + outlier detection
                if current_bbox is not None:
                    sx1, sy1, sx2, sy2 = current_bbox
                    center_x = (sx1 + sx2) / 2
                    center_y = (sy1 + sy2) / 2
                    
                    # Outlier detection: check if deviates too much from last stable position
                    is_outlier = False
                    if stable_bbox_centers[position] is not None:
                        stable_cx, stable_cy = stable_bbox_centers[position]
                        displacement = np.sqrt((center_x - stable_cx)**2 + (center_y - stable_cy)**2)
                        if displacement > outlier_threshold:
                            is_outlier = True
                            # Use last stable center point
                            center_x, center_y = stable_cx, stable_cy
                    
                    if bbox_centers[position] is None:
                        bbox_centers[position] = (center_x, center_y)
                        # First detection, treat as stable value
                        if not is_outlier:
                            stable_bbox_centers[position] = (center_x, center_y)
                    else:
                        # Smoothly update center point: EMA with alpha=0.7
                        alpha = 0.7
                        prev_cx, prev_cy = bbox_centers[position]
                        bbox_centers[position] = (
                            alpha * center_x + (1 - alpha) * prev_cx,
                            alpha * center_y + (1 - alpha) * prev_cy
                        )
                        # If not outlier, update stable value
                        if not is_outlier:
                            stable_bbox_centers[position] = bbox_centers[position]
            else:
                lm, emb = None, None
            
            # Create output for all frames (regardless of whether bbox center exists)
            if bbox_centers[position] is not None:
                # Crop using unified max_crop_size and smoothed center point
                center_x, center_y = bbox_centers[position]
                half_size = max_crop_size / 2
                sx1 = center_x - half_size
                sy1 = center_y - half_size
                sx2 = center_x + half_size
                sy2 = center_y + half_size
                crop_size = max_crop_size
                
                # Crop face image
                x1_int, y1_int = int(sx1), int(sy1)
                x2_int, y2_int = int(sx2), int(sy2)
                cropped_face = frame[y1_int:y2_int, x1_int:x2_int].copy()
                
                # Initialize motion visualization image
                motion_vis = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                step_mean_motion = 0.0
                
                # If current frame has landmark data, process it
                if lm is not None:
                    # Transform keypoints from original image coordinates to cropped region coordinates, and scale to target_size
                    lm_in_crop = lm.copy()
                    lm_in_crop[:, 0] -= sx1
                    lm_in_crop[:, 1] -= sy1
                    
                    # Scale to target_size space (because final output will be resized to target_size)
                    scale_factor = target_size / max_crop_size
                    lm_in_crop *= scale_factor
                    
                    # Outlier detection: check if keypoints deviate too much from stable value
                    is_landmark_outlier = False
                    if stable_landmarks[position] is not None:
                        stable_lm = stable_landmarks[position]
                        # Calculate average displacement of all keypoints
                        displacement = np.linalg.norm(lm_in_crop - stable_lm, axis=-1).mean()
                        if displacement > outlier_threshold:
                            is_landmark_outlier = True
                            # Use last stable keypoints
                            lm_in_crop = stable_lm.copy()
                    
                    # Calculate motion distance (within reference space)
                    if prev_landmarks[position] is not None:
                        prev_lm = prev_landmarks[position]  # Already in reference space
                        # Select eyebrow and eye keypoints, calculate absolute average displacement
                        prev_sel = prev_lm[eye_brow_indices, :]
                        curr_sel = lm_in_crop[eye_brow_indices, :]
                        disp = curr_sel - prev_sel  # (N, 2)
                        magnitudes = np.linalg.norm(disp, axis=-1)  # (N,)
                        step_mean_motion = float(max(0.0, float(magnitudes.mean())))

                        # Frame time (seconds)
                        t_sec = (frame_idx + 0.0) / fps if fps > 0 else 0.0
                        include = True
                        # If speaking intervals provided, only count score when this position is "silent"
                        include = not is_speaking_at(position, t_sec)

                        if include:
                            # Weight: frame duration (seconds)
                            frame_weight = 1.0 / fps if fps > 0 else 1.0
                            total_motion_sum += step_mean_motion * frame_weight
                            total_weight_sum += frame_weight
                            # Accumulate per position (using position string as key)
                            pos_key = position  # "left" or "right"
                            track_motion_sum[pos_key] = track_motion_sum.get(pos_key, 0.0) + step_mean_motion * frame_weight
                            track_weight_sum[pos_key] = track_weight_sum.get(pos_key, 0.0) + frame_weight
                        
                        # Draw keypoint motion diagram (draw for all frames)
                        # Draw eye keypoints and displacement vectors
                        for i in range(len(eye_brow_indices)):
                            pt_prev = tuple(prev_sel[i].astype(int))
                            pt_curr = tuple(curr_sel[i].astype(int))
                            # Draw current keypoint
                            cv2.circle(motion_vis, pt_curr, 2, (0, 255, 0), -1)
                            # Draw displacement vector
                            cv2.arrowedLine(motion_vis, pt_prev, pt_curr, (0, 0, 255), 1, tipLength=0.3)
                        
                        # Add text information: show motion distance and whether counted in score
                        status = "SILENT" if include else "SPEAKING"
                        cv2.putText(motion_vis, f"Motion: {step_mean_motion:.2f}px ({status})", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # If outlier, add warning marker
                        if is_landmark_outlier:
                            cv2.putText(motion_vis, "OUTLIER - Using Stable", 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        # First frame, draw current keypoints
                        curr_sel = lm_in_crop[eye_brow_indices, :]
                        for i in range(len(eye_brow_indices)):
                            pt_curr = tuple(curr_sel[i].astype(int))
                            cv2.circle(motion_vis, pt_curr, 2, (0, 255, 0), -1)
                        cv2.putText(motion_vis, "First Frame", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Update previous frame keypoints for this position (in reference space)
                    prev_landmarks[position] = lm_in_crop
                    
                    # If not outlier, update stable keypoints
                    if not is_landmark_outlier:
                        stable_landmarks[position] = lm_in_crop.copy()
                    
                    # Update track information (for consistency check)
                    matched_tid = 0 if position == "left" else 1
                    if matched_tid not in active_tracks:
                        active_tracks[matched_tid] = FaceTrack(matched_tid)
                    active_tracks[matched_tid].add(lm, emb if emb is not None else None)
                else:
                    # No face detected at this position in current frame, display "Not Detected"
                    cv2.putText(motion_vis, "Not Detected", 
                                (10, crop_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Save debug videos for all frames (regardless of detection)
                if save_debug_videos and output_dir:
                    cropped_frames_buffer[position].append(cropped_face)
                    motion_frames_buffer[position].append(motion_vis)
            else:
                # This position never detected a face (bbox_center is None), create black placeholder frame of uniform size
                if save_debug_videos and output_dir:
                    # Ensure valid size is used
                    placeholder_size = max(max_crop_size, 1)
                    black_frame = np.zeros((placeholder_size, placeholder_size, 3), dtype=np.uint8)
                    black_vis = black_frame.copy()
                    cv2.putText(black_vis, "No Face Detected", 
                                (10, placeholder_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cropped_frames_buffer[position].append(black_frame)
                    motion_frames_buffer[position].append(black_vis)

        # Progress bar update
        if pbar is not None:
            pbar.update(1)
        
        # Increment frame index
        frame_idx += 1

    # Close video reading
    cap.release()
    
    # If failed, directly return failure marker
    if pbar is not None:
        pbar.close()

    if failure:
        return [], -1.0

    # Calculate weighted average motion distance
    if total_weight_sum == 0:
        return [], -1.0
    overall_avg_motion = float(total_motion_sum / total_weight_sum)
    per_face_avgs: List[Tuple[str, float]] = []
    for pos in ["left", "right"]:
        weight = track_weight_sum.get(pos, 0.0)
        if weight > 0:
            per_face_avgs.append((pos, float(track_motion_sum[pos] / weight)))
    
    # Save debug videos
    if save_debug_videos and output_dir and cropped_frames_buffer:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        for position in ["left", "right"]:
            if len(cropped_frames_buffer[position]) == 0:
                continue
            
            # Use unified target_size as final output size
            output_size = target_size
            
            # Save cropped face video (resize to target_size)
            cropped_video_path = os.path.join(output_dir, f"{video_basename}_{position}_cropped.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_cropped = cv2.VideoWriter(cropped_video_path, fourcc, fps, (output_size, output_size))
            for frame in cropped_frames_buffer[position]:
                # Check if frame is valid, if invalid create black placeholder frame
                if frame is None or frame.size == 0:
                    frame = np.zeros((output_size, output_size, 3), dtype=np.uint8)
                # All frames uniformly resize to target_size
                elif frame.shape[:2] != (output_size, output_size):
                    frame = cv2.resize(frame, (output_size, output_size))
                out_cropped.write(frame)
            out_cropped.release()
            
            # Save keypoint motion video (resize to target_size)
            motion_video_path = os.path.join(output_dir, f"{video_basename}_{position}_motion.mp4")
            out_motion = cv2.VideoWriter(motion_video_path, fourcc, fps, (output_size, output_size))
            for frame in motion_frames_buffer[position]:
                # Check if frame is valid, if invalid create black placeholder frame
                if frame is None or frame.size == 0:
                    frame = np.zeros((output_size, output_size, 3), dtype=np.uint8)
                elif frame.shape[:2] != (output_size, output_size):
                    frame = cv2.resize(frame, (output_size, output_size))
                out_motion.write(frame)
            out_motion.release()
            
            # Use ffmpeg to horizontally concatenate two videos
            hstack_video_path = os.path.join(output_dir, f"{video_basename}_{position}_combined.mp4")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', cropped_video_path,
                '-i', motion_video_path,
                '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                hstack_video_path
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Saved merged video: {hstack_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg merge failed ({position}): {e.stderr.decode()}")
    
    return per_face_avgs, overall_avg_motion


def main():
    parser = argparse.ArgumentParser(description="Calculate average motion distance of face keypoints between adjacent frames (only count silent frames; single video or directory batch processing; strict consistency check)")
    parser.add_argument("--video", type=str, default=None, help="Input video path (mutually exclusive with --dir)")
    parser.add_argument("--dir", type=str, default=None, help="Input directory containing videos (mutually exclusive with --video)")
    parser.add_argument("--ctx_id", type=int, default=1, help="GPU device ID, -1 for CPU")
    parser.add_argument("--dist", type=float, default=60.0, help="Centroid distance threshold for cross-frame matching, in pixels")
    parser.add_argument("--emb", type=float, default=0.3, help="Embedding matching cosine distance threshold (smaller is stricter)")
    parser.add_argument("--speaker_json", type=str, default="/nfs/zzzhong/codes/virtual_human/MultiPersonBenchmark/data/speaker_duration.json", help="Speaking intervals JSON path (contains left/right intervals, in seconds)")
    parser.add_argument("--save_videos", action="store_true", help="Whether to save debug videos (cropped face + keypoint motion + merged)")
    parser.add_argument("--output_dir", type=str, default="./debug_videos", help="Debug video output directory")
    args = parser.parse_args()

    if (args.video is None) == (args.dir is None):
        print("Please use --video or --dir (mutually exclusive)")
        return

    # Read speaking intervals JSON
    speaker_map: Dict[str, Dict[str, List[List[float]]]] = {}
    if args.speaker_json and os.path.exists(args.speaker_json):
        try:
            with open(args.speaker_json, 'r') as f:
                speaker_map = json.load(f)
        except Exception:
            speaker_map = {}

    if args.video is not None:
        # Initialize shared model
        shared_extractor = LandmarkExtractor(ctx_id=args.ctx_id)
        shared_face_analysis = FaceAnalysis(
            allowed_modules=['detection'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        shared_face_analysis.prepare(ctx_id=args.ctx_id, det_thresh=0.2, det_size=(224, 224))
        
        base_name = os.path.basename(args.video)
        per_face_avgs, overall_avg_motion = compute_video_average_motion_distance(
            args.video,
            ctx_id=args.ctx_id,
            distance_threshold=args.dist,
            embed_threshold=args.emb,
            extractor=shared_extractor,
            show_progress=True,
            speaker_intervals=speaker_map.get(base_name),
            save_debug_videos=args.save_videos,
            output_dir=args.output_dir,
            face_analysis=shared_face_analysis
        )
        if overall_avg_motion < 0:
            print("Track break or face count anomaly, calculation failed, overall average motion distance: -1")
            return
        for pos, avg in per_face_avgs:
            print(f"{pos} position average motion distance: {avg:.6f}")
        print(f"Overall average motion distance: {overall_avg_motion:.6f}")
        return

    # Directory batch processing
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files: List[str] = []
    try:
        for name in sorted(os.listdir(args.dir)):
            path = os.path.join(args.dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in exts:
                video_files.append(path)
    except FileNotFoundError:
        print(f"Directory does not exist: {args.dir}")
        return

    if not video_files:
        print("No video files found in directory")
        return

    rows: List[List[str]] = []
    success_scores: List[float] = []
    # In batch processing: reuse the same model instance
    shared_extractor = LandmarkExtractor(ctx_id=args.ctx_id)
    shared_face_analysis = FaceAnalysis(
        allowed_modules=['detection'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    shared_face_analysis.prepare(ctx_id=args.ctx_id, det_thresh=0.2, det_size=(224, 224))
    
    for vf in video_files:
        base_name = os.path.basename(vf)
        per_face_avgs, overall_avg_motion = compute_video_average_motion_distance(
            vf,
            ctx_id=args.ctx_id,
            distance_threshold=args.dist,
            embed_threshold=args.emb,
            extractor=shared_extractor,
            show_progress=True,
            speaker_intervals=speaker_map.get(base_name),
            save_debug_videos=args.save_videos,
            output_dir=args.output_dir,
            face_analysis=shared_face_analysis
        )
        face_summary = ";".join([f"{pos}:{avg:.6f}" for pos, avg in per_face_avgs]) if overall_avg_motion >= 0 else ""
        score_str = f"{overall_avg_motion:.6f}" if overall_avg_motion >= 0 else "-1"
        rows.append([os.path.basename(vf), score_str, face_summary])
        if overall_avg_motion >= 0:
            success_scores.append(overall_avg_motion)

    avg_score = sum(success_scores) / len(success_scores) if success_scores else -1.0
    base = os.path.basename(os.path.normpath(args.dir))
    out_name = f"{avg_score:.6f}_{base}.csv" if avg_score >= 0 else f"-1_{base}.csv"
    # CSV defaults to current working directory to avoid insufficient permissions in target directory
    out_path = os.path.join(os.getcwd(), out_name)

    # Write CSV: filename, overall, per_face(track:mean;...)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "overall_avg_motion", "per_face_avg_motion"]) 
        for r in rows:
            writer.writerow(r)

    if avg_score >= 0:
        print(f"Saved: {out_path}, directory average score: {avg_score:.6f}")
    else:
        print(f"Saved: {out_path}, no successful samples in directory, average score is -1")


if __name__ == "__main__":
    main()


