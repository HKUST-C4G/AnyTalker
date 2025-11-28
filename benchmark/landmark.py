import cv2
import insightface
import numpy as np
from typing import List, Tuple, Optional, Generator
import os


class LandmarkExtractor:
    """Face landmark and embedding extraction tool class"""
    
    # Index mapping from 106-point landmarks to 68-point landmarks
    landmark106to68 = [1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,    # 17 cheek points
                       43,48,49,51,50,      # 5 left eyebrow points
                       102,103,104,105,101, # 5 right eyebrow points
                       72,73,74,86,78,79,80,85,84, # 9 nose points
                       35,41,42,39,37,36,   # 6 left eye points
                       89,95,96,93,91,90,   # 6 right eye points
                       52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54 # 20 mouth points
                       ]
    
    def __init__(self, ctx_id: int = -1):
        """
        Initialize face analysis model
        :param ctx_id: GPU device ID, -1 means use CPU
        """
        self.model = insightface.app.FaceAnalysis()
        # Enable recognition module to get embedding
        try:
            self.model.prepare(ctx_id=ctx_id, det_thresh=0.2, det_size=(640, 640))
        except TypeError:
            # Some versions don't support det_size parameter
            self.model.prepare(ctx_id=ctx_id, det_thresh=0.2)
        
        # Color list for distinguishing different faces
        self.colors = [
            (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (0, 128, 255), (128, 0, 255), (0, 255, 128), (255, 128, 0), (128, 255, 0)
        ]
    
    def extract_landmarks_from_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Extract 106-point landmarks and embedding for all faces from a single image
        :param image: Input image (H, W, 3) in RGB format
        :return: List, each element is (landmarks(106,2), embedding(d,))
        """
        faces = self.model.get(image)
        
        # If more than 2 faces are detected, keep only the 2 largest by area
        if len(faces) > 2:
            # Calculate area for each face and sort
            faces_with_area = []
            for face in faces:
                if hasattr(face, 'bbox'):
                    bbox = face.bbox
                elif isinstance(face, dict) and 'bbox' in face:
                    bbox = face['bbox']
                else:
                    continue
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                faces_with_area.append((face, area))
            
            # Sort by area in descending order, take top 2
            faces_with_area.sort(key=lambda x: x[1], reverse=True)
            faces = [face for face, _ in faces_with_area[:2]]
        
        results: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        
        for face in faces:
            landmarks = None
            embedding = None
            if isinstance(face, dict) and 'landmark_2d_106' in face:
                landmarks = face['landmark_2d_106']
                # Compatible with different key names
                if 'normed_embedding' in face:
                    embedding = face['normed_embedding']
                elif 'embedding' in face:
                    embedding = face['embedding']
            elif hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106
                # Embedding from insightface Face object
                if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                    embedding = face.normed_embedding
                elif hasattr(face, 'embedding') and face.embedding is not None:
                    embedding = face.embedding
            
            if landmarks is not None:
                lm = np.asarray(landmarks, dtype=np.float32)
                emb = None if embedding is None else np.asarray(embedding, dtype=np.float32)
                results.append((lm, emb))
        
        return results
    
    def extract_landmarks_from_video(self, video_path: str) -> Generator[Tuple[int, List[Tuple[np.ndarray, Optional[np.ndarray]]]], None, None]:
        """
        Extract landmarks frame by frame from video
        :param video_path: Video file path
        :return: Generator, yields (frame_idx, [(landmarks, embedding), ...])
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks_list = self.extract_landmarks_from_image(frame_rgb)
                
                yield frame_idx, landmarks_list
                frame_idx += 1
        finally:
            cap.release()
    
    def visualize_landmarks(self, image: np.ndarray, landmarks_list: List[np.ndarray], 
                          point_size: int = 2) -> np.ndarray:
        """
        Visualize only eyebrow and eye landmarks on black background (no numeric labels)
        :param image: Reference image for determining output size
        :param landmarks_list: List of landmarks (each element is (landmarks, embedding) or structure containing landmarks)
        :param point_size: Landmark point size
        :return: Visualization image
        """
        canvas = np.zeros_like(image)

        # Eyebrow and eye indices based on 106-point definition
        eyebrow_and_eye_indices = [
            43, 48, 49, 51, 50,        # 5 left eyebrow points
            102, 103, 104, 105, 101,   # 5 right eyebrow points
            35, 41, 42, 39, 37, 36,    # 6 left eye points
            89, 95, 96, 93, 91, 90     # 6 right eye points
        ]

        for idx, landmarks in enumerate(landmarks_list):
            color = self.colors[idx % len(self.colors)]
            pts = landmarks[0].astype(int)

            # Draw only eyebrow and eye points, no numbers
            for landmark_idx in eyebrow_and_eye_indices:
                if 0 <= landmark_idx < len(pts):
                    px = int(pts[landmark_idx][0])
                    py = int(pts[landmark_idx][1])
                    cv2.circle(canvas, (px, py), point_size, color, -1)

        return canvas
    
    def save_landmarks_video(self, video_path: str, output_path: str, 
                           point_size: int = 2, fps: Optional[int] = None) -> None:
        """
        Process video and save landmark visualization video
        :param video_path: Input video path
        :param output_path: Output video path
        :param point_size: Landmark point size
        :param fps: Output video frame rate, None means use original video frame rate
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video information
        if fps is None:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set video codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_idx, landmarks_list in self.extract_landmarks_from_video(video_path):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                canvas = self.visualize_landmarks(frame_rgb, landmarks_list, point_size)
                
                # Convert back to BGR and write to video
                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                out.write(canvas_bgr)
                
                print(f"Processing frame {frame_idx}, detected {len(landmarks_list)} faces")
        finally:
            cap.release()
            out.release()


def main():
    """Main function, demonstrates how to use LandmarkExtractor"""
    # Initialize extractor
    extractor = LandmarkExtractor()
    
    # Process video
    video_path = "xxx"  # Can be changed to video path
    output_path = "landmarks_output.mp4"
    
    if os.path.exists(video_path):
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Process video
            extractor.save_landmarks_video(video_path, output_path)
            print(f"Landmark video saved to: {output_path}")
        else:
            # Process single image
            image = cv2.imread(video_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks_list = extractor.extract_landmarks_from_image(image_rgb)
            canvas = extractor.visualize_landmarks(image_rgb, landmarks_list)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            # Resize to 1024x1024 for easier viewing
            canvas_bgr_resized = cv2.resize(canvas_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("output.jpg", canvas_bgr_resized)
            print(f"Detected {len(landmarks_list)} faces, landmark image (1024x1024) saved to: output.jpg")


if __name__ == '__main__':
    main()