from PIL import Image
from insightface.app import FaceAnalysis
import numpy as np
import os
from pathlib import Path
import time
import argparse
import cv2

class FaceInference:
    """人脸检测推理类，封装insightface的推理功能"""
    
    def __init__(self, det_thresh=0.5, det_size=(640, 640), ctx_id=0):
        """
        初始化人脸检测器
        
        Args:
            det_thresh: 检测阈值
            det_size: 检测图像尺寸
            ctx_id: GPU设备ID，如果为None则自动检测当前进程的local_rank
        """
        self.face_analysis = FaceAnalysis(
            allowed_modules=['detection'], 
            providers=['CUDAExecutionProvider'], 
            provider_options=[{"device_id": str(ctx_id)}], # make sure use provider_options to specify GPU ran
        )
        
        self.face_analysis.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
    
    def _make_square_bbox(self, x1, y1, x2, y2, image_width, image_height):
        """
        将矩形bbox转换为方形bbox，保持人脸比例不变
        
        Args:
            x1, y1, x2, y2: 原始bbox坐标
            image_width, image_height: 图像尺寸
            
        Returns:
            tuple: (new_x1, new_y1, new_x2, new_y2) 方形bbox坐标
        """
        # 计算原始bbox的中心点和尺寸
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # 取较大的边作为方形的边长
        square_size = max(width, height)
        
        # 计算方形bbox的坐标
        half_size = square_size / 2
        new_x1 = center_x - half_size
        new_y1 = center_y - half_size
        new_x2 = center_x + half_size
        new_y2 = center_y + half_size
        
        # 处理边界情况，确保方形bbox在图像范围内
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
        
        # 再次确保坐标在有效范围内
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image_width, new_x2)
        new_y2 = min(image_height, new_y2)
        
        return new_x1, new_y1, new_x2, new_y2
    
    def infer_from_array(self, image_array, n=None):
        """
        对输入numpy数组进行人脸检测推理
        
        Args:
            image_array: numpy数组，形状为[H, W, 3]，值范围为0-255
            n: 选择前n个最大的人脸，如果为None则选择所有人脸
            
        Returns:
            dict: 包含检测结果的字典，格式为：
                  {
                      'faces': 检测到的人脸列表,
                      'bboxes': bbox列表，每个元素为[x, y, width, height],
                      'masks': mask列表，每个元素为单通道mask图像,
                      'masked_images': masked图像列表，每个元素为应用mask后的图像,
                      'image_shape': 原始图像的形状 (height, width, channels)
                  }
                  如果未检测到人脸，返回中心区域矩形作为默认bbox
        """
        try:
            if image_array is None:
                print("错误：输入图像数组为空")
                return {}
            
            # 确保图像数组是正确的格式
            if len(image_array.shape) != 3 or image_array.shape[2] != 3:
                print(f"错误：图像数组形状不正确，期望[H, W, 3]，实际{image_array.shape}")
                return {}
            
            # 确保数据类型和值范围正确
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            
            faces = self.face_analysis.get(image_array)
            height, width = image_array.shape[:2]
            
            if not faces: 
                return {
                    'faces': [],
                    'bboxes': [],
                    'masks': [],
                    'masked_images': [],
                    'image_shape': image_array.shape
                }
            
            # 先按人脸面积大小排序，选择前n个最大的人脸
            if n is not None and n > 0:
                # 计算每个人脸的面积并排序
                faces_with_area = [(face, (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1])) for face in faces]
                faces_with_area.sort(key=lambda x: x[1], reverse=True)  # 按面积从大到小排序
                faces = [face for face, _ in faces_with_area[:n]]  # 取前n个最大的人脸
                # print(f"选择了前{n}个最大的人脸，总面积分别为: {[area for _, area in faces_with_area[:n]]}")
            
            # 再按x坐标从左到右排序
            faces = sorted(faces, key=lambda x: x['bbox'][0])
            
            # 生成bbox、mask和masked图像
            bboxes = []
            masks = []
            masked_images = []
            
            for i, face in enumerate(faces):
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # 将矩形bbox转换为方形bbox
                square_x1, square_y1, square_x2, square_y2 = self._make_square_bbox(
                    x1, y1, x2, y2, width, height
                )
                
                # 创建方形mask
                mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
                mask[int(square_y1):int(square_y2), int(square_x1):int(square_x2)] = 1.0
                
                # 创建mask与原图相乘的结果
                masked_image = image_array.copy()
                masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)
                
                bboxes.append([square_x1, square_y1, square_x2 - square_x1, square_y2 - square_y1])
                masks.append(mask)
                masked_images.append(masked_image)
                
                # print(f"  人脸 {i+1}: 原始bbox=[{x1:.1f}, {y1:.1f}, {x2-x1:.1f}, {y2-y1:.1f}] -> 方形bbox=[{square_x1:.1f}, {square_y1:.1f}, {square_x2-square_x1:.1f}, {square_y2-square_y1:.1f}]")
            
            return {
                'faces': faces,
                'bboxes': bboxes,
                'masks': masks,
                'masked_images': masked_images,
                'image_shape': image_array.shape
            }
            
        except Exception as e:
            print(f"处理图像数组时出错: {str(e)}")
            # 异常情况下也返回中心区域
            if 'image_array' in locals() and image_array is not None:
                return {
                    'faces': [],
                    'bboxes': [],
                    'masks': [],
                    'masked_images': [],
                    'image_shape': image_array.shape
                }
            
            return {}
    
    def infer(self, image_path, n=None):
        """
        对输入图像进行人脸检测推理
        
        Args:
            image_path: 图像文件路径或图片
            n: 选择前n个最大的人脸，如果为None则选择所有人脸
            
        Returns:
            dict: 包含检测结果的字典，格式为：
                  {
                      'faces': 检测到的人脸列表,
                      'bboxes': bbox列表，每个元素为[x, y, width, height],
                      'masks': mask列表，每个元素为单通道mask图像,
                      'masked_images': masked图像列表，每个元素为应用mask后的图像,
                      'image_shape': 原始图像的形状 (height, width, channels)
                  }
                  如果未检测到人脸，返回中心区域矩形作为默认bbox
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"错误：无法读取图像 {image_path}")
                # 无法读取图像，返回空结果
                return {}
            
            faces = self.face_analysis.get(image)
            height, width = image.shape[:2]
            
            if not faces:
                print(f"警告：图像 {os.path.basename(image_path)} 中未检测到人脸，使用中心区域作为默认方形bbox")
                
                # 计算中心区域方形（边长为原图较小边的50%）
                min_dim = min(width, height)
                square_size = min_dim // 2
                center_x, center_y = width // 2, height // 2
                
                x1 = center_x - square_size // 2
                y1 = center_y - square_size // 2
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # 确保bbox在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # 创建中心区域的方形mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
                
                # 创建masked图像
                masked_image = image.copy()
                masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)
                
                return {
                    'faces': [],
                    'bboxes': [[x1, y1, x2 - x1, y2 - y1]],
                    'masks': [mask],
                    'masked_images': [masked_image],
                    'image_shape': image.shape
                }
            
            # 先按人脸面积大小排序，选择前n个最大的人脸
            if n is not None and n > 0:
                # 计算每个人脸的面积并排序
                faces_with_area = [(face, (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1])) for face in faces]
                faces_with_area.sort(key=lambda x: x[1], reverse=True)  # 按面积从大到小排序
                faces = [face for face, _ in faces_with_area[:n]]  # 取前n个最大的人脸
            
            # 再按x坐标从左到右排序
            faces = sorted(faces, key=lambda x: x['bbox'][0])
            
            # 生成bbox、mask和masked图像
            bboxes = []
            masks = []
            masked_images = []
            
            for i, face in enumerate(faces):
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # 将矩形bbox转换为方形bbox
                square_x1, square_y1, square_x2, square_y2 = self._make_square_bbox(
                    x1, y1, x2, y2, width, height
                )
                
                # 创建方形mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[int(square_y1):int(square_y2), int(square_x1):int(square_x2)] = 1.0
                
                # 创建mask与原图相乘的结果
                masked_image = image.copy()
                masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)
                
                bboxes.append([square_x1, square_y1, square_x2 - square_x1, square_y2 - square_y1])
                masks.append(mask)
                masked_images.append(masked_image)
                
            return {
                'faces': faces,
                'bboxes': bboxes,
                'masks': masks,
                'masked_images': masked_images,
                'image_shape': image.shape
            }
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            # 异常情况下也返回中心区域方形
            if 'image' in locals() and image is not None:
                height, width = image.shape[:2]
                
                # 计算中心区域方形（边长为原图较小边的50%）
                min_dim = min(width, height)
                square_size = min_dim // 2
                center_x, center_y = width // 2, height // 2
                
                x1 = center_x - square_size // 2
                y1 = center_y - square_size // 2
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # 确保bbox在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # 创建中心区域的方形mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
                
                # 创建masked图像
                masked_image = image.copy()
                masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)
                
                return {
                    'faces': [],
                    'bboxes': [[x1, y1, x2 - x1, y2 - y1]],
                    'masks': [mask],
                    'masked_images': [masked_image],
                    'image_shape': image.shape
                }
            
            return {}


class FaceProcessor:
    def __init__(self, det_thresh=0.5, det_size=(640, 640)):
        self.face_analysis = FaceAnalysis(allowed_modules=['detection'])
        self.face_analysis.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    
    def _make_square_bbox(self, x1, y1, x2, y2, image_width, image_height):
        """
        将矩形bbox转换为方形bbox，保持人脸比例不变
        
        Args:
            x1, y1, x2, y2: 原始bbox坐标
            image_width, image_height: 图像尺寸
            
        Returns:
            tuple: (new_x1, new_y1, new_x2, new_y2) 方形bbox坐标
        """
        # 计算原始bbox的中心点和尺寸
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # 取较大的边作为方形的边长
        square_size = max(width, height)
        
        # 计算方形bbox的坐标
        half_size = square_size / 2
        new_x1 = center_x - half_size
        new_y1 = center_y - half_size
        new_x2 = center_x + half_size
        new_y2 = center_y + half_size
        
        # 处理边界情况，确保方形bbox在图像范围内
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
        
        # 再次确保坐标在有效范围内
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(image_width, new_x2)
        new_y2 = min(image_height, new_y2)
        
        return new_x1, new_y1, new_x2, new_y2

    def get_face_bbox_and_mask(self, image):
        faces = self.face_analysis.get(image)
        if not faces:
            print("警告：图像中未检测到人脸。")
            return None, None, None
        
        # 按x坐标从左到右排序
        faces = sorted(faces, key=lambda x: x['bbox'][0])
        
        height, width = image.shape[:2]
        bboxes = []
        masks = []
        masked_images = []
        
        for i, face in enumerate(faces):
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # 将矩形bbox转换为方形bbox
            square_x1, square_y1, square_x2, square_y2 = self._make_square_bbox(
                x1, y1, x2, y2, width, height
            )
            
            # 创建方形mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[int(square_y1):int(square_y2), int(square_x1):int(square_x2)] = 1.0
            
            # 创建mask与原图相乘的结果
            masked_image = image.copy()
            masked_image = cv2.bitwise_and(masked_image, masked_image, mask=mask)
            
            bboxes.append([square_x1, square_y1, square_x2 - square_x1, square_y2 - square_y1])
            masks.append(mask)
            masked_images.append(masked_image)
            
        return bboxes, masks, masked_images

def main():
    parser = argparse.ArgumentParser(description='Process images to detect faces and save bbox, mask, and masked images.')
    parser.add_argument('--input_dir', type=str, default="./data/bbox_test_input", help='Directory containing input images.')
    parser.add_argument('--bbox_output_dir', type=str, default="./temp/bbox", help='Directory to save bbox npy files.')
    parser.add_argument('--mask_output_dir', type=str, default="./temp/mask", help='Directory to save mask images.')
    parser.add_argument('--masked_image_output_dir', type=str, default="./temp/masked_images", help='Directory to save masked images.')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.bbox_output_dir, exist_ok=True)
    os.makedirs(args.mask_output_dir, exist_ok=True)
    os.makedirs(args.masked_image_output_dir, exist_ok=True)
    
    # 初始化人脸检测器
    face_processor = FaceProcessor()
    
    # 支持的图像格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(args.input_dir):
        if Path(file).suffix.lower() in supported_formats:
            image_files.append(file)
    
    if not image_files:
        print(f"警告：在目录 {args.input_dir} 中未找到支持的图像文件")
        return
       
    # 处理每个图像
    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"  错误：无法读取图像 {image_path}")
            continue
        
        # 获取人脸检测结果
        bboxes, masks, masked_images = face_processor.get_face_bbox_and_mask(image)
        
        if bboxes is None:
            print(f"  跳过：未检测到人脸")
            continue
        
        # 生成基础文件名（不含扩展名）
        base_name = Path(image_file).stem
        
        # 保存bbox为npy文件
        bbox_file = os.path.join(args.bbox_output_dir, f"{base_name}_bbox.npy")
        np.save(bbox_file, np.array(bboxes))
        
        # 保存mask和masked图像
        for i, (mask, masked_image) in enumerate(zip(masks, masked_images)):
            # 保存mask
            mask_file = os.path.join(args.mask_output_dir, f"{base_name}_face{i+1}_mask.png")
            cv2.imwrite(mask_file, mask)
            
            # 保存masked图像
            masked_image_file = os.path.join(args.masked_image_output_dir, f"{base_name}_face{i+1}_masked.png")
            cv2.imwrite(masked_image_file, masked_image)
            
            print(f"  已保存人脸{i+1}的mask: {mask_file}")
            print(f"  已保存人脸{i+1}的masked图像: {masked_image_file}")
    
    print(f"\n处理完成！")
    print(f"bbox文件保存在: {args.bbox_output_dir}")
    print(f"mask文件保存在: {args.mask_output_dir}")
    print(f"masked图像保存在: {args.masked_image_output_dir}")


if __name__ == "__main__":
    main()
