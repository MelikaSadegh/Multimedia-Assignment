import cv2
import numpy as np
import albumentations as A
import os
from tqdm import tqdm

class AdvancedAugmentation:
    """Augmentation پیشرفته برای دیتاست پوست"""
    
    def __init__(self, target_samples=980):
        self.target_samples = target_samples
        self.transform = self.get_advanced_transform()
    
    def get_advanced_transform(self):
        """تبدیل‌های پیشرفته برای تصاویر پوست"""
        return A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
            
            # Color transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                brightness_by_max=True,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Quality transformations
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.2),
            
            # Blur and sharpness
            A.Blur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.1),
            
            # Advanced augmentations
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                mask_fill_value=None,
                p=0.3
            ),
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
            
            # Post-processing
            A.ToFloat(max_value=255.0, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
                p=1.0
            ),
        ])
    
    def augment_class(self, images_paths, class_name, output_dir):
        """Augment یک کلاس خاص"""
        os.makedirs(output_dir, exist_ok=True)
        
        augmented_images = []
        current_count = len(images_paths)
        
        print(f"Augmenting class {class_name}: {current_count} -> {self.target_samples}")
        
        # اگر تعداد کافی است، فقط undersample
        if current_count >= self.target_samples:
            selected = np.random.choice(images_paths, self.target_samples, replace=False)
            for img_path in selected:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented_images.append(img_path)
            return augmented_images
        
        # اگر نیاز به augmentation داریم
        needed = self.target_samples - current_count
        augmentation_per_image = max(1, needed // current_count)
        
        for img_path in tqdm(images_paths, desc=f"Augmenting {class_name}"):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented_images.append(img_path)
                
                # ایجاد augmented samples
                for i in range(augmentation_per_image):
                    augmented = self.transform(image=img)
                    aug_img = augmented['image']
                    
                    # ذخیره تصویر
                    aug_filename = f"{class_name}_aug_{len(augmented_images)}_{os.path.basename(img_path)}"
                    aug_path = os.path.join(output_dir, aug_filename)
                    
                    # تبدیل به uint8 برای ذخیره
                    if aug_img.dtype == np.float32:
                        aug_img = (aug_img * 255).astype(np.uint8)
                    
                    cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    augmented_images.append(aug_path)
                    
                    if len(augmented_images) >= self.target_samples:
                        break
                
                if len(augmented_images) >= self.target_samples:
                    break
                    
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")
                continue
        
        # اگر هنوز کافی نیست، random augmentation بیشتر
        while len(augmented_images) < self.target_samples:
            random_img_path = np.random.choice(images_paths)
            try:
                img = cv2.imread(random_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                augmented = self.transform(image=img)
                aug_img = augmented['image']
                
                aug_filename = f"{class_name}_extra_{len(augmented_images)}.jpg"
                aug_path = os.path.join(output_dir, aug_filename)
                
                if aug_img.dtype == np.float32:
                    aug_img = (aug_img * 255).astype(np.uint8)
                
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                augmented_images.append(aug_path)
                
            except Exception as e:
                continue
        
        return augmented_images[:self.target_samples]

# استفاده از کلاس
def run_advanced_augmentation():
    aug = AdvancedAugmentation(target_samples=980)
    
    # اینجا باید دیتاست واقعی را بارگذاری کنید
    # paths_by_class = load_your_dataset()
    
    # برای هر کلاس:
    # augmented_paths = aug.augment_class(paths_by_class[class_name], class_name, './data/augmented')