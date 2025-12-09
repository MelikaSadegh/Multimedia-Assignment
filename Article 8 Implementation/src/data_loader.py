import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config

class HAM10000Dataset(Dataset):
    """Dataset کلاس برای HAM10000"""
    
    def __init__(self, metadata_df: pd.DataFrame, 
                 image_dir_part1: str, 
                 image_dir_part2: str,
                 transform=None,
                 mode: str = 'train'):
        
        self.metadata_df = metadata_df
        self.image_dir_part1 = image_dir_part1
        self.image_dir_part2 = image_dir_part2
        self.transform = transform
        self.mode = mode
        self.config = Config()
        
        # تصحیح مسیر تصاویر
        self._prepare_image_paths()
        
    def _prepare_image_paths(self):
        """آماده‌سازی مسیر تصاویر"""
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.metadata_df.iterrows():
            image_id = row['image_id']
            label = self.config.class_mapping[row['dx']]
            
            # جستجوی تصویر در دو پوشه
            image_path = None
            
            # بررسی در پوشه اول
            possible_path1 = os.path.join(self.image_dir_part1, f"{image_id}.jpg")
            if os.path.exists(possible_path1):
                image_path = possible_path1
            
            # بررسی در پوشه دوم
            if not image_path:
                possible_path2 = os.path.join(self.image_dir_part2, f"{image_id}.jpg")
                if os.path.exists(possible_path2):
                    image_path = possible_path2
            
            # بررسی در هر دو پوشه با پسوند‌های مختلف
            if not image_path:
                for ext in ['.jpg', '.jpeg', '.png']:
                    possible_path1 = os.path.join(self.image_dir_part1, f"{image_id}{ext}")
                    possible_path2 = os.path.join(self.image_dir_part2, f"{image_id}{ext}")
                    
                    if os.path.exists(possible_path1):
                        image_path = possible_path1
                        break
                    elif os.path.exists(possible_path2):
                        image_path = possible_path2
                        break
            
            if image_path:
                self.image_paths.append(image_path)
                self.labels.append(label)
            else:
                print(f"Warning: Image {image_id} not found in any directory")
        
        print(f"Loaded {len(self.image_paths)} images for {self.mode} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # بارگذاری تصویر
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # اعمال transform
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # پیش‌پردازش پایه
                image = cv2.resize(image, self.config.image_size)
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # برگرداندن یک تصویر placeholder در صورت خطا
            placeholder = np.zeros((*self.config.image_size, 3), dtype=np.uint8)
            placeholder = torch.from_numpy(placeholder).permute(2, 0, 1).float()
            return placeholder, label

class DataLoaderModule:
    """مدیریت بارگذاری داده‌ها"""
    
    def __init__(self, config: Config):
        self.config = config
        self.metadata_df = None
        
    def load_metadata(self) -> pd.DataFrame:
        """بارگذاری متادیتای دیتاست"""
        try:
            self.metadata_df = pd.read_csv(self.config.metadata_path)
            print(f"Loaded metadata with {len(self.metadata_df)} samples")
            
            # نمایش توزیع کلاس‌ها
            print("\nClass Distribution:")
            print(self.metadata_df['dx'].value_counts())
            
            return self.metadata_df
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def get_train_transforms(self):
        """تبدیل‌های آموزشی"""
        return A.Compose([
            A.Resize(self.config.image_size[0], self.config.image_size[1]),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),  # اصلاح شده
            A.VerticalFlip(p=0.5),    # اضافه شده
            A.Transpose(p=0.5),
            A.Rotate(limit=self.config.rotation_range, p=0.5),  # اصلاح نام
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # مقدار ثابت
                contrast_limit=0.2,    # مقدار ثابت
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def get_val_transforms(self):
        """تبدیل‌های اعتبارسنجی"""
        return A.Compose([
            A.Resize(self.config.image_size[0], self.config.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """تقسیم داده‌ها به آموزش، اعتبارسنجی و تست"""
        
        # اول تست را جدا کن
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            stratify=df['dx'],
            random_state=self.config.random_state
        )
        
        # سپس اعتبارسنجی را از آموزش جدا کن
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config.val_size / (1 - self.config.test_size),
            stratify=train_val_df['dx'],
            random_state=self.config.random_state
        )
        
        print(f"\nDataset Split:")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """ایجاد DataLoaderها"""
        
        if self.metadata_df is None:
            self.load_metadata()
        
        # تقسیم داده‌ها
        train_df, val_df, test_df = self.split_data(self.metadata_df)
        
        # ایجاد datasets
        train_dataset = HAM10000Dataset(
            train_df,
            self.config.images_part1,
            self.config.images_part2,
            transform=self.get_train_transforms(),
            mode='train'
        )
        
        val_dataset = HAM10000Dataset(
            val_df,
            self.config.images_part1,
            self.config.images_part2,
            transform=self.get_val_transforms(),
            mode='val'
        )
        
        test_dataset = HAM10000Dataset(
            test_df,
            self.config.images_part1,
            self.config.images_part2,
            transform=self.get_val_transforms(),
            mode='test'
        )
        
        # ایجاد dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_class_distribution(self):
        """تحلیل توزیع کلاس‌ها"""
        if self.metadata_df is None:
            self.load_metadata()
        
        class_counts = self.metadata_df['dx'].value_counts()
        
        print("\nDetailed Class Distribution:")
        for dx, count in class_counts.items():
            print(f"{dx}: {count} samples ({count/len(self.metadata_df)*100:.2f}%)")
        
        return class_counts