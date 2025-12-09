import os
from dataclasses import dataclass

@dataclass
class Config:
    """پیکربندی پروژه"""
    
    # مسیرهای داده
    data_dir: str = "data"
    images_part1: str = os.path.join("data", "HAM10000_images_part_1")
    images_part2: str = os.path.join("data", "HAM10000_images_part_2")
    metadata_path: str = os.path.join("data", "HAM10000_metadata.csv")
    
    # پارامترهای تصویر
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    
    # کلاس‌ها
    class_mapping = {
        'akiec': 0,  # Actinic Keratoses
        'bcc': 1,    # Basal cell carcinoma
        'bkl': 2,    # Benign keratosis-like lesions
        'df': 3,     # Dermatofibroma
        'mel': 4,    # Melanoma
        'nv': 5,     # Melanocytic nevi
        'vasc': 6    # Vascular lesions
    }
    
    reverse_class_mapping = {
        0: 'akiec',
        1: 'bcc',
        2: 'bkl',
        3: 'df',
        4: 'mel',
        5: 'nv',
        6: 'vasc'
    }
    
    class_names = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevus',
        'vasc': 'Vascular Lesion'
    }
    
    # پارامترهای آموزش
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout_rate: float = 0.3
    
    # پارامترهای ViT
    vit_patch_size: int = 16
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 12
    vit_mlp_dim: int = 3072
    
    # پارامترهای Augmentation
    augmentation_prob: float = 0.7
    rotation_range: int = 15
    brightness_range: tuple = (0.8, 1.2)
    contrast_range: tuple = (0.8, 1.2)
    saturation_range: tuple = (0.8, 1.2)
    
    # مسیرهای ذخیره
    model_save_dir: str = "models/saved_models"
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    reports_dir: str = "results/reports"
    
    # پارامترهای تقسیم داده
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    
    def __post_init__(self):
        """ایجاد دایرکتوری‌های لازم"""
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    