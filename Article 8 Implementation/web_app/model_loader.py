import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import json

class SkinCancerModel:
    """کلاس برای بارگذاری و مدیریت مدل سرطان پوست"""
    
    def __init__(self, model_path=None):
        # تنظیمات
        self.image_size = 224
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {self.device}")
        
        # تعریف کلاس‌ها
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_descriptions = {
            'akiec': "Actinic Keratoses",
            'bcc': "Basal Cell Carcinoma", 
            'bkl': "Benign Keratosis",
            'df': "Dermatofibroma",
            'mel': "Melanoma",
            'nv': "Melanocytic Nevus",
            'vasc': "Vascular Lesion"
        }
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # بارگذاری مدل
        self.model = self._load_model(model_path)
        self.model_type = "ResNet18"
        self.accuracy = 0.85  # از آموزش واقعی می‌توانید مقدار واقعی را قرار دهید
    
    def _load_model(self, model_path=None):
        """بارگذاری مدل آموزش دیده"""
        print("🔄 Loading model...")
        
        # اولویت مسیرهای مدل
        if model_path is None:
            model_paths = [
                'models/converted_resnet.pth',
                '../models/converted_resnet.pth',
                '../models/fast_model_best.pth',
                'models/fast_model_best.pth'
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"📂 Found model at: {path}")
                    break
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # ایجاد مدل ResNet18
        model = models.resnet18(pretrained=False)
        
        # تنظیم لایه خروجی برای 7 کلاس
        # طبق تحلیل state_dict، مدل شما fc لایه‌های custom دارد
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.class_names))
        )
        
        # بارگذاری وزن‌ها
        print(f"📥 Loading weights from {model_path}...")
        
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        # تشخیص ساختار checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # بارگذاری وزن‌ها با strict=False برای تطبیق بهتر
        model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model
    
    def is_cancer_class(self, class_name):
        """تشخیص آیا کلاس سرطان است"""
        cancer_classes = ['mel', 'bcc', 'akiec']
        return class_name in cancer_classes
    
    def get_risk_level(self, class_name):
        """دریافت سطح ریسک"""
        risk_levels = {
            'mel': 'Very High',
            'bcc': 'High',
            'akiec': 'High',
            'vasc': 'Medium',
            'bkl': 'Low',
            'df': 'Very Low',
            'nv': 'Very Low'
        }
        return risk_levels.get(class_name, 'Unknown')
    
    def get_recommendation(self, class_name, confidence):
        """دریافت توصیه پزشکی"""
        recommendations = {
            'mel': f"⚠️ ملانوما شناسایی شد ({confidence:.1f}% اطمینان). مراجعه فوری به متخصص پوست ضروری است.",
            'bcc': f"⚠️ کارسینوم سلول پایه‌ای ({confidence:.1f}% اطمینان). مشاوره پزشکی در اسرع وقت توصیه می‌شود.",
            'akiec': f"⚠️ کراتوز اکتینیک ({confidence:.1f}% اطمینان). این وضعیت پیش‌سرطانی است. مشاوره پزشکی توصیه می‌شود.",
            'vasc': f"🔶 ضایعه عروقی ({confidence:.1f}% اطمینان). مشاوره پزشکی برای تشخیص دقیق توصیه می‌شود.",
            'bkl': f"✅ ضایعه شبه کراتوز خوش‌خیم ({confidence:.1f}% اطمینان). نظارت منظم کافی است.",
            'df': f"✅ درماتوفیبروما ({confidence:.1f}% اطمینان). معمولاً خوش‌خیم است.",
            'nv': f"✅ خال ملانوسیتی ({confidence:.1f}% اطمینان). معمولاً خوش‌خیم است."
        }
        return recommendations.get(class_name, "برای تشخیص دقیق با متخصص پوست مشورت کنید.")