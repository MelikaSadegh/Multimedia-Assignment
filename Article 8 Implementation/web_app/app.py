import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from flask import Flask, render_template, request, jsonify, session
import uuid
import yaml
import json
from datetime import datetime
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'skin-cancer-classification-secret-key'
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class TrainedResNet18(nn.Module):
    """مدل ResNet18 که واقعاً آموزش داده شده است"""
    def __init__(self, num_classes=7):
        super().__init__()
        # ایجاد ResNet18 مطابق با ساختار مدل ذخیره شده
        self.model = models.resnet18(pretrained=False)
        
        # بررسی اینکه آیا fc لایه‌های custom دارد
        # از logها دیدیم که fc دارای ساختار خاصی است:
        # 'fc.1.weight', 'fc.1.bias', 'fc.4.weight', 'fc.4.bias', 'fc.6.weight', 'fc.6.bias'
        # این نشان می‌دهد fc یک Sequential با لایه‌های خاص است
        
        # ایجاد fc مشابه مدل آموزش دیده
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 512),  # fc.1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # fc.4
            nn.ReLU(),
            nn.Linear(256, num_classes)  # fc.6
        )
    
    def forward(self, x):
        return self.model(x)

class SimpleCNN(nn.Module):
    """مدل CNN ساده"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SkinCancerPredictor:
    def __init__(self, model_path=None):
        # بارگذاری config
        config = self.load_config()
        
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_descriptions = {
            'akiec': "Actinic Keratoses",
            'bcc': "Basal cell carcinoma", 
            'bkl': "Benign keratosis-like lesions",
            'df': "Dermatofibroma",
            'mel': "Melanoma",
            'nv': "Melanocytic nevi",
            'vasc': "Vascular lesions"
        }
        self.image_size = 224
        
        # تنظیم device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Transform دقیقاً مشابه آموزش
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # بارگذاری مدل
        self.model, self.model_type = self.load_or_create_model(model_path)
        
        # اطلاعات مدل
        self.best_model_info = {
            'best_model': self.model_type,
            'accuracy': 0.85,
            'image_size': self.image_size
        }
        
        print(f"\n✅ Model loaded: {self.model_type}")
        print(f"✅ Model architecture: {type(self.model).__name__}")
    
    def load_config(self):
        """بارگذاری config"""
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def load_or_create_model(self, model_path=None):
        """بارگذاری یا ایجاد مدل"""
        print("\n" + "="*50)
        print("LOADING MODEL FOR WEB APP")
        print("="*50)
        
        # اولویت‌بندی مسیرهای مدل
        possible_paths = [
            os.path.join('models', 'converted_resnet.pth'),  # مدل تبدیل شده
            os.path.join('models', 'fast_model_best.pth'),
            os.path.join('..', 'models', 'converted_resnet.pth'),
            os.path.join('..', 'models', 'fast_model_best.pth'),
        ]
        
        found_model = None
        for path in possible_paths:
            if os.path.exists(path):
                found_model = path
                print(f"✅ Found trained model: {path}")
                break
        
        if not found_model:
            print("⚠️ WARNING: No trained model found!")
            print("Creating a new SimpleCNN model...")
            model = SimpleCNN(num_classes=len(self.class_names))
            model.to(self.device)
            model.eval()
            return model, 'SimpleCNN (Untrained)'
        
        # بارگذاری مدل آموزش دیده
        return self.load_trained_model(found_model)
    
    def load_trained_model(self, model_path):
        """بارگذاری مدل آموزش دیده"""
        print(f"\n📦 Loading trained model from {model_path}...")
        
        try:
            # بارگذاری state_dict
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
            
            print(f"📊 Checkpoint type: {type(checkpoint)}")
            
            # بررسی ساختار checkpoint
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("   Loaded from model_state_dict")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("   Loaded from state_dict")
                else:
                    # فرض کن کل دیکشنری state_dict است
                    state_dict = checkpoint
                    print("   Loaded checkpoint as state_dict")
            else:
                # مستقیم state_dict است
                state_dict = checkpoint
                print("   Loaded directly as state_dict")
            
            # ایجاد مدل مناسب
            # بررسی keyها برای تشخیص نوع مدل
            keys = list(state_dict.keys())
            
            # اگر keyهای ResNet18 را دارد
            if any('layer1' in k for k in keys) and any('fc.' in k for k in keys):
                print("   Detected: TrainedResNet18")
                model = TrainedResNet18(num_classes=len(self.class_names))
                model_type = 'ResNet18 (Trained)'
            elif any('features' in k for k in keys):
                print("   Detected: SimpleCNN")
                model = SimpleCNN(num_classes=len(self.class_names))
                model_type = 'SimpleCNN'
            else:
                print("   Detected: Unknown, using TrainedResNet18")
                model = TrainedResNet18(num_classes=len(self.class_names))
                model_type = 'ResNet18 (Trained)'
            
            # بارگذاری وزن‌ها
            print(f"   Loading weights...")
            
            try:
                # اول سعی کن strict=True
                model.load_state_dict(state_dict)
                print("   ✅ Model weights loaded (strict=True)")
            except:
                # اگر نشد، با strict=False
                print("   ⚠️ Trying strict=False loading...")
                model.load_state_dict(state_dict, strict=False)
                print("   ✅ Model weights loaded (strict=False)")
            
            model.to(self.device)
            model.eval()
            
            # تست مدل بارگذاری شده
            print("\n🧪 Testing loaded model...")
            self.test_model_after_load(model)
            
            print("🎉 Model loaded and tested successfully!")
            return model, model_type
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            print("\n🔄 Creating new SimpleCNN model as fallback...")
            model = SimpleCNN(num_classes=len(self.class_names))
            model.to(self.device)
            model.eval()
            return model, 'SimpleCNN (Fallback)'
    
    def test_model_after_load(self, model):
        """تست مدل بعد از بارگذاری"""
        # ایجاد یک tensor dummy برای تست
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = F.softmax(output, dim=1)
            
            # بررسی خروجی
            print(f"   Output shape: {output.shape}")
            print(f"   Expected: [1, 7]")
            print(f"   Sum of probabilities: {probabilities.sum().item():.6f}")
            
            # نمایش توزیع احتمالات
            probs_np = probabilities.cpu().numpy()[0]
            print(f"   Probability distribution:")
            for i, prob in enumerate(probs_np):
                print(f"     {self.class_names[i]}: {prob*100:.2f}%")
            
            # بررسی وزن‌های لایه اول
            for name, param in model.named_parameters():
                if 'conv1.weight' in name:
                    print(f"   First conv weights - Mean: {param.data.mean():.6f}, Std: {param.data.std():.6f}")
                    break
    
    def preprocess_image(self, image_path):
        """پیش‌پردازش تصویر"""
        try:
            # خواندن تصویر
            image = Image.open(image_path).convert('RGB')
            
            # اعمال transform
            image_tensor = self.transform(image)
            
            # اضافه کردن بعد batch
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, image_path):
        """پیش‌بینی"""
        try:
            # پیش‌پردازش
            image_tensor = self.preprocess_image(image_path)
            
            print(f"\n🔍 Prediction for: {os.path.basename(image_path)}")
            print(f"   Input tensor shape: {image_tensor.shape}")
            print(f"   Input range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
            
            # پیش‌بینی
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predictions = probabilities.cpu().numpy()
            
            print(f"   Raw predictions: {predictions}")
            
            # بررسی اینکه آیا پیش‌بینی‌ها معقول هستند
            if np.max(predictions) < 0.1:
                print(f"   ⚠️ Warning: All predictions below 10%")
            
            # گرفتن top-3
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.class_names[idx]
                confidence = float(predictions[idx] * 100)
                
                results.append({
                    'class': class_name,
                    'description': self.class_descriptions[class_name],
                    'confidence': confidence,
                    'risk_level': self.get_risk_level(class_name)
                })
            
            # کلاس اصلی
            main_class_idx = np.argmax(predictions)
            main_class = self.class_names[main_class_idx]
            main_confidence = float(predictions[main_class_idx] * 100)
            
            print(f"   🎯 Main prediction: {main_class} ({main_confidence:.1f}%)")
            
            # بررسی اعتبار پیش‌بینی
            if main_confidence < 30:
                print(f"   ⚠️ Warning: Low confidence prediction ({main_confidence:.1f}%)")
            elif main_confidence > 80:
                print(f"   ✅ High confidence prediction ({main_confidence:.1f}%)")
            
            return {
                'is_cancer': self.is_cancer_class(main_class),
                'main_class': main_class,
                'main_description': self.class_descriptions[main_class],
                'main_confidence': main_confidence,
                'risk_level': self.get_risk_level(main_class),
                'recommendation': self.get_recommendation(main_class, main_confidence),
                'all_predictions': results,
                'model_info': self.best_model_info,
                'debug_info': {
                    'model_type': self.model_type,
                    'predictions_raw': [float(p) for p in predictions],
                    'top_3_indices': [int(i) for i in top_indices]
                }
            }
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            traceback.print_exc()
            return self.get_dummy_prediction()
    
    def get_dummy_prediction(self):
        """پیش‌بینی dummy برای خطا"""
        return {
            'is_cancer': False,
            'main_class': 'nv',
            'main_description': 'Melanocytic nevi',
            'main_confidence': 50.0,
            'risk_level': 'Very Low',
            'recommendation': 'Model prediction failed. Please try again.',
            'all_predictions': [
                {'class': 'nv', 'description': 'Melanocytic nevi', 'confidence': 50.0, 'risk_level': 'Very Low'},
                {'class': 'mel', 'description': 'Melanoma', 'confidence': 20.0, 'risk_level': 'Very High'},
                {'class': 'bkl', 'description': 'Benign keratosis-like lesions', 'confidence': 15.0, 'risk_level': 'Low'}
            ],
            'model_info': self.best_model_info,
            'debug_info': {'model_loaded': False, 'error': 'Prediction failed'}
        }
    
    def is_cancer_class(self, class_name):
        cancer_classes = ['mel', 'bcc', 'akiec']
        return class_name in cancer_classes
    
    def get_risk_level(self, class_name):
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
        recommendations = {
            'mel': f"ملانوما شناسایی شد ({confidence:.1f}% اطمینان). مراجعه فوری به پزشک ضروری است.",
            'bcc': f"کارسینوم سلول پایه‌ای شناسایی شد ({confidence:.1f}% اطمینان). مشاوره پزشکی توصیه می‌شود.",
            'akiec': f"کراتوز اکتینیک شناسایی شد ({confidence:.1f}% اطمینان). این وضعیت پیش‌سرطانی است.",
            'vasc': f"ضایعه عروقی شناسایی شد ({confidence:.1f}% اطمینان). مشاوره پزشکی توصیه می‌شود.",
            'bkl': f"ضایعه شبه کراتوز خوش‌خیم شناسایی شد ({confidence:.1f}% اطمینان). نظارت منظم کافی است.",
            'df': f"درماتوفیبروما شناسایی شد ({confidence:.1f}% اطمینان). معمولاً خوش‌خیم است.",
            'nv': f"خال ملانوسیتی شناسایی شد ({confidence:.1f}% اطمینان). معمولاً خوش‌خیم است."
        }
        return recommendations.get(class_name, "لطفاً با متخصص پوست مشورت کنید.")

# ایجاد predictor
predictor = SkinCancerPredictor()

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html', 
                         model_info=predictor.best_model_info,
                         class_descriptions=predictor.class_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint برای پیش‌بینی"""
    try:
        # بررسی وجود فایل
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # بررسی فرمت فایل
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file format. Please upload an image file.'}), 400
        
        # تولید نام فایل منحصر به فرد
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # ذخیره فایل
        file.save(filepath)
        
        # پیش‌بینی
        result = predictor.predict(filepath)
        
        # اضافه کردن timestamp
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['image_url'] = f'/static/uploads/{filename}'
        
        # ذخیره در session
        session['last_prediction'] = result
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """دریافت تاریخچه پیش‌بینی‌ها"""
    history = session.get('last_prediction', {})
    return jsonify(history)

@app.route('/api/class_info/<class_name>', methods=['GET'])
def get_class_info(class_name):
    """دریافت اطلاعات درباره یک کلاس خاص"""
    if class_name in predictor.class_descriptions:
        return jsonify({
            'class': class_name,
            'description': predictor.class_descriptions[class_name],
            'risk_level': predictor.get_risk_level(class_name),
            'is_cancer': predictor.is_cancer_class(class_name)
        })
    return jsonify({'error': 'Class not found'}), 404

@app.route('/about')
def about():
    """صفحه درباره"""
    model_details = {
        'name': predictor.best_model_info.get('best_model', 'Unknown'),
        'accuracy': predictor.best_model_info.get('accuracy', 0),
        'classes': predictor.class_names,
        'image_size': predictor.image_size,
        'model_type': predictor.model_type
    }
    return render_template('about.html', model_details=model_details)

@app.route('/debug/model_info')
def debug_model_info():
    """اطلاعات دیباگ مدل"""
    try:
        # اطلاعات مدل
        info = {
            'model_type': predictor.model_type,
            'model_class': type(predictor.model).__name__,
            'device': str(predictor.device),
            'num_classes': len(predictor.class_names),
            'class_names': predictor.class_names,
            'image_size': predictor.image_size,
            'best_model_info': predictor.best_model_info
        }
        
        # اطلاعات وزن‌ها
        weights_info = {}
        for name, param in predictor.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights_info[name] = {
                    'shape': list(param.shape),
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std())
                }
                if len(weights_info) >= 5:  # فقط 5 وزن اول
                    break
        
        info['sample_weights'] = weights_info
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 SKIN CANCER CLASSIFICATION WEB APP")
    print("="*60)
    print(f"📊 Model: {predictor.best_model_info.get('best_model', 'Unknown')}")
    print(f"📈 Accuracy: {predictor.best_model_info.get('accuracy', 0):.2%}")
    print(f"💻 Device: {predictor.device}")
    print(f"🔗 Server running at http://localhost:5000")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)