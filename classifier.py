import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as transforms

import argparse
import numpy as np
import os
import shutil
from tqdm import tqdm
import time


from datasets import load_dataset
from PIL import Image
from pathlib import Path

class BlurClassifier(nn.Module):
    def __init__(self):
        super(BlurClassifier, self).__init__()
        
        # using a smaller network since we're just detecting blur
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))  
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class StreamingBlurDataset(IterableDataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset("acozma/imagenet-1k-rand_blur", split=split, streaming=True)
        self.transform = transform
        
    def __iter__(self):
        for item in self.dataset:
            orig_image = item['image']
            blur_image = item['conditioning_image']
            
            if not isinstance(orig_image, Image.Image):
                orig_image = Image.fromarray(orig_image)
            if not isinstance(blur_image, Image.Image):
                blur_image = Image.fromarray(blur_image)
                
            if self.transform:
                orig_image = self.transform(orig_image)
                blur_image = self.transform(blur_image)
                
            # stack both images
            images = torch.stack([orig_image, blur_image])
            labels = torch.tensor([0, 1], dtype=torch.float32)
            
            yield images, labels


def train_model(model, train_loader, max_steps=1000, device='cuda'):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(total=max_steps)
    start_time = time.time()
    last_time = start_time
    
    for step, (images, labels) in enumerate(train_loader, 1):
        b, pair, c, h, w = images.shape
        images = images.view(-1, c, h, w)
        labels = labels.view(-1)
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if step % 10 == 0:
            torch.cuda.empty_cache()

            current_time = time.time()
            seconds_per_iter = (current_time - last_time) / 10
            last_time = current_time
            
            pbar.set_postfix({
                'loss': f'{running_loss/10:.4f}',
                'acc': f'{100*correct/total:.2f}%',
                'sec/iter': f'{seconds_per_iter:.3f}'
            })
            running_loss = 0.0
            correct = 0
            total = 0
            
        pbar.update(1)
        if step >= max_steps:
            break
            
    pbar.close()

def predict_blur(model, image_path, transform, device, threshold=0.8):
    model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob = model(image).item()
        
        return prob > threshold, prob
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

def process_directory(model, source_dir, output_dir, device, transform):
    blurred_dir = os.path.join(output_dir, 'blurred')
    normal_dir = os.path.join(output_dir, 'normal')
    os.makedirs(blurred_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    
    # process all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    for file_path in Path(source_dir).rglob('*'):
        if file_path.suffix in image_extensions:
            is_blurry, confidence = predict_blur(model, str(file_path), transform, device)
            
            if is_blurry is not None:  
                dest_dir = blurred_dir if is_blurry else normal_dir
                dest_path = os.path.join(dest_dir, file_path.name)
                
                # handle duplicate filenames
                if os.path.exists(dest_path):
                    base = os.path.splitext(file_path.name)[0]
                    ext = file_path.suffix
                    counter = 1
                    while os.path.exists(dest_path):
                        new_name = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(dest_dir, new_name)
                        counter += 1
                
                print(f"{'Blurry' if is_blurry else 'Normal'} ({confidence:.2f}): {file_path.name}")
                shutil.copy2(str(file_path), dest_path)

def main():
    parser = argparse.ArgumentParser(description='Blur Detection Model')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Training
    ########################
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    
    # Inference 
    ########################
    inf_parser = subparsers.add_parser('inf', help='Run inference on directory')
    inf_parser.add_argument('--source', required=True, help='Source directory containing images')
    inf_parser.add_argument('--output', required=True, help='Output directory for sorted images')
    inf_parser.add_argument('--model', default='blur_classifier.pth', help='Path to model weights')
    inf_parser.add_argument('--threshold', default=0.8, help='Blur detection threshold')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # imagenet values are fine for this
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    if args.mode == 'train':
                
        train_data = StreamingBlurDataset(transform=transform)
        train_loader = DataLoader(train_data, batch_size=16)
        
        model = BlurClassifier()
        train_model(model, train_loader, max_steps=args.steps, device=device)
        
        torch.save(model.state_dict(), 'blur_classifier.pth')
        print("Model saved to blur_classifier.pth")
        
    elif args.mode == 'inf':
        model = BlurClassifier()
        model.load_state_dict(torch.load(args.model, weights_only=True))
        model = model.to(device)
        
        process_directory(model, args.source, args.output, device, transform)
        print(f"\nProcessing complete. Check {args.output} for results.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
