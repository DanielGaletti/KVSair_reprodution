import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import os
from tqdm import tqdm
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4 
BATCH_SIZE = 8
NUM_EPOCHS = 150 
NUM_WORKERS = 2 
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
PIN_MEMORY = True
PREDICTION_THRESHOLD = 0.5

NADAM_BETAS = (0.9, 0.999)

DATA_PATH = "kvasir_dataset/"
IMAGE_DIR = os.path.join(DATA_PATH, "images")
MASK_DIR = os.path.join(DATA_PATH, "masks")

TEST_SIZE = 0.2
VALID_SIZE = 0.5

RANDOM_SEED = 42

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # ReLU como não-linearidade [cite: 186]
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.channel_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.channel_shortcut:
            identity = self.channel_shortcut(x)
        out += identity
        return self.relu(out)

class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()

        self.encoder1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = ResidualBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1_out = self.encoder1(x)
        e2_in = self.pool1(e1_out)
        e2_out = self.encoder2(e2_in)
        e3_in = self.pool2(e2_out)
        e3_out = self.encoder3(e3_in)
        e4_in = self.pool3(e3_out)
        e4_out = self.encoder4(e4_in)
        
        b_in = self.pool4(e4_out)
        b_out = self.bottleneck(b_in)

        d4_in = self.upconv4(b_out)
        d4_out = self.decoder4(torch.cat([d4_in, e4_out], dim=1))
        
        d3_in = self.upconv3(d4_out)
        d3_out = self.decoder3(torch.cat([d3_in, e3_out], dim=1))

        d2_in = self.upconv2(d3_out)
        d2_out = self.decoder2(torch.cat([d2_in, e2_out], dim=1))
        
        d1_in = self.upconv1(d2_out)
        d1_out = self.decoder1(torch.cat([d1_in, e1_out], dim=1))
        
        out = self.out_conv(d1_out)
        out = self.sigmoid(out) # Sigmoid para ter uma saída entre 0 e 1

        return out

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > PREDICTION_THRESHOLD).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def iou_score(preds, targets, smooth=1e-6):
    preds = (preds > PREDICTION_THRESHOLD).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255.0] = 1.0 # Normaliza a máscara para 0 e 1
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Training Epoch")
    
    total_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def check_metrics(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_sum = 0
    iou_sum = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            
            # Calcular Dice e IoU
            dice_sum += dice_coefficient(preds, y)
            iou_sum += iou_score(preds, y)

    model.train()
    
    avg_dice = dice_sum / len(loader)
    avg_iou = iou_sum / len(loader)
    
    print(f"Dice coefficient: {avg_dice:.4f}")
    print(f"Mean IoU: {avg_iou:.4f}")

    return avg_dice, avg_iou

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Coletar todos os nomes de arquivos
    all_image_names = sorted(os.listdir(IMAGE_DIR))
    
    # Criar caminhos completos
    image_paths = [os.path.join(IMAGE_DIR, name) for name in all_image_names]
    mask_paths = [os.path.join(MASK_DIR, name) for name in all_image_names]

    # Dividir os dados
    # Primeiro, separamos o conjunto de teste (20% do total)
    train_val_paths, test_paths = train_test_split(
        list(zip(image_paths, mask_paths)), 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED
    )
    # Depois, dividimos o restante em treino e validação
    train_paths, val_paths = train_test_split(
        train_val_paths, 
        test_size=VALID_SIZE, 
        random_state=RANDOM_SEED
    )
    
    train_img_paths, train_mask_paths = zip(*train_paths)
    val_img_paths, val_mask_paths = zip(*val_paths)
    test_img_paths, test_mask_paths = zip(*test_paths)
    
    print(f"Tamanho do dataset: {len(image_paths)} imagens")
    print(f"Treino: {len(train_img_paths)} | Validação: {len(val_img_paths)} | Teste: {len(test_img_paths)}")
    
    # Augmentations do paper [cite: 180]
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomCrop(height=int(IMAGE_HEIGHT*0.9), width=int(IMAGE_WIDTH*0.9)),
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), # Redimensiona de volta
            A.RandomBrightnessContrast(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3), # Simula Cutout/Random Erasing
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Criar modelo, loss, otimizador
    model = ResUNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss() # O paper usou Dice Coefficient como loss function [cite: 186]
    optimizer = optim.Nadam(model.parameters(), lr=LEARNING_RATE, betas=NADAM_BETAS) # Otimizador Nadam [cite: 185]
    
    # DataLoaders
    train_dataset = KvasirDataset(train_img_paths, train_mask_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    
    val_dataset = KvasirDataset(val_img_paths, val_mask_paths, transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    
    test_dataset = KvasirDataset(test_img_paths, test_mask_paths, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    scaler = torch.cuda.amp.GradScaler()
    best_val_dice = -1.0
    
    # Loop de treinamento
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        
        # Avaliação no conjunto de validação
        print(f"\n--- Checando métricas no Epoch {epoch+1} ---")
        print("Resultados no conjunto de VALIDAÇÃO:")
        val_dice, val_iou = check_metrics(val_loader, model, device=DEVICE)
        
        # Salvar o melhor modelo
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(checkpoint, filename="best_model.pth.tar")
            
        end_time = time.time()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Duração: {end_time - start_time:.2f}s - Loss Treino: {train_loss:.4f}")
        print("-------------------------------------------\n")

    # Avaliação final no conjunto de teste com o melhor modelo
    print("\n\n--- Carregando o melhor modelo e avaliando no conjunto de TESTE ---")
    load_checkpoint(torch.load("best_model.pth.tar"), model)
    test_dice, test_iou = check_metrics(test_loader, model, device=DEVICE)
    print("\nResultados Finais no Conjunto de Teste:")
    print(f"Dice Coefficient: {test_dice:.6f}")
    print(f"Mean IoU: {test_iou:.6f}")

if __name__ == "__main__":
    main()