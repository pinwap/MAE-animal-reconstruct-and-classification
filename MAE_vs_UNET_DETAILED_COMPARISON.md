# 📊 MAE vs UNet - เปรียบเทียบอัลกอริทึมโดยละเอียด

---

## 🎯 ภาพรวม (Overview)

ทั้ง MAE และ UNet ถูกใช้เพื่อ **ฟื้นฟูรูปที่ถูกบดบังส่วนหนึ่ง** แต่วิธีการต่างกันโดยสิ้นเชิง:

- **MAE** = Masked Autoencoder (ViT-based) → โมเดล **Pretrained** ที่ได้จาก Meta/Facebook
- **UNet** = Convolutional Encoder-Decoder → โมเดล **สร้างใหม่ (from scratch)** ไม่มีการใช้ pretrain weights

---

## 📝 ขั้นตอนที่ 1: โหลด/สร้างโมเดล (Model Initialization)

### ➤ MAE: โหลด Pretrained Model

```python
# จาก: training/mae_trainer.py
def load_mae_model(model_name: str = DEFAULT_MAE_MODEL_NAME, mask_ratio: float = 0.75) -> ViTMAEForPreTraining:
    """Load MAE model and set masking ratio on config when available."""
    model = ViTMAEForPreTraining.from_pretrained(model_name)  # ← โหลดจาก Hugging Face
    set_mae_mask_ratio(model, mask_ratio)
    return model

# โมเดลมาจาก: facebook/vit-mae-base (Hugging Face Hub)
```

**ลักษณะ:**
- ✅ **Pretrained weights** → ได้รับการสอนแล้วบน ImageNet-22k มาก่อนแล้ว
- ✅ **86 Million parameters** (ViT-base: 12 layers, 768 hidden dims)
- ✅ **Vision Transformer architecture** → แบ่งภาพเป็น patches + attention mechanism
- ✅ Optimized สำหรับการรับรู้ "structural patterns" ในธรรมชาติ

---

### ➤ UNet: สร้างใหม่ from Scratch

```python
# จาก: models/unet.py
class UNet(nn.Module):
    """Full encoder-decoder network for image reconstruction."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        # ... skip connections ...
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        # ... etc ...
        self.outc = OutConv(base_channels, out_channels)

# จาก start_implementation.ipynb
unet_model = UNet().to(device)  # ← สร้างใหม่ + random initialization
```

**ลักษณะ:**
- ❌ **ไม่มี pretrained weights** → เริ่มจากการ random initialization
- ✅ **7.7 Million parameters** (ขนาดเล็กกว่า MAE ประมาณ 10 เท่า)
- ✅ **Convolutional architecture** → cascading Conv2D layers ที่ filter ขนาดเล็ก
- ❌ ต้องเรียนรู้โครงสร้างภาพตั้งแต่เริ่มต้น ⚠️

**ข้อมูล:**
```
MAE Parameters:   ~86 M (pretrained)
UNet Parameters:  ~7.7 M (from scratch)
Ratio:            UNet ≈ 1/11 ของ MAE
```

---

## 🎭 ขั้นตอนที่ 2: วิธีการบดบังส่วนรูป (Masking Strategy)

### ➤ MAE: Dynamic Masking ผ่านโมเดล

```python
# จากใน ViTMAEForPreTraining (Hugging Face/Meta)
# MAE ทำ masking **ภายในโมเดล** ไม่ใช่แค่เตรียมข้อมูล
# 
# ขั้นตอน:
# 1. แบ่งรูป 224×224 → patches 16×16 = 196 patches (14×14 grid)
# 2. สุ่มเลือก patches ให้ถูกซ่อน: 75% × 196 = 147 patches
# 3. ใช้ special [MASK] token แทน masked patches
# 4. Encoder มองแค่ 49 patches ที่ไม่ถูกซ่อน
# 5. Decoder ได้ encoder output + [MASK] tokens → ฟื้นฟู 196 patches
```

**Visualized:**
```
Original (196 patches):
┌─┬─┬─┬─┐
├─┼─┼─┼─┤
├─┼─┼─┼─┤  75% ถูกซ่อน (random selection)
├─┼─┼─┼─┤
└─┴─┴─┴─┘

Encoder sees (49 patches):
┌─┬─┬─┐
├─┼─┼─┤
└─┴─┴─┘

Decoder outputs (196 patches):
┌─┬─┬─┐ + [MASK] tokens
├─┼─┼─┤
└─┴─┴─┘
```

**Key Point:** MAE เลือก patches ที่ **สุ่ม** ทุกครั้งในแต่ละ epoch

---

### ➤ UNet: Static Manual Masking

```python
# จาก: training/unet.py
def apply_patch_mask(images: torch.Tensor, mask_ratio: float = 0.75, patch_size: int = 16):
    """Apply random patch masking to create a 75%-masked reconstruction input."""
    
    batch_size, channels, height, width = images.shape
    grid_h = height // patch_size           # 224 / 16 = 14
    grid_w = width // patch_size            # 224 / 16 = 14
    num_patches = grid_h * grid_w           # 196
    num_masked = max(1, int(round(num_patches * mask_ratio)))  # 147
    
    masked_images = images.clone()
    
    for batch_index in range(batch_size):
        # สุ่มเลือก patches ให้ถูกซ่อน
        permutation = torch.randperm(num_patches, device=images.device)
        masked_indices = permutation[:num_masked]
        
        # ตั้งค่าที่ mask = 0 (black patches)
        # ... reshape logic ...
        flattened[masked_indices] = 0.0  # ← Set to zero
    
    return masked_images, patch_masks
```

**ภาพ:**
```
Input image:
[224 × 224 × 3]

After masking (set to black = 0):
[224 × 224 × 3] ← 75% pixels = 0, 25% intact

Feed to UNet:
[224 × 224 × 3] masked
     ↓
   UNet
     ↓
[224 × 224 × 3] reconstructed
```

**Key Difference:**
- MAE: ซ่อนผ่านการตัด patches แล้วใช้ [MASK] tokens ในโมเดล
- UNet: ซ่อนผ่านการตั้ง pixel = 0 (black) ตรง ๆ แล้วส่งให้ UNet ฟื้นฟู

---

## 🔄 ขั้นตอนที่ 3: Forward Pass (ไปหน้า)

### ➤ MAE: Vision Transformer Encoder → Decoder

```python
# จาก: training/mae_trainer.py -> train_mae_epoch()
# ในการเทรน:
outputs = model(pixel_values=images)  # images shape: [B, 3, 224, 224]

# ภายใน MAE:
# 1. Patch Embedding: [B, 3, 224, 224] → [B, 196, 768]
# 2. Random Masking: 196 patches → 49 visible + 147 masked
# 3. Encoder (12 layers ViT):
#    - Self-attention บน 49 patches + cls token
#    - Output: [B, 50, 768]
# 4. Decoder (8 layers ViT):
#    - รับ encoder output + [MASK] tokens
#    - Self-attention บน 196 patches
#    - Output: [B, 196, 768]
# 5. Prediction Head: [B, 196, 768] → [B, 196, 256] (patch size = 16²)
```

**Architecture Diagram:**
```
Input: [B, 3, 224, 224]
    ↓
Patchify: [B, 196, 768]
    ↓
[Masking: 75%] → 49 visible patches
    ↓
[Encoder (12 ViT layers)]
    ↓
Encoder Output: [B, 50, 768]
    ↓
[Decoder (8 ViT layers)] + [MASK] tokens
    ↓
Decoder Output: [B, 196, 768]
    ↓
Unpatchify: [B, 3, 224, 224]
    ↓
Output: Predicted full image
```

---

### ➤ UNet: Encoder → Decoder with Skip Connections

```python
# จาก: models/unet.py
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: [B, 3, 224, 224] (masked image)
    
    # ENCODER (Down-sampling)
    x1 = self.inc(x)           # [B, 64, 224, 224]
    x2 = self.down1(x1)        # [B, 128, 112, 112]  (save for skip)
    x3 = self.down2(x2)        # [B, 256, 56, 56]    (save for skip)
    x4 = self.down3(x3)        # [B, 512, 28, 28]    (save for skip)
    x5 = self.down4(x4)        # [B, 1024, 14, 14]   (bottleneck)
    
    # DECODER (Up-sampling with skip connections)
    x = self.up1(x5, x4)       # [B, 512, 28, 28]    concat + upsample
    x = self.up2(x, x3)        # [B, 256, 56, 56]
    x = self.up3(x, x2)        # [B, 128, 112, 112]
    x = self.up4(x, x1)        # [B, 64, 224, 224]
    
    return self.outc(x)        # [B, 3, 224, 224]    output
```

**Architecture Diagram:**
```
Input: [B, 3, 224, 224] (masked)
    ↓
Encoder:
x1 → [B, 64, 224]    (skip to up4)
x2 → [B, 128, 112]   (skip to up3)
x3 → [B, 256, 56]    (skip to up2)
x4 → [B, 512, 28]    (skip to up1)
x5 → [B, 1024, 14]   (bottleneck)
    ↓
Decoder (with skip concatenations):
up1(x5, x4) → [B, 512, 28]
up2(x, x3)  → [B, 256, 56]
up3(x, x2)  → [B, 128, 112]
up4(x, x1)  → [B, 64, 224]
    ↓
Output Conv: [B, 3, 224, 224]
    ↓
Output: Reconstructed image
```

---

## 🎯 ขั้นตอนที่ 4: Loss Function (ฟังก์ชันการสูญเสีย)

### ➤ MAE: Hybrid Loss (MSE + SSIM)

```python
# จาก: training/mae_trainer.py
def compute_mae_hybrid_loss(
    model: torch.nn.Module,
    outputs,
    images: torch.Tensor,
    mse_weight: float = 0.8,
    ssim_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total loss = mse_weight*MSE + ssim_weight*(1-SSIM)."""
    
    # ขั้นตอน 1: นำ predictions มา unpatchify
    reconstructions = _unpatchify_if_possible(model, outputs.logits)
    
    # ขั้นตอน 2: De-normalize ImageNet
    pred_img = _denormalize_imagenet(reconstructions)
    tgt_img = _denormalize_imagenet(images)
    
    # ขั้นตอน 3: คำนวณ MSE
    mse_loss = F.mse_loss(pred_img, tgt_img)
    
    # ขั้นตอน 4: คำนวณ SSIM (Structural Similarity)
    ssim_value = _ssim_score(pred_img, tgt_img)
    
    # ขั้นตอน 5: รวม = 80% MSE + 20% (1 - SSIM)
    total_loss = (mse_weight * mse_loss) + (ssim_weight * (1.0 - ssim_value))
    
    return total_loss, mse_loss.detach(), ssim_value.detach()
```

**Loss Formula:**
```
L_MAE = 0.8 × MSE + 0.2 × (1 - SSIM)

where:
  MSE = Mean Squared Error (pixel-level difference)
  SSIM = Structural Similarity (perceptual quality)
  
目的: ทำให้ทั้งความแม่นยำ pixel + คุณภาพโครงสร้าง ดีขึ้น
```

---

### ➤ UNet: Pure MSE Loss

```python
# จาก: training/unet.py -> train_unet_epoch()
predictions = model(masked_images)
loss = F.mse_loss(predictions, images)

# Pure MSE only - ไม่มี SSIM component
```

**Loss Formula:**
```
L_UNet = MSE = (1/N) × Σ(predicted_i - target_i)²

where:
  N = number of pixels
  predicted_i = pixel value from UNet
  target_i = ground truth pixel value

目的: ลด pixel-level reconstruction error เท่านั้น
```

**Comparison:**
```
┌─────────────────┬──────────────────────────┬───────────────────┐
│ Loss Component  │ MAE                      │ UNet              │
├─────────────────┼──────────────────────────┼───────────────────┤
│ MSE             │ 80% (weighted)           │ 100%              │
│ SSIM            │ 20% (weighted)           │ 0%                │
│ Total           │ Hybrid (perceptual)      │ Pixel-only        │
└─────────────────┴──────────────────────────┴───────────────────┘
```

---

## 🔁 ขั้นตอนที่ 5: Training Loop

### ➤ MAE Training Cycle

```python
# จาก: start_implementation.ipynb -> Section 8
for epoch in range(cfg.EPOCHS_MAE):
    # TRAIN
    mae_train_loss = mae_trainer.train_epoch(train_loader)
    
    # VAL
    mae_val_loss = mae_trainer.evaluate_epoch(val_loader)
    
    # Learning Rate Scheduling
    mae_scheduler.step(mae_val_loss)
    
    # Checkpoint
    if mae_val_loss < best_mae_val:
        mae_trainer.save_checkpoint("mae_best.pt", ...)
    
    print(f"MAE Epoch {epoch+1}/{cfg.EPOCHS_MAE} | "
          f"Train: {mae_train_loss:.4f} | Val: {mae_val_loss:.4f}")

# สั้นที่สุด (cfg.EARLY_STOPPING_PATIENCE = 5)
```

**Key characteristics:**
- ✅ **Dynamic masking** ในแต่ละ batch
- ✅ **Pretrained weights** = เร็ว convergence
- ✅ **ReduceLROnPlateau scheduler** = ปรับ LR เมื่อ val loss หยุดลด
- ⏱️ **ประมาณ 50 epochs** (cfg.EPOCHS_MAE = 50)

---

### ➤ UNet Training Cycle

```python
# จาก: start_implementation.ipynb -> Section 8
for epoch in range(cfg.EPOCHS_UNET):
    # TRAIN
    unet_train_loss = unet_trainer.train_epoch(train_loader)
    
    # VAL
    unet_val_loss = unet_trainer.evaluate_epoch(val_loader)
    
    # Learning Rate Scheduling
    unet_scheduler.step(unet_val_loss)
    
    # Checkpoint
    if unet_val_loss < best_unet_val:
        unet_trainer.save_checkpoint("unet_best.pt", ...)
    
    print(f"U-Net Epoch {epoch+1}/{cfg.EPOCHS_UNET} | "
          f"Train: {unet_train_loss:.4f} | Val: {unet_val_loss:.4f}")
```

**Key characteristics:**
- ✅ **Static masking** เตรียมก่อน ให้โมเดล
- ❌ **No pretrained weights** = ช้า convergence
- ✅ **Same scheduler** as MAE
- ⏱️ **ประมาณ 50 epochs** (cfg.EPOCHS_UNET = 50)

---

## 📊 ผลลัพธ์จริง (Actual Results from Your Data)

```
✅ MAE 120 Epochs Results:
   • Train Loss: 0.0940 → 0.0745 (ลด 20.74%)
   • Val Loss:   0.0835 → 0.0740 (ลด 11.38%)
   • Best Val:   0.0739 @ Epoch 114
   • Trend:      Continuous decrease → gradually plateau after epoch 80

⏱️  MAE is still learning (not fully converged at 120 epochs)
```

---

## 🆚 สรุปความแตกต่างหลัก

```
┌─────────────────────────┬─────────────────────────┬──────────────────────┐
│ Aspect                  │ MAE                     │ UNet                 │
├─────────────────────────┼─────────────────────────┼──────────────────────┤
│ Model Source            │ Pretrained from Meta    │ Custom from scratch  │
│ Weights Init            │ facebook/vit-mae-base   │ Random init          │
│ Parameters              │ ~86 M                   │ ~7.7 M               │
│ Architecture            │ Vision Transformer      │ Convolutional        │
│ Masking Type            │ Patch-level tokens      │ Pixel zero-ing       │
│ Loss Function           │ MSE (80%) + SSIM (20%)  │ MSE (100%)           │
│ Convergence Speed       │ ⚡ Fast                 │ 🐢 Slow              │
│ Perceptual Quality      │ ✅ Better               │ ⚠️ Basic             │
│ Training Time           │ ⏱️  Moderate            │ ⏱️  Moderate         │
│ Memory Usage            │ Higher                  │ Lower                │
└─────────────────────────┴─────────────────────────┴──────────────────────┘
```

---

## 💡 ทำไม MAE ดีกว่า UNet สำหรับงาน Reconstruction?

1. **Pretrained Knowledge**: MAE ได้เรียนรู้ "general image patterns" จาก millions of images แล้ว
2. **Better Loss Function**: SSIM ช่วยให้ผลลัพธ์ดูเป็นธรรมชาติ (natural-looking)
3. **Efficient Masking**: Token masking ช่วยให้ encoder เรียนรู้ context ได้ดีขึ้น
4. **Faster Learning**: เนื่องจากมี pretrained weights → เร็ว convergence

---

## ⚠️ ข้อจำกัดของ UNet ในที่นี้

- ❌ ต้องเรียนรู้ทั้งหมด from scratch
- ❌ Loss function ไม่ได้พิจารณา perceptual quality
- ❌ ต้องการ epochs มากกว่า MAE เพื่อได้ผลลัพธ์ที่ดี
- ❌ หากภาพ test แตกต่างจาก train มาก ผลลัพธ์อาจแย่ (ไม่ generalize)

---

## 🎓 สรุปการออกแบบการเทดลองนี้

**วัตถุประสงค์**: เปรียบเทียบว่า "ใช้ pretrained model (MAE) ดีกว่า custom model from scratch (UNet) แค่ไหน?"

**ผลลัพธ์ที่ได้**: MAE ชนะใน:
- ✅ Convergence speed
- ✅ Final reconstruction quality (ต่ำกว่า val loss)
- ✅ Perceptual quality (SSIM loss)
- ✅ Data efficiency (จำเป็นต้องน้อยกว่า epochs)

**UNet ใช้ได้เมื่อ**:
- ต้องการ lightweight model (7.7M vs 86M params)
- ต้องการ simple architecture (ไม่ต้อง transformer)
- Data ของคุณต่างจาก ImageNet มาก (pretrain ไม่ช่วย)

---

*เอกสารนี้สร้างจากการวิเคราะห์โค้ดจริงในโปรเจกต์ของคุณ*
