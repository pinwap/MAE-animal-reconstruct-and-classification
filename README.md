# MAE Animal Reconstruction and Classification

โปรเจคนี้เป็นงานสร้างภาพคืน (image reconstruction/inpainting) และจำแนกชนิดสัตว์ โดยใช้ชุดข้อมูล Animals-10

แกนหลักคือเปรียบเทียบ 2 แนวทาง reconstruction:
1. MAE (ViT-MAE)
2. U-Net (baseline)

จากนั้นใช้ผลของ MAE ไปทำ classification fine-tune และนำระบบไปต่อเป็นเว็บเดโม

## เป้าหมายงาน

1. ฝึก MAE ให้เติมภาพส่วนที่ถูกปิดเป็น patch ได้
2. ฝึก U-Net ภายใต้เงื่อนไข masking เดียวกันเพื่อเป็น baseline
3. นำ encoder จาก MAE มาใช้ต่อกับ classifier head เพื่อจำแนกสัตว์ 10 คลาส
4. เปรียบเทียบ MAE กับ U-Net ด้วย masked-MSE และตัวอย่างภาพ
5. เปิดใช้งานผ่านเว็บ (อัปโหลดรูป -> เลือก patch ที่ปิด -> ดูผล MAE/UNet + top-k prediction)

## MAE ใช้ Pretrained อย่างไร

1. งาน reconstruction ใช้สถาปัตยกรรม ViT-MAE-Base แล้วโหลด weight ที่ฝึกไว้จาก `weight/mae_reconstruction.pt`
2. งาน classification ใช้ MAE encoder ต่อกับ MLP head และโหลด/finetune จาก `weight/mae_cls_best.pth`
3. เมื่อ inference จะ reconstruct ก่อน แล้วนำผล MAE reconstruction ไปเข้า classifier

สรุป: โปรเจคนี้ใช้แนวคิด transfer learning จาก MAE ทั้งใน reconstruction และ classification finetune

## เปรียบเทียบกับ U-Net

1. MAE: ใช้กลไก mask token/patch ภายใน ViT แล้ว decoder ทำนาย patch ที่หายไป
2. U-Net: รับภาพที่ patch ที่ปิดถูกทำเป็นสีดำ แล้วเรียนรู้การเติมจากบริบทรอบข้าง
3. การประเมินหลักใช้ masked-MSE (วัดเฉพาะพื้นที่ที่ถูกปิด)
4. ค่าที่ต่ำกว่าถือว่าดีกว่าในงานสร้างภาพคืน

## โครงสร้างโปรเจค

```text
MAE-animal-reconstruct-and-classification/
├─ start_implementation.ipynb         # จุดรันหลักฝั่ง Kaggle
├─ inference.py                       # CLI inference (reconstruction + classification)
├─ INFERENCE.md                       # คู่มือ inference แบบละเอียด
├─ data/
│  └─ animals10.py                    # โหลดข้อมูล/split/transforms
├─ models/
│  └─ unet.py                         # โครงสร้าง U-Net
├─ training/
│  ├─ mae_trainer.py                  # train/eval MAE
│  ├─ unet.py                         # train/eval U-Net
│  ├─ classification.py               # finetune classifier จาก MAE encoder
│  └─ evaluation.py                   # ประเมินและเปรียบเทียบผล
├─ utils/
│  └─ common.py                       # utility กลาง (seed, checkpoint, metrics)
├─ weight/                            # น้ำหนักโมเดลที่ใช้งาน
│  ├─ mae_reconstruction.pt
│  ├─ unet_best.pt
│  └─ mae_cls_best.pth
└─ web_demo/
   ├─ apps/web/                       # Next.js frontend
   └─ services/inference/             # FastAPI inference service
```

## วิธีรัน (สรุปสั้นพร้อมใช้งาน)

### 1) รัน Inference แบบ CLI

ติดตั้งด้วย `uv` ตาม `pyproject.toml` และรันได้ทันที:

```bash
uv run inference.py --image data/dog.jpg --mask 74,75,88,89
```

ผลลัพธ์จะถูกบันทึกใน `inference_outputs/` เช่น:
1. `<stem>_masked_input.png`
2. `<stem>_mae_recon.png`
3. `<stem>_unet_recon.png`

### 2) รันฝั่ง Notebook/Kaggle

1. เปิด `start_implementation.ipynb`
2. ตั้ง dataset Animals-10 ให้ path ตรงตามที่ระบุในโน้ตบุ๊ก
3. รันจากต้นจนจบเพื่อ train/evaluate และบันทึกผล

### 3) รันเว็บเดโม (Frontend + Backend)

ที่โฟลเดอร์ `web_demo/`:

```bash
make install
```

เปิด 2 terminal:

```bash
# Terminal 1
make dev-inference   # FastAPI: http://localhost:8000

# Terminal 2
make dev-web         # Next.js: http://localhost:3000
```

จากนั้นเปิด `http://localhost:3000`

## น้ำหนักโมเดลที่ต้องมี

วางไฟล์ใน `weight/` (หรือใน `web_demo/services/inference/weights/` สำหรับเว็บเดโม):
1. `mae_reconstruction.pt`
2. `unet_best.pt`
3. `mae_cls_best.pth`
