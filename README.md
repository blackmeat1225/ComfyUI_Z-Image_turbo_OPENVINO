<div align="center">
(ReadMe →Claude)
<img src="https://github.com/user-attachments/assets/710daad0-e456-4198-9dc3-6506fd410e8d" alt="ComfyUI Z-Image Turbo OpenVINO Banner" width="100%"/>

# ⚡ ComfyUI\_Z-Image\_turbo\_OPENVINO

### Text-to-Image at Blazing Speed — Powered by Intel OpenVINO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue?logo=github)](https://github.com/comfyanonymous/ComfyUI)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2025.4-0071C5?logo=intel)](https://github.com/openvinotoolkit/openvino)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)

</div>

---

## 🌟 簡介 / Introduction

> **為 Intel 內顯玩家量身打造的 ComfyUI 加速節點！**  
> A custom ComfyUI node that unleashes Intel CPU + iGPU power via OpenVINO.

這個節點由 **Claude / DeepSeek / Gemini AI** 協助開發，讓配備 Intel CPU（如 i5-1135G7）的使用者，能在 ComfyUI 中透過 Intel 內顯（iGPU）執行 **Z-Image Turbo OV Model**，獲得驚人的生成速度。

| 方案 | 解析度 | Steps | 時間 |
|------|--------|-------|------|
| 🐢 Z-Image Turbo GGUF Q2 | 512×512 | 7 | ~1500 秒 |
| ⚡ **本節點 (OpenVINO)** | 512×512 | 7 | **~90 秒 (約快 20 倍！)** |

> 💡 **模型作者（非本人）：** [hsuwill000 — Z-Image-Turbo-ov](https://huggingface.co/hsuwill000/Z-Image-Turbo-ov)

---

## 🚀 安裝 / Installation

> ⚠️ **強烈建議使用全新的虛擬環境 (VENV)**

### Step 1 — 取得專案 / Clone the repo

```powershell
cd /ComfyUI/custom_nodes/
git clone https://github.com/blackmeat1225/ComfyUI_Z-Image_turbo_OPENVINO.git
cd /ComfyUI/custom_nodes/ComfyUI_Z-Image_turbo_OPENVINO
```
model_path=D:\ComfyUI\models\Z_image_Turbo
### Step 2 — 安裝依賴 / Install requirements

> ⚠️ **必須先執行 uninstall 步驟！**

```powershell
python.exe -m pip install --upgrade pip
pip3 uninstall -y optimum transformers optimum-intel diffusers

pip3 install git+https://github.com/huggingface/diffusers
pip3 install git+https://github.com/openvino-dev-samples/optimum-intel.git@zimage
pip3 install nncf
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install openvino==2025.4
```

---

## 🎨 使用方式 / How to Use

啟動 ComfyUI 後，在左側 Custom Nodes 面板中找到本節點：

<div align="center">
<img src="https://github.com/user-attachments/assets/cb6e847f-9f85-46a5-b243-3a03ce1e409c" width="480" alt="節點面板"/>
</div>

---

### 🟢 Method 1 — Text to Image（純文字生成）

*由 Claude AI 設計*

直接用文字描述你想要的畫面：

```
The subject in the image is an anime-style character with long, straight silver hair.
She is wearing a black, form-fitting outfit that includes a lace bra and a matching skirt.
The outfit has a glossy finish, giving it a sleek and elegant look...
```

<div align="center">
<img src="https://github.com/user-attachments/assets/a90905be-3ec4-47a8-953c-a8aca4191199" width="100%" alt="Method 1 示例"/>
</div>

---

### 🟡 Method 2 — Florence2 Combo（偽 ControlNet）

*由 Gemini AI 設計*

> 📦 需要額外安裝：`pip install timm`

看兩張圖，比較久，但效果穩定，"偶爾沒有"火柴人殘影。

<div align="center">
<img src="https://github.com/user-attachments/assets/b2d22fcd-e312-4d77-b650-0afc3c3a4a5b" width="100%" alt="Method 2 示例"/>
</div>

---

### 🔵 Method 3 — Qwen2.5-VL-7B（偽 Image to Image）

*由 DeepSeek AI 設計*

👯

<div align="center">
<img src="https://github.com/user-attachments/assets/cfe5c726-ba71-48ba-9f79-1e8b8d455021" width="100%" alt="Method 3 示例"/>
</div>

---

### 🟣 Method 4 — Qwen2.5-VL-7B（偽 ControlNet）
一眼看一個 → 把兩張圖先合併成一張再看，速度較快，但偶爾會出雙胞胎 
*由 DeepSeek AI 設計*

<div align="center">
<img src="https://github.com/user-attachments/assets/66b1850f-21f8-400c-aabf-976c3fe831c0" width="100%" alt="Method 4 示例"/>
</div>

### 🟣 Method 5 — Loras Merged 

Step1: use  ai-toolkits  Traning your loras model (need GPU )<BR>
https://github.com/ostris/ai-toolkit<BR>
<img width="1644" height="544" alt="image" src="https://github.com/user-attachments/assets/de5aeb60-bd68-49da-b16b-c1ae743c35b3" /><B>
IF you dont want training your own loras you can down load others loras

Step2:merge your loras to Z Image turbo<BR>
https://huggingface.co/Tongyi-MAI/Z-Image-Turbo<BR>


SNAPSHOT is where your Z image turbo you put<BR>
LORA_PATH is where you traned lora model<BR>
OUT_TRANSFORMER is where the lora merged model<BR>
make a python file as below if named lora_merge.py :<BR>
```VS CODE
import os
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from safetensors.torch import load_file
import torch

SNAPSHOT = r'D:\huggingface_cache\models--Tongyi-MAI--Z-Image-Turbo\snapshots\f332072aa78be7aecdf3ee76d5c247082da564a6'
LORA_PATH = r'D:\ComfyUI\models\loras\CCY_CP_automagic.safetensors'
OUT_TRANSFORMER = r'D:\Z-Image-CCY-diffusers\transformer'
os.makedirs(OUT_TRANSFORMER, exist_ok=True)

print('載入模型架構...')
model = ZImageTransformer2DModel.from_pretrained(
    SNAPSHOT + r'\transformer',
    torch_dtype=torch.bfloat16,
)

print('載入 LoRA...')
lora = load_file(LORA_PATH)

# ── 印出前幾個 key，確認格式 ──────────────────────────
print('LoRA key 範例：')
for k in list(lora.keys())[:10]:
    print(' ', k)

# ── 統一移除常見前綴 ──────────────────────────────────
def strip_prefix(k):
    for prefix in ['diffusion_model.', 'transformer.', 'model.']:
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k

lora_clean = {strip_prefix(k): v.float() for k, v in lora.items()}

# ── 收集所有 base 名稱 ────────────────────────────────
lora_bases = {k.replace('.lora_A.weight', '')
            for k in lora_clean if '.lora_A.weight' in k}

# ── 讀取 alpha（支援 per-layer 或全局）────────────────
def get_alpha(lora_clean, base):
    # 嘗試 per-layer alpha key（ComfyUI 常見格式）
    alpha_key = f'{base}.alpha'
    if alpha_key in lora_clean:
        return lora_clean[alpha_key].item()
    return None  # fallback: 用 rank 當 alpha（scale=1）

# ── Merge ─────────────────────────────────────────────
state = {k: v.clone() for k, v in model.state_dict().items()}
merged_count = 0
skipped = []

for base in sorted(lora_bases):
    A = lora_clean.get(f'{base}.lora_A.weight')  # shape: [rank, in]
    B = lora_clean.get(f'{base}.lora_B.weight')  # shape: [out, rank]
    if A is None or B is None:
        continue

    rank = A.shape[0]
    alpha = get_alpha(lora_clean, base)
    scale = (alpha / rank) if alpha is not None else 1.0  # alpha=None → scale=1

    delta = (B @ A).bfloat16() * scale

    target_key = base + '.weight'
    if target_key in state:
        if state[target_key].shape == delta.shape:
            state[target_key] += delta
            merged_count += 1
        else:
            skipped.append(f'shape mismatch: {target_key} '
                        f'{state[target_key].shape} vs {delta.shape}')
    else:
        skipped.append(f'key not found: {target_key}')

print(f'Merged {merged_count} 層')
if skipped:
    print(f'跳過 {len(skipped)} 層：')
    for s in skipped[:10]:
        print(' ', s)

model.load_state_dict(state, strict=False)

print('儲存 merged transformer...')
model.save_pretrained(OUT_TRANSFORMER)
print('完成！')
```


than RUN : python lora_merge.py <br>


Step3  Transform to openvino<BR>
you will get the lora merged model in OUT_TRANSFORMER<br>
than run<br>
optimum-cli export openvino   --model D:\Z-Image-CCY-diffusers   --task text-to-image   --library diffusers   D:\Z-Image-CCY-ov  --weight-format int4   --group-size 64   --ratio 1.0 <br>
you will get a openvino model in D:\Z-Image-CCY-ov <br>
Step4 <br>
before rename the transfomer folder in  D:\Z-Image-CCY-ov  to 1980_transformer(whatever you want)<br>
than put inot your ComfyUI  model folder  <BR>

Step5 :<BR>
select your own transformer Run youw own loras by openvino<BR>
Z-image-turbo ORIGINAL <BR>
<img width="1056" height="603" alt="image" src="https://github.com/user-attachments/assets/a1624674-226e-4b29-844f-f3881cc04fd6" /><BR>

Z-image-turbo merge loars <BR>

<img width="1120" height="641" alt="image" src="https://github.com/user-attachments/assets/b2323eee-8285-4ed4-a08b-8f626ec2c6b8" /><BR>

## 📋 節點說明 / Node Reference

| 節點名稱 | 功能 | 備註 |
|----------|------|------|
| `ZITNT_SIMPLE` | 純文字生成圖片 | 🌟 主節點，推薦使用 |
| Florence2 Combo | 圖片引導生成 | 需安裝 `timm` |
| Qwen2.5-VL 單張 | Image to Image | 偶有雙胞胎問題 |
| Qwen2.5-VL 雙張 | ControlNet 風格 | 速度稍慢但穩定 |

> 其他節點為 AI 協助生成，`ZITNT_SIMPLE` 是作者最推薦的核心節點，其他節點可自行探索。

---

## 🙏 致謝 / Credits

- 🧠 **模型作者：** [hsuwill000](https://huggingface.co/hsuwill000/Z-Image-Turbo-ov) — Z-Image-Turbo-ov
- 🤖 **AI 協作：** Claude、DeepSeek、Gemini
- ⚙️ **核心技術：** [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) · [ComfyUI](https://github.com/comfyanonymous/ComfyUI) · [HuggingFace Diffusers](https://github.com/huggingface/diffusers)

---

<div align="center">

**希望讓所有 Intel 內顯玩家都能享受 AI 繪圖的樂趣！** 🎉  
*Made with ❤️ for the Intel iGPU community*

</div>
