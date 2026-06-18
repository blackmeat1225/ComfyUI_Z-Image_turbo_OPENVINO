import os
import torch
import numpy as np
import folder_paths
from PIL import Image
from optimum.intel import OVZImagePipeline
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
import openvino as ov


# =========================================================
# 節點一：🧠 Florence-2 Vision Director
# =========================================================
class Florence2VisionDirector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.current_model_id = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_image": ("IMAGE",),
                "character_image": ("IMAGE",),
                "model_id": (
                    ["microsoft/Florence-2-base", "microsoft/Florence-2-large"],
                    {"default": "microsoft/Florence-2-base"}
                ),
                "task": ([
                    "<CAPTION>",
                    "<DETAILED_CAPTION>",
                    "<MORE_DETAILED_CAPTION>",
                ], {"default": "<DETAILED_CAPTION>"}),
                "style": ([
                    "Realistic Photo",
                    "3D Render",
                    "Architectural Drawing",
                    "Oil Painting",
                    "Banana Style",
                    "None"
                ], {"default": "Realistic Photo"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Z-Image-Turbo-byKAO"

    def analyze(self, pose_image, character_image, model_id, task, style, **kwargs):
        if self.model is None or self.current_model_id != model_id:
            print(f"--- [Florence-2] 載入模型: {model_id} ---")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config._attn_implementation = "eager"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, config=config, trust_remote_code=True
            )
            self.model = model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.current_model_id = model_id
            print(f"--- [Florence-2] 載入完成 ---")

        def get_desc(img_tensor):
            pil_img = Image.fromarray(
                np.clip(255. * img_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGB")
            inputs = self.processor(
                text=task, images=pil_img, return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                use_cache=False
            )
            results = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            parsed = self.processor.post_process_generation(
                results, task=task, image_size=pil_img.size
            )
            return parsed[task]

        print("[Florence-2] 分析姿勢圖...")
        desc_pose = get_desc(pose_image)
        print(f"[Florence-2] 姿勢描述: {desc_pose}")

        print("[Florence-2] 分析角色圖...")
        desc_char = get_desc(character_image)
        print(f"[Florence-2] 角色描述: {desc_char}")

        style_map = {
            "Realistic Photo":       "raw photo, 8k uhd, photorealistic, cinematic lighting",
            "3D Render":             "octane render, unreal engine 5, stylized 3d, soft shadows",
            "Architectural Drawing": "architectural blueprint, technical drawing, clean lines, white background",
            "Oil Painting":          "textured brush strokes, oil on canvas, artistic masterpiece",
            "Banana Style":          "yellow and vibrant pop art style, vibrant yellow, high saturation",
            "None":                  ""
        }
        style_keywords = style_map.get(style, "")

        noise_words = [
            "made of multiple lines", "different colors",
            "green, blue, yellow, and red", "flower-like structure",
            "simple and cartoon-like", "cartoon-like", "stick figure",
            "colorful lines", "skeletal"
        ]
        cleaned_pose = desc_pose
        for word in noise_words:
            cleaned_pose = cleaned_pose.replace(word, "").strip()

        positive = (
            f"The subject has the following appearance: {desc_char}. "
            f"The subject is performing this exact body pose and action: {cleaned_pose}. "
            f"Do NOT draw stick figures or skeletal diagrams. "
            f"No color stick."
            f"Render a realistic human replicating the pose. "
            f"{style_keywords}"
        )
        negative = (
            "blurry, low quality, distorted, bad anatomy, "
            "(colorful lines:1.3), stick figure, skeleton, wire frame, "
            "sketches, diagram, low resolution, watermark"
        )

        print("[Florence-2] ✅ Prompt 生成完成")
        return (positive, negative)


# =========================================================
# 節點二：⚡ Z-Image Turbo (OpenVINO) + Denoise 控制
# =========================================================
class ZImageTurboOpenVINO:
    def __init__(self):
        self.pipe = None
        self.current_model = ""
        self.current_device = ""

    @classmethod
    def INPUT_TYPES(s):
        try:
            core = ov.Core()
            ov_devices = core.available_devices
        except Exception:
            ov_devices = ["CPU"]

        device_list = []
        if "GPU" in ov_devices:
            device_list.append("GPU")
        device_list.append("CPU")

        return {
            "required": {
                "model_name":     ("STRING", {"default": "Z_image_turbo_OPENVINO"}),
                "positive_cond":  ("STRING", {"forceInput": True}),
                "negative_cond":  ("STRING", {"forceInput": True}),
                "width":          ("INT",    {"default": 512,  "min": 256,  "max": 1024, "step": 64}),
                "height":         ("INT",    {"default": 512,  "min": 256,  "max": 1024, "step": 64}),
                "steps":          ("INT",    {"default": 7,    "min": 1,    "max": 25}),
                "cfg":            ("FLOAT",  {"default": 0.0,  "min": 0.0,  "max": 5.0,  "step": 0.1}),
                "seed":           ("INT",    {"default": 0,    "min": 0,    "max": 0xffffffffffffffff}),
                "device":         (device_list, {"default": device_list[0]}),

                # ✅ Denoise 控制
                # 1.0 = 完全從噪音生成 (txt2img)
                # 0.0~0.99 = 保留部分參考圖結構 (img2img)
                "denoise":        ("FLOAT",  {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
            },
            "optional": {
                # 當 denoise < 1.0 時需要提供參考圖，否則忽略
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo-byKAO"

    def load_model(self, model_name, device):
        load_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
        if not os.path.exists(load_path):
            load_path = os.path.join(folder_paths.models_dir, model_name)
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"[Z-Turbo] 找不到模型: {model_name}\n"
                f"請確認模型放在:\n"
                f"  {os.path.join(folder_paths.models_dir, 'diffusers', model_name)}\n"
                f"  {os.path.join(folder_paths.models_dir, model_name)}"
            )

        print(f"[Z-Turbo] 載入模型: {load_path} | 設備: {device}")
        pipe = OVZImagePipeline.from_pretrained(
            load_path,
            device=device,
            compile=True
        )
        print("[Z-Turbo] ✅ 模型載入完成")
        return pipe

    def tensor_to_pil(self, img_tensor):
        """ComfyUI IMAGE tensor → PIL Image"""
        arr = np.clip(255. * img_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def generate(self, model_name, positive_cond, negative_cond,
                 width, height, steps, cfg, seed, device,
                 denoise=1.0, reference_image=None):

        pos_text = positive_cond if isinstance(positive_cond, str) else str(positive_cond)
        neg_text = negative_cond if isinstance(negative_cond, str) else str(negative_cond)

        if (self.pipe is None
                or self.current_model != model_name
                or self.current_device != device):
            self.pipe = self.load_model(model_name, device)
            self.current_model = model_name
            self.current_device = device

        generator = torch.Generator("cpu").manual_seed(seed)

        # ─────────────────────────────────────────────
        # Denoise 模式判斷
        #   denoise = 1.0  → 純文字生成 (txt2img)，忽略 reference_image
        #   denoise < 1.0  → 圖生圖 (img2img)，需要 reference_image
        # ─────────────────────────────────────────────
        # denoise 對應 diffusers 的 strength 參數：
        #   strength=1.0 → 全部步數重新生成
        #   strength=0.5 → 只走後半段步數，保留更多原圖結構
        # 實際推理步數 = int(steps * denoise)，至少保留 1 步
        # ─────────────────────────────────────────────

        if denoise >= 1.0:
            # ── txt2img 模式 ──
            print(f"[Z-Turbo] 模式: txt2img | steps={steps} cfg={cfg} seed={seed}")
            print(f"[Z-Turbo] Prompt: {pos_text[:80]}...")

            result = self.pipe(
                prompt=pos_text,
                negative_prompt=neg_text,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator
            ).images[0]

        else:
            # ── img2img 模式 ──
            if reference_image is None:
                print("[Z-Turbo] ⚠️  denoise < 1.0 但未提供 reference_image，改用 txt2img 模式")
                result = self.pipe(
                    prompt=pos_text,
                    negative_prompt=neg_text,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator
                ).images[0]
            else:
                ref_pil = self.tensor_to_pil(reference_image).resize(
                    (width, height), Image.LANCZOS
                )
                # 實際執行步數
                actual_steps = max(1, int(steps * denoise))
                print(
                    f"[Z-Turbo] 模式: img2img | "
                    f"denoise={denoise:.2f} steps={steps}→實際{actual_steps} "
                    f"cfg={cfg} seed={seed}"
                )
                print(f"[Z-Turbo] Prompt: {pos_text[:80]}...")

                result = self.pipe(
                    prompt=pos_text,
                    negative_prompt=neg_text,
                    image=ref_pil,
                    strength=denoise,           # 對應 denoise 強度
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator
                ).images[0]

        print("[Z-Turbo] ✅ 生成完成")
        img_np = np.array(result).astype(np.float32) / 255.0
        return (torch.from_numpy(img_np)[None,],)
# =========================================================
# 節點三：⚡ Z-Image Turbo (OpenVINO)-SIMPLE
# =========================================================
class ZIMT_SIMPLE:
    def __init__(self):
        self.pipe = None
        self.current_model = ""
        self.current_device = ""

    @classmethod
    def INPUT_TYPES(s):
        try:
            core = ov.Core()
            ov_devices = core.available_devices
        except Exception:
            ov_devices = ["CPU"]

        device_list = []
        if "GPU" in ov_devices:
            device_list.append("GPU")
        device_list.append("CPU")
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "一個可愛的女孩，動漫風格"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 8}),
                "steps": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "device":         (device_list, {"default": device_list[0]}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo-byKAO"

    def generate(self, prompt, width, height, steps, seed, device):
        # 只載入一次 pipeline
        if self.pipe is None:
            core = ov.Core()
            device = "GPU" if "GPU" in core.available_devices else "CPU"
            print(f"🔧 Z-Image-Turbo-ov 使用裝置: {device}")
            self.pipe = OVZImagePipeline.from_pretrained("hsuwill000/Z-Image-Turbo-ov", device=device)

        generator = torch.Generator("cpu").manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,   # 此模型建議固定 0.0
            generator=generator,
        ).images[0]

        # 轉成 ComfyUI 格式 (BHWC, float32 0~1)
        image_np = np.array(output).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)

# =========================================================
# 節點四：🔍 Prompt Preview (除錯用)
# =========================================================
class PromptPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Z-Image-Turbo-byKAO"

    def preview(self, positive_prompt, negative_prompt):
        print("\n" + "="*60)
        print("[Prompt Preview] ✅ Positive:")
        print(positive_prompt)
        print("\n[Prompt Preview] ❌ Negative:")
        print(negative_prompt)
        print("="*60 + "\n")
        return {
            "ui": {
                "text": [f"[POS]\n{positive_prompt}\n\n[NEG]\n{negative_prompt}"]
            },
            "result": (positive_prompt, negative_prompt)
        }
# =========================================================
# 節點五：⚡ ZIMT_LORAS - 支援 Transformer 選擇
# =========================================================
class ZIMT_LORAS:
    def __init__(self):
        self.pipe = None
        self.current_model = ""
        self.current_device = ""
        self.current_transformer = ""

    # 固定模型根目錄
    MODEL_ROOT = r"F:\huggingface_cache\hub\models--hsuwill000--Z-Image-Turbo-ov\snapshots\d10ba44f52e2650cac10999ef9e10014d104cf55"

    @staticmethod
    def _scan_transformers(model_root: str) -> list[str]:
        """
        自動掃描 model_root 下所有名稱包含 'transformer' 的子資料夾。
        'transformer'（預設）永遠排在第一位。
        """
        if not os.path.isdir(model_root):
            return ["transformer"]

        entries = [
            d for d in os.listdir(model_root)
            if "transformer" in d.lower()
            and os.path.isdir(os.path.join(model_root, d))
        ]

        # 確保預設 "transformer" 排第一
        default = "transformer"
        others  = sorted(e for e in entries if e != default)
        result  = ([default] if default in entries else []) + others

        return result if result else ["transformer"]

    @classmethod
    def INPUT_TYPES(s):
        try:
            core = ov.Core()
            ov_devices = core.available_devices
        except Exception:
            ov_devices = ["CPU"]

        device_list = []
        if "GPU" in ov_devices:
            device_list.append("GPU")
        device_list.append("CPU")

        # ✅ 每次開啟節點時動態掃描
        transformer_list = ZIMT_LORAS._scan_transformers(ZIMT_LORAS.MODEL_ROOT)
        print(f"[ZIMT_LORAS] 偵測到 Transformer: {transformer_list}")

        return {
            "required": {
                "prompt":      ("STRING",  {"multiline": True, "default": "一個可愛的女孩，動漫風格"}),
                "width":       ("INT",     {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "height":      ("INT",     {"default": 512, "min": 256, "max": 4096, "step": 8}),
                "steps":       ("INT",     {"default": 7,   "min": 1,   "max": 20,   "step": 1}),
                "seed":        ("INT",     {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "device":      (device_list,      {"default": device_list[0]}),
                "transformer": (transformer_list, {"default": transformer_list[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo-byKAO"

    def _need_reload(self, device, transformer):
        """判斷是否需要重新載入 pipeline"""
        return (
            self.pipe is None
            or self.current_device != device
            or self.current_transformer != transformer
        )

    def _build_pipe(self, device, transformer):
        """
        從模型快照根目錄載入 pipeline，
        並把 transformer 資料夾替換成使用者選擇的版本。
        """
        model_root = self.MODEL_ROOT
        transformer_path = os.path.join(model_root, transformer)

        if not os.path.isdir(transformer_path):
            raise FileNotFoundError(
                f"[ZIMT_LORAS] 找不到 Transformer 資料夾:\n  {transformer_path}"
            )

        print(f"[ZIMT_LORAS] 載入模型根目錄 : {model_root}")
        print(f"[ZIMT_LORAS] 使用 Transformer: {transformer}  →  {transformer_path}")
        print(f"[ZIMT_LORAS] 推理裝置        : {device}")

        # OVZImagePipeline 預設讀取根目錄下名為 "transformer" 的子資料夾。
        # 若使用者選擇非預設 transformer，我們透過暫時符號連結（或直接指定路徑）來替換。
        # 這裡採用「複製 model_index.json 指向不同 transformer 子目錄」的方式，
        # 最相容的做法是：先讀取 model_index，動態修改 transformer 路徑後載入。
        import json, shutil, tempfile

        # 讀取原始 model_index.json
        index_path = os.path.join(model_root, "model_index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        if transformer == "transformer":
            # 使用原始設定，直接從 model_root 載入
            load_path = model_root
        else:
            # 建立暫存目錄，放入修改過的 model_index.json 與 symlink
            tmp_dir = tempfile.mkdtemp(prefix="zimt_loras_")
            try:
                # 修改 index：把 transformer 欄位指向新子目錄名稱
                # OVZImagePipeline 使用 {"_class_name": ..., "_diffusers_version": ...}
                # 的格式，transformer key 通常是 "transformer"
                new_index = dict(index)
                # 不需改 key，只需確保資料夾名稱是 "transformer"
                # 做法：把選定的 transformer 資料夾 symlink 成 tmp/transformer
                dst_transformer = os.path.join(tmp_dir, "transformer")

                # Windows 需要管理員或開發者模式才能建 symlink；
                # 用 junction（目錄連結）更相容
                try:
                    os.symlink(transformer_path, dst_transformer,
                               target_is_directory=True)
                except (OSError, NotImplementedError):
                    # fallback：直接用 shutil.copytree（速度較慢但保證可用）
                    print(f"[ZIMT_LORAS] symlink 失敗，改用複製（首次較慢）...")
                    shutil.copytree(transformer_path, dst_transformer)

                # 其他必要元件建立 symlink 指回 model_root
                for sub in ["scheduler", "text_encoder", "tokenizer",
                            "vae_decoder"]:
                    src = os.path.join(model_root, sub)
                    dst = os.path.join(tmp_dir, sub)
                    if os.path.exists(src) and not os.path.exists(dst):
                        try:
                            os.symlink(src, dst, target_is_directory=True)
                        except (OSError, NotImplementedError):
                            shutil.copytree(src, dst)

                # 複製 model_index.json
                shutil.copy2(index_path, os.path.join(tmp_dir, "model_index.json"))

                load_path = tmp_dir
            except Exception as e:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise e

        pipe = OVZImagePipeline.from_pretrained(
            load_path,
            device=device,
            compile=True
        )
        print("[ZIMT_LORAS] ✅ 模型載入完成")
        return pipe

    def generate(self, prompt, width, height, steps, seed, device, transformer):
        if self._need_reload(device, transformer):
            self.pipe = self._build_pipe(device, transformer)
            self.current_device = device
            self.current_transformer = transformer

        generator = torch.Generator("cpu").manual_seed(seed)

        print(f"[ZIMT_LORAS] 生成中 | transformer={transformer} | "
              f"size={width}x{height} | steps={steps} | seed={seed}")

        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,   # 此模型固定 0.0
            generator=generator,
        ).images[0]

        image_np = np.array(output).astype(np.float32) / 255.0        image_tensor = torch.from_numpy(image_np)[None,]

        print("[ZIMT_LORAS] ✅ 生成完成")
        return (image_tensor,)
# =========================================================
# 節點註冊
# =========================================================
NODE_CLASS_MAPPINGS = {
    "Florence2VisionDirector": Florence2VisionDirector,
    "ZImageTurboOpenVINO":     ZImageTurboOpenVINO,
    "ZIMT_SIMPLE":             ZIMT_SIMPLE,
    "ZIMT_LORAS":              ZIMT_LORAS,        # ✅ 新增
    "PromptPreview":           PromptPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2VisionDirector": "🧠 Florence-2 Vision Director",
    "ZImageTurboOpenVINO":     "⚡ Z-Image Turbo (OpenVINO)",
    "ZIMT_SIMPLE":             "⚡ZIMT_SIMPLE",
    "ZIMT_LORAS":              "⚡ZIMT_LORAS",    # ✅ 新增
    "PromptPreview":           "🔍 Prompt Preview",
}
# ```

# ---

# ## Denoise 參數說明

# | denoise 值 | 模式 | 效果 |
# |---|---|---|
# | `1.0` | txt2img | 完全從噪音生成，忽略 reference_image |
# | `0.75` | img2img | 保留 25% 原圖結構，大幅改變內容 |
# | `0.5` | img2img | 保留 50% 原圖結構，適合風格轉換 |
# | `0.25` | img2img | 保留 75% 原圖結構，微調細節 |

# ## 接線方式
# ```
# txt2img 模式（denoise=1.0）:
# [Load Image 姿勢] ──► [🧠 Florence-2] ──positive/negative──► [⚡ Z-Image Turbo] ──► [Preview]
# [Load Image 角色] ──►                    reference_image 不接

# img2img 模式（denoise<1.0）:
# [Load Image 姿勢] ──► [🧠 Florence-2] ──positive/negative──► [⚡ Z-Image Turbo] ──► [Preview]
# [Load Image 角色] ──►                                          ▲
# [Load Image 參考] ────────────────── reference_image ──────────┘
