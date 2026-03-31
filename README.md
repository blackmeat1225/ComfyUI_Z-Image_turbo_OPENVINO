# ComfyUI_Z-Image_turbo_OPENVINO
Openvino Custom_node run in  Comfyui , Very Very Very Fast For  INTEL CPU wiht internal GPU
這個節點，可以讓intel CPU (i5 1135G7)在 ComfyUI 內調用， GPU  run Z-Image turbo OV Model ，這是我目前用過圖片品質好，且速度也夠快的
希望讓intel 內顯的玩家也可以試試看, Z image turbo gguf Q2 跑一張512x512 step7 就要花約1500s,  現在用這個node 快了約20倍，趕快來試吧
![chrome_SdntWHBXma](https://github.com/user-attachments/assets/710daad0-e456-4198-9dc3-6506fd410e8d)
重點：
 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu

模型是這位大大作的
https://huggingface.co/hsuwill000/Z-Image-Turbo-ov
