# ComfyUI_Z-Image_turbo_OPENVINO
![chrome_SdntWHBXma](https://github.com/user-attachments/assets/710daad0-e456-4198-9dc3-6506fd410e8d)<br>
<br>
<b>

Openvino Custom_node run in  Comfyui , Very Very Very Fast For  INTEL CPU wiht internal GPU<Br>
這個節點是由Claud AI指導，可以讓intel CPU (i5 1135G7)在 ComfyUI 內調用， GPU  run Z-Image turbo OV Model ，<br>
這是我目前用過圖片品質好，且速度也夠快的<br>
希望讓intel 內顯的玩家也可以試試看, <br>
Z image turbo gguf Q2 跑一張512x512 step7 就要花約1500s,<br>
現在用這個node 快了約20倍，趕快來試吧<br>

Installation:<Br>
***strongly recommand to use ALL NEW VENV<br>
1.get repo<br>
powershell<br>
cd /ComfyUI/custom_nodes/<br>
git clone https://github.com/blackmeat1225/ComfyUI_Z-Image_turbo_OPENVINO.git<br>
cd  /ComfyUI/custom_nodes/ComfyUI_Z-Image_turbo_OPENVINO<Br>
<Br>
2.install requirements<br>
powershell<bR>
pip install -r requirements_OPENVINO.txt<br>
<BR>
重點：<br>
 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu<br>
<br>
模型是這位大大作的<Br>
https://huggingface.co/hsuwill000/Z-Image-Turbo-ov<br>
