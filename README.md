# ComfyUI_Z-Image_turbo_OPENVINO (TEXT TO IMAGE)
![chrome_SdntWHBXma](https://github.com/user-attachments/assets/710daad0-e456-4198-9dc3-6506fd410e8d)<br>
<br>
<b>

Openvino Custom_node run in  ComfyUI , Very Very Very Fast For  INTEL CPU wiht internal GPU<Br>
這個節點是由Claud /DEEPSEEK/ GEMINI AI指導，可以讓intel CPU (i5 1135G7)在 ComfyUI 內調用， GPU  run Z-Image turbo OV Model ，<br>
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
2.install requirements  **must do uninstall**<br>
powershell<bR>
python.exe -m pip install --upgrade pip<br>
pip3 uninstall -y optimum transformers optimum-intel diffusers<br>
pip3 install git+https://github.com/huggingface/diffusers<br>
pip3 install git+https://github.com/openvino-dev-samples/optimum-intel.git@zimage<br>
pip3 install nncf<br>
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu<br>
pip3 install openvino==2025.4<br>
<BR>
<br>
模型是這位大大作的<Br>
Model **author** not me :  
https://huggingface.co/hsuwill000/Z-Image-Turbo-ov<br>
<br>
HOw to use:<BR>

turn ComfyUI & check left custom_nodes:<BR>


<img width="563" height="553" alt="image" src="https://github.com/user-attachments/assets/cb6e847f-9f85-46a5-b243-3a03ce1e409c" />
(Claude AI)
ZITNT_SIMPLE
Method1: text input
描述你想要的圖片<BR>
just scribe your image <BR>
Prompt:The subject in the image is an anime-style character with long, straight silver hair. She is wearing a black, form-fitting outfit that includes a lace bra and a matching skirt. The outfit has a glossy finish, giving it a sleek and elegant look. She is also wearing sheer black stockings. The background features a room with teal walls, a framed picture on the wall, and two side tables with lamps on them. The overall setting suggests a sophisticated and intimate atmosphere.
  The subject in the image is an anime-style character with long, straight silver hair. She is wearing a black, form-fitting outfit that includes a lace bra and a matching skirt. The outfit has a glossy finish, giving it a sleek and elegant look. She is also wearing sheer black stockings. The background features a room with teal walls, a framed picture on the wall, and two side tables with lamps on them. The overall setting suggests a sophisticated and intimate atmosphere.<BR>
<img width="1570" height="857" alt="image" src="https://github.com/user-attachments/assets/a90905be-3ec4-47a8-953c-a8aca4191199" />


Finish
the other nodes are AI support , only this node is waht i want other may be better.you can try <BR>

(GEMNINI)→看兩張圖所以比較久，但還是會有火柴人的影子。
Method2:florence2 Combo (偽Controlnet) **need pip install timm" <br>
<img width="1851" height="1480" alt="image" src="https://github.com/user-attachments/assets/b2d22fcd-e312-4d77-b650-0afc3c3a4a5b" />


(DEEPSEEK)
Method3:Qwen2.5-VL-7B-Instruct-ov-int4 ( 偽IMAGE to  IMAGE) <BR>

<img width="2137" height="1144" alt="image" src="https://github.com/user-attachments/assets/cfe5c726-ba71-48ba-9f79-1e8b8d455021" />
(DEEPSEEK)一眼看一個→把兩張先併成一張看。比較快，但……偶爾會出雙胞胎
Methdo4:Qwen2.5-VL-7B-Instruct-ov-int4(偽Controlnet)<BR>
<img width="2040" height="1625" alt="image" src="https://github.com/user-attachments/assets/66b1850f-21f8-400c-aabf-976c3fe831c0" />


