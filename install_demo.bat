mkdir llavamed_demo
cd llavamed_demo
@for %%P in (py python) do @%%P -m venv llava_env  && goto :done
:done
llava_env\Scripts\python -m pip install --upgrade pip
llava_env\Scripts\python -m pip install git+https://github.com/helena-intel/Llava-Med.git
llava_env\Scripts\huggingface-cli download helenai/llava-med-imf16-llmint8 --local-dir llava-med-imf16-llmint8
git clone https://github.com/LiangXin1001/LLaVA_Med_qa50_images.git data
tar -xf data\qa50_images.zip
move qa50_images data
curl -O https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/data/eval/llava_med_eval_qa50_qa.jsonl
mkdir data\eval
move llava_med_eval_qa50_qa.jsonl data\eval

echo llava_env\Scripts\huggingface-cli download helenai/llava-med-imf16-llmint8 --local-dir llava-med-imf16-llmint8 > update.bat
echo curl -O  https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/llavamed_inference_openvino.ipynb >> update.bat
echo curl -O  https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/llavamed_inference_openvino.py >> update.bat
echo curl -O  https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/app.py >> update.bat
echo curl -O  https://raw.githubusercontent.com/helena-intel/LLaVA-Med/refs/heads/main/launch_app.bat >> update.bat
call update.bat
