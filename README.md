https://www.digitalocean.com/community/tutorials/how-to-install-and-manage-supervisor-on-ubuntu-and-debian-vps

sudo apt update && sudo apt install supervisor
sudo systemctl status supervisor
sudo nano /etc/supervisor/conf.d/dl.conf

\home\ubuntu\stable-diffusion-webui\webui.sh
export COMMANDLINE_ARGS="--listen"

watch -n 1 nvidia-smi
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Create ENV
python3 -m venv venv
source venv/bin/activate
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -r requirements.txt
pip install -e .


cd ..
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ../stablediffusion

pip install --force-reinstall httpcore==0.15
pip install timm


cd /content/stablediffusion
wget https://huggingface.co/stabilityai/stable-diffusion-2-depth/blob/main/512-depth-ema.ckpt


wget https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt

mkdir midas_models
cd /content/stablediffusion/midas_models
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt



cd /home/ubuntu/stablediffusion
source /home/ubuntu/stablediffusion/venv/bin/activate

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
source venv/bin/activate
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt


ln -s sd-v1-4-full-ema.ckpt models/ldm/stable-diffusion-v1/model.ckpt 

pip install omegaconf
pip install einops
python3 scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sudo apt install python3.8-venv
sudo apt install python3-pip

cd stable-diffusion-webui
source venv/bin/activate
pip install --force-reinstall httpcore==0.15
deactivate

wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt


ln -s /home/ubuntu/stable-diffusion-webui/v1-5-pruned.ckpt models/Stable-diffusion/model.ckpt

ln -s /home/ubuntu/stable-diffusion-webui/sd-v1-4-full-ema.ckpt models/Stable-diffusion/model.ckpt 
ln -s /home/ubuntu/stable-diffusion-webui/v1-5-pruned.ckpt models/Stable-diffusion/v1-5-pruned.ckpt
ln -s /home/ubuntu/stable-diffusion-webui/v2-1_768-ema-pruned.ckpt models/Stable-diffusion/v2-1_768-ema-pruned.ckpt

v1-5-pruned.ckpt

/home/ubuntu/stable-diffusion-webui/models/Stable-diffusion

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sudo apt-get update
sudo apt install wget git python3 python3-venv python3-pip
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
cd /home/ubuntu/stable-diffusion-webui
./webui.sh


---------

demo.queue(default_enabled=True).launch(server_name="0.0.0.0", debug=True)


source venv1/bin/activate
