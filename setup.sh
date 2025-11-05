conda create -y -n rail python=3.10
conda activate rail
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# pip install flash_attn==2.7.4.post1
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
# pip install biopython==1.84
# pip install MDAnalysis==2.8.0
# pip install biotite==1.0.1
# pip install OmegaConf
pip install transformers

