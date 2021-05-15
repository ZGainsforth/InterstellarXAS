conda create --name xmm python=3.8
conda deactivate
conda activate xmm

conda install numpy pandas scipy xlrd openpyxl
conda install -c plotly plotly
pip install streamlit
