# WK_2626_Applied_Natural_Language_Processing

## Initiate venv
> python -m venv venv_nlp

> venv_nlp\Scripts\activate

## Conda Venv
conda create --name transformer_env

deactivate
conda deactivate

conda install -c anaconda ipykernel
conda install -c conda-forge ipywidgets
conda install -c huggingface transformers
conda install -c conda-forge sentencepiece
conda install pytorch torchvision torchaudio cpuonly -c pytorch
