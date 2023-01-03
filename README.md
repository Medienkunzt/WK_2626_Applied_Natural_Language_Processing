# WK_2626_Applied_Natural_Language_Processing

## Initiate venv
> python -m venv venv_nlp

> venv_nlp\Scripts\activate

## Conda Venv
conda create --name transformer_env

conda activate transformer_env
conda deactivate

conda install -c anaconda ipykernel
conda install -c huggingface transformers
conda install pytorch torchvision torchaudio cpuonly -c pytorch
