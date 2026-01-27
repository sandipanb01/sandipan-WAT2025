!pip uninstall -y transformers tokenizers torchvision trl
!pip install -U \
    transformers\
    tokenizers \
    datasets\
    sacrebleu\
    torch\
    torchvision\
    tqdm \
    peft\
    bitsandbytes \
    accelerate
!pip install trl
#!pip install git+https://github.com/Unbabel/COMET.git
!pip install evaluate
