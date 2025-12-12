!pip uninstall -y transformers tokenizers torchvision trl
!pip install -U \
    transformers==4.54.1 \
    tokenizers==0.21.1 \
    datasets==3.5.0 \
    sacrebleu==2.5.1 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    tqdm==4.66.5 \
    peft==0.13.2 \
    bitsandbytes \
    accelerate
!pip install trl==0.25.0 #0.12.1
#!pip install git+https://github.com/Unbabel/COMET.git
!pip install evaluate
