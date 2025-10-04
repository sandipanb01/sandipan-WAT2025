!pip uninstall -y transformers tokenizers
!pip install -U \
    transformers==4.54.1 \
    tokenizers==0.21.1 \
    datasets==3.5.0 \
    sacrebleu==2.5.1 \
    torch==2.6.0 \
    tqdm==4.66.5 \
    vllm==0.8.5.post1 \
    trl==0.12.1 \
    peft==0.13.2 \
    bitsandbytes \
    accelerate
pip install git+https://github.com/Unbabel/COMET.git
!pip install evaluate
