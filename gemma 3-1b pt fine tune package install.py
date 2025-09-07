# Clean slate: uninstall older conflicting versions
!pip uninstall -y transformers trl peft bitsandbytes accelerate || true

# Install latest stable versions
!pip install -U "transformers>=4.40.0" "trl>=0.9.4" "peft>=0.11.1" "accelerate>=0.30.1" bitsandbytes datasets sacrebleu sentencepiece protobuf wandb
