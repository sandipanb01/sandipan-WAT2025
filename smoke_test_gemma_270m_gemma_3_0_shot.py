import pandas as pd
from pathlib import Path

# Path to the results directory created in the previous step
results_dir = Path("./gemma3_post_ft_analysis")
files = ["analysis_english_hindi.csv", "analysis_hindi_english.csv"]

print("="*80)
print("STRING INSPECTOR: VISUALIZING MODEL OUTPUT FAILURE MODES")
print("="*80)

for file_name in files:
    file_path = results_dir / file_name
    if file_path.exists():
        df = pd.read_csv(file_path)
        direction = file_name.replace("analysis_", "").replace(".csv", "").upper()
        
        print(f"\n>>> DIRECTION: {direction}")
        # Inspecting first 5 samples
        for i, row in df.head(5).iterrows():
            print(f"\nSample {i+1}:")
            print(f"  Source: {row['source']}")
            print(f"  Ref:    {row['reference']}")
            print(f"  Pred:   {row['prediction']}")
            print(f"  LID:    {row['lid_accuracy']} | Copy Rate: {row['semantic_copy_rate']:.4f}")
        print("-" * 40)
    else:
        print(f"File {file_name} not found in {results_dir}")

print("\n" + "="*80)
