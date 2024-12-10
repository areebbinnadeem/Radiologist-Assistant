import gzip
import shutil

input_file = "model/r2gen_model_Bert.pth"
compressed_file = "model/r2gen_model_Bert.pth.gz"
print("Starting compression...")
with open(input_file, 'rb') as f_in:
    with gzip.open(compressed_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Compression completed.")
print(f"Compressed model saved as: {compressed_file}")
