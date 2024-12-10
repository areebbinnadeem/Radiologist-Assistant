import torch
from torch.quantization import quantize_dynamic

# Load the original model on CPU
original_model = torch.load("model/r2gen_model_Bert.pth", map_location=torch.device('cpu'))

# Apply quantization
quantized_model = quantize_dynamic(original_model, {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model, "model/r2gen_model_Bert_quantized.pth")

print("Quantized model saved successfully!")
