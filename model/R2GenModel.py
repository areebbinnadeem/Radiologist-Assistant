import torch
from transformers import T5ForConditionalGeneration, BertTokenizer
import torchvision.models as models
import torch.nn as nn


class R2GenModel(torch.nn.Module):
    def __init__(self, model_name='t5-small', device=None, dropout_prob=0.1):
        super(R2GenModel, self).__init__()
        # Use the specified device or fall back to CPU if CUDA is unavailable
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the tokenizer for BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load the T5 model for conditional generation
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # ResNet101 as the visual feature extractor
        self.visual_extractor = models.resnet101(pretrained=True).to(self.device)
        self.visual_extractor.fc = torch.nn.Linear(self.visual_extractor.fc.in_features, self.model.config.d_model).to(self.device)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, attention_mask, images, labels=None):
        """
        Forward method for the model.
        - input_ids: Tokenized input text IDs.
        - attention_mask: Mask for padded tokens in input.
        - images: Input images for visual features.
        - labels: Ground truth labels (optional, for training).
        """
        visual_features = self.extract_visual_features(images)
        visual_features = self.dropout(visual_features)  # Apply dropout
        if labels is not None:
            # Training mode
            return self.model(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=(visual_features,), labels=labels)
        else:
            # Inference mode
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=(visual_features,))

    def extract_visual_features(self, images):
        """
        Extracts visual features using ResNet101.
        """
        images = images.to(self.device)
        visual_features = self.visual_extractor(images)
        visual_features = visual_features.unsqueeze(1)  # Add sequence dimension
        return visual_features

    def generate_caption(self, input_ids, images, max_length=50):
        """
        Generates a caption given input text and images.
        - input_ids: Tokenized input text IDs.
        - images: Input images for visual features.
        - max_length: Maximum length of generated text.
        """
        visual_features = self.extract_visual_features(images)
        generated_ids = self.model.generate(input_ids=input_ids, encoder_outputs=(visual_features,), max_length=max_length, num_beams=1)
        generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        return generated_texts


def load_model(weights_path, model_name='t5-small', device=None):
    """
    Load a pretrained R2GenModel with weights.
    """
    model = R2GenModel(model_name=model_name, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(model.device)))
    model.eval()
    return model
