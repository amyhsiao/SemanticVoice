import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig

class ASTVoiceModel(nn.Module):
    """
    AST Model wrapper for extracting high-dimensional audio embeddings (768-d).
    Used as the 'audio space' in Procrustes Alignment.
    """
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_classes=5):
        super(ASTVoiceModel, self).__init__()
        # Load pre-trained AST backbone
        self.ast = ASTModel.from_pretrained(model_name)
        
        # Regression head (as used in the study for fine-tuning)
        # Note: Procrustes Alignment uses the output *before* this head (the [CLS] token embedding)
        self.regressor = nn.Sequential(
            nn.Linear(self.ast.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        # input_values: [batch, max_length, n_mels]
        outputs = self.ast(input_values)
        
        # Use the [CLS] token embedding as the audio feature (768-d)
        # pooled_output = outputs.pooler_output # Transformers library pooler
        # Or more explicitly, the first token's state:
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # For prediction tasks
        logits = self.regressor(cls_embedding)
        
        return {
            "embeddings": cls_embedding,
            "logits": logits
        }

def load_model(checkpoint_path=None, device="cpu"):
    """
    Loads the AST model. If checkpoint_path is provided, loads trained weights.
    """
    model = ASTVoiceModel()
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
