import timm
import torch

from models.emotion_detector import EmotionDetector


def test_loads_convnext_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "convnext_emotion.pth"

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=7)
    torch.save(model.state_dict(), checkpoint_path)

    detector = EmotionDetector(
        model_path=str(checkpoint_path),
        model_name="convnext_tiny",
        num_classes=7,
    )

    assert detector is not None
    assert len(detector.emotions) == 7
