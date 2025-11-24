import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import XRayDataset
from transforms import get_val_transform
from model import get_model


def load_model(checkpoint_path, device):
    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # evaluation mode
    return model


def predict():

    test_csv = "/home/jupyter-nafisha/X-ray/CSVs/test.csv"
    img_dir = "/home/jupyter-nafisha/X-ray/Data"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (choose best_model.pth or last_model.pth)
    model = load_model("best_model.pth", device)

    # Dataset & loader
    test_dataset = XRayDataset(test_csv, img_dir, transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    all_preds = []
    all_image_names = []

    with torch.no_grad():
        for images, _, image_names in test_loader:   # assuming dataset returns (img, label_dummy, filename)
            images = images.to(device)

            outputs = model(images)              # model outputs logits
            probs = torch.softmax(outputs, dim=1)  # convert to probabilities
            preds = torch.argmax(probs, dim=1)     # predicted class

            all_preds.extend(preds.cpu().numpy().tolist())
            all_image_names.extend(list(image_names))

    # Save predictions
    df = pd.DataFrame({
        "image_name": all_image_names,
        "prediction": all_preds
    })

    df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")


if __name__ == "__main__":
    predict()
