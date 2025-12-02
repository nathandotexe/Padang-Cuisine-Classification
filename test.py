import cv2
import torch
from torchvision.transforms import ToTensor, Resize
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image


MODEL_PATH = "./runs/train_003/model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Load model
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]
num_classes = len(classes)

model = build_model(num_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

resize = Resize((224, 224))
to_tensor = ToTensor()


# ================================
# TEST SINGLE IMAGE
# ================================
def test_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Invalid image path")
        return

    img_pil = Image.fromarray(rgb)
    t = to_tensor(resize(img_pil)).unsqueeze(0).to(DEVICE)


    with torch.no_grad():
        logits = model(t)
        pred = torch.argmax(logits, 1).item()

    print("Prediction:", classes[pred])


# ================================
# TEST WITH WEBCAM
# ================================
def test_webcam(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    print("[INFO] Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert numpy → PIL
        img_pil = Image.fromarray(rgb)

        # Resize + ToTensor
        t = to_tensor(resize(img_pil)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(t)
            pred = classes[torch.argmax(logits, 1).item()]

        cv2.putText(frame, pred, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("EfficientNet Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



# ================================
# RUN MODE
# ================================
if __name__ == "__main__":
    MODE = "webcam"   # webcam | image
    if MODE == "image":
        test_image("tes1.jpg")
    else:
        test_webcam()
