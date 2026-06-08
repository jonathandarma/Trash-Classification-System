import os
import joblib
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from feature_extraction import extract_features_from_image

DATASET_DIR = Path(
    "trash_project/dataset/final"
)

MODEL_PATH = "model/trash_xgboost_model.pkl"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

def augment_image(image):

    augmented = [image]

    augmented.append(
        cv2.convertScaleAbs(
            image,
            alpha=1.15,
            beta=15
        )
    )

    augmented.append(
        cv2.convertScaleAbs(
            image,
            alpha=0.85,
            beta=-15
        )
    )

    h, w = image.shape[:2]

    for angle in [-15, -10, 10, 15]:

        matrix = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

        augmented.append(rotated)

    return augmented

def load_dataset():

    X = []
    y = []

    for class_id, class_name in enumerate(CLASSES):

        class_folder = (
            DATASET_DIR /
            class_name
        )

        image_files = list(
            class_folder.glob("*")
        )

        print(
            f"{class_name}: "
            f"{len(image_files)}"
        )

        for image_file in tqdm(image_files):

            image = cv2.imread(
                str(image_file)
            )

            if image is None:
                continue

            try:

                for aug_image in augment_image(image):

                    features = extract_features_from_image(
                    aug_image
                    )

                X.append(features)

                y.append(class_id)

            except Exception:

                continue

    return (
        np.array(X, dtype=np.float32),
        np.array(y)
    )


def main():

    print(
        "Loading dataset..."
    )

    X, y = load_dataset()

    print(
        "\nDataset shape:"
    )

    print(X.shape)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    gamma=0.1,

    reg_alpha=0.05,

    reg_lambda=1.0,

    objective="multi:softprob",

    num_class=len(CLASSES),

    random_state=42,

    tree_method="hist",

    n_jobs=-1
)

    print(
        "\nTraining..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
    y_test,
    predictions
    )

    print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASSES
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    os.makedirs(
        "model",
        exist_ok=True
    )

    joblib.dump({

        "model": model,

        "classes": CLASSES,

        "scaler": scaler

    }, MODEL_PATH)

    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    main()