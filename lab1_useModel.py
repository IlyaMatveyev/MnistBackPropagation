import os
import argparse
import pickle
import numpy as np
import cv2


# ---------------------------
# Активации
# ---------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=0, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=0, keepdims=True)


# ---------------------------
# Предсказание для одного изображения
# ---------------------------
def predict_image(img_path: str, model_path: str, image_size=(28, 28)) -> int:
    # Проверяем, что модель существует
    if not os.path.isfile(model_path):
        raise ValueError(f"Файл модели не найден: {model_path}")

    # Загружаем модель
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

    # Загружаем изображение
    if not os.path.isfile(img_path):
        raise ValueError(f"Файл изображения не найден: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {img_path}")


    # Нормализуем пиксели в диапазон [0, 1]
    img_f = img.astype(np.float32) / 255.0

    # Преобразуем в столбец (784, 1)
    x = img_f.reshape(-1, 1)

    # Прямой проход
    Z1 = W1 @ x + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    probs = softmax(Z2)

    # Находим индекс с наибольшей вероятностью
    pred_class = int(np.argmax(probs))

    print(f"Предсказанный класс: {pred_class}")
    print("Вероятности по классам:", probs.ravel())

    return pred_class


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    predict_image(args.image_path, args.model_path)


if __name__ == "__main__":
    main()
