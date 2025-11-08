import os
import argparse
import numpy as np
import cv2
import time
import pickle
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------------------
# Активации и loss
# ---------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative_from_activation(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)

def softmax(z: np.ndarray) -> np.ndarray:
    # Находим максимумы для каждого столбца(для каждого примера).
    # Чтобы избежать переполнения (Inf) в дальнейших вычислениях. 
    z_max = np.max(z, axis=0, keepdims=True) 
    
	#Вычисляем разницу каждого значения в примере и максимального значения в примере.
    #Затем по каждому полученному значению вычисляем e. Получим положительные значения в пределах [0;1]. 
    e = np.exp(z - z_max)
    
	#Каждый элемент делим на сумму своего столбца, тем самым получая вероятности.
    return e / np.sum(e, axis=0, keepdims=True)

def cross_entropy_loss(Y_true_onehot: np.ndarray, Y_pred_probs: np.ndarray) -> float:
	#Вводим очень малое число, чтобы избежать ошибки при 
	# вычислении логарифма в дальнейших вычислениях. 
	# Т.к. log(0) будет стремиться к -Inf, а добавляя eps 
	# мы можем быть уверены что вычисления дадут конечный результат.
    eps = 1e-12
    
    #Получаем вектор положительных значений ошибки для каждого примера.
    loss_per_example = -np.sum(Y_true_onehot * np.log(Y_pred_probs + eps), axis=0)
    
    #Возвращаем среднее значение ошибки по батчу.
    return float(np.mean(loss_per_example))

# ---------------------------
# Загрузка изображений из директории
# ---------------------------
def load_images_from_dir_flat(data_dir: str, image_size: Tuple[int,int]=(28,28), max_images: int=None):
    # Проверка: существует ли папка.
    if not os.path.isdir(data_dir):
        raise ValueError(f"Директория не найдена: {data_dir}")

	#Получаем список всех .png файлов.
    files = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
    files.sort()
    if len(files) == 0:
        raise ValueError(f"PNG изображения не найдены в {data_dir}")


    X_list = [] #Изображения в виде массивов
    y_list = [] #Метки (цифры)
    count = 0
    for fname in files:
        if max_images is not None and count >= max_images:
            break
        fpath = os.path.join(data_dir, fname) #Получаем полный путь изображения.

        # Парсим метку из имени файла.
        try:
            label_str = fname.split('_')[0]
            label = int(label_str)
        except Exception:
            print(f"Не удалось прочитать изображение  '{fname}', (пропущено).")
            continue

        # Читаем изображение в чб.
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось прочитать изображение '{fpath}' (пропущено).")
            continue

        # Изменение размера изображения, если требуется.
        if (img.shape[1], img.shape[0]) != image_size:
            img = cv2.resize(img, image_size)

        # нормализуем в диапазон [0,1]
        img_f = img.astype(np.float32) / 255.0

        X_list.append(img_f.reshape(-1))  	# Добавляем в список изображение приведённое к вектору 28*28.
        y_list.append(label)				# Добавляем метку в список.
        count += 1

    X = np.stack(X_list, axis=0)  # Матрица, в которой строки - отдельные изображения.
    y = np.array(y_list, dtype=np.int32) # Список меток.
    print(f"Загружено {X.shape[0]} изображений из '{data_dir}'")
    return X, y

# ---------------------------
# Получение верных ответов.
# ---------------------------
def to_one_hot(y: np.ndarray, k: int) -> np.ndarray:
    m = y.shape[0]
    Y = np.zeros((k, m), dtype=np.float32)
    Y[y, np.arange(m)] = 1.0
    return Y

# ---------------------------
# Инициализация параметров (веса и смещения)
# ---------------------------
def init_parameters(n_in: int, n_h: int, n_out: int):
    
    # Веса между входным и скрытым слоем. + Нормализация чтобы значения не были слишком большими (Xavier-ish).
    W1 = np.random.randn(n_h, n_in) * np.sqrt(1.0 / n_in)
    
	#Смещения по умолчанию все 0.
    b1 = np.zeros((n_h, 1), dtype=np.float32) 
    
	# Веса между скрытым и выходным слоем. + Нормализация чтобы значения не были слишком большими (Xavier-ish).
    W2 = np.random.randn(n_out, n_h) * np.sqrt(1.0 / n_h)
    
	#Смещения по умолчанию все 0.
    b2 = np.zeros((n_out, 1), dtype=np.float32) 
    

    return W1.astype(np.float32), b1, W2.astype(np.float32), b2

# ---------------------------
# Прямой проход + Обратное распространение ошибки.
# ---------------------------
def forward_backward_update(X_batch: np.ndarray, Y_batch_onehot: np.ndarray,
                            W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                            lr: float):
    
    m = X_batch.shape[0] #Кол-во примеров в батче.
    
    # Транспонируем, чтобы получить вид столбцы-примеры.
    X_t = X_batch.T 

    # Прямой проход.
    Z1 = W1 @ X_t + b1 #Взвешенные суммы нейронов(скрытого слоя) по всем примерам батча.
    A1 = sigmoid(Z1) #Применение функции активации (нейронов скрытого слоя).
    
    Z2 = W2 @ A1 + b2 #Взвешенные суммы нейронов(выходного слоя) по всем примерам батча.
    Y_hat = softmax(Z2) #Получаем вероятности по каждому классу каждого примера.
    loss = cross_entropy_loss(Y_batch_onehot, Y_hat) #Высчитываем среднюю ошибку по батчу.


    # Обратный проход
    
	#Ошибки каждого выходного нейрона для каждого примера в батче.
    Delta2 = Y_hat - Y_batch_onehot 
	# Средние градиенты функции потерь по весам между скрытым и выходным.
    dW2 = (1.0 / m) * (Delta2 @ A1.T) 
	# Средние градиенты смещений выходного слоя.
    db2 = (1.0 / m) * np.sum(Delta2, axis=1, keepdims=True) 

    # Перенос ошибки с выходного на скрыты слой
    dA1 = W2.T @ Delta2 
    dZ1 = dA1 * sigmoid_derivative_from_activation(A1) 
    dW1 = (1.0 / m) * (dZ1 @ X_t.T) 
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True) 

    # Обновление параметров.
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return W1, b1, W2, b2, loss

# ---------------------------
# Предикты по чанкам.
# ---------------------------
def predict_probs_in_chunks(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, chunk_size: int=2000):
    m = X.shape[0]
    parts = []
    for i in range(0, m, chunk_size):
        Xc = X[i:i+chunk_size]
        Xt = Xc.T
        Z1 = W1 @ Xt + b1
        A1 = sigmoid(Z1)
        Z2 = W2 @ A1 + b2
        probs = softmax(Z2)
        parts.append(probs)
    return np.concatenate(parts, axis=1)  # (k, m)

# ---------------------------
# Обучение.
# ---------------------------
def train_and_evaluate(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, y_test: np.ndarray,
                       epochs: int, lr: float, batch_size: int, hidden_units: int, out_dir: str):
    
	# Кол-во входных и выходных нейронов.
    n_input = x_train.shape[1]
    n_output = 10

    # init params
    W1, b1, W2, b2 = init_parameters(n_input, hidden_units, n_output)

    num_train = x_train.shape[0]
    steps_per_epoch = max(1, num_train // batch_size)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # Перемешиваем примеры, чтобы они были в разном порядке в разных эпохах.
        perm = np.random.permutation(num_train)
        x_sh = x_train[perm]
        y_sh = y_train[perm]

		# Ошибка в рамках эпохи.
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            
            Xb = x_sh[start:end] 
            yb = y_sh[start:end]

			# Верные ответы формы (k, m) k - число классов(строки), m - число примеров в батче(столбцы).
            Yb_oh = to_one_hot(yb, n_output) 

			# Обновление весов и вычисление ошибок.
            W1, b1, W2, b2, loss = forward_backward_update(Xb, Yb_oh, W1, b1, W2, b2, lr)
            epoch_loss += loss

        epoch_loss /= steps_per_epoch

        # Оценка полученных параметров на тестовой выборке.
        val_probs = predict_probs_in_chunks(x_test, W1, b1, W2, b2, chunk_size=2000)
        val_loss = cross_entropy_loss(to_one_hot(y_test, n_output), val_probs)
        val_preds = np.argmax(val_probs, axis=0)
        val_acc = float(np.mean(val_preds == y_test))

        history['train_loss'].append(epoch_loss) #сред. ошибка в эпохе
        history['val_loss'].append(val_loss) #сред. ошибка на тест. данных
        history['val_acc'].append(val_acc) #точность предсказаний на тест. данных.

        t1 = time.time()
        print(f"Epoch {epoch}/{epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  time={t1-t0:.1f}s")

    # Сохранение модели и истории обучения.
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "mnist_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'history': history}, f)
    print("Модель сохранена в:", model_path)

    # Сохраняем графики.
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Val Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"))
    plt.close()
    print("История обучения сохранена в:", os.path.join(out_dir, "training_history.png"))

    return model_path, history


# ---------------------------
# Main.
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    print("Загрузка тренировочных данных из:", args.train_dir)
    x_train, y_train = load_images_from_dir_flat(args.train_dir)
    print("Загрузка тестовых данных из:", args.test_dir)
    x_test, y_test = load_images_from_dir_flat(args.test_dir)

    start_time = time.time()
    model_path, history = train_and_evaluate(x_train, y_train, x_test, y_test,
                                             epochs=args.epochs, lr=args.lr,
                                             batch_size=args.batch_size, hidden_units=args.hidden,
                                             out_dir=args.out_dir)
    elapsed = time.time() - start_time
    print(f"Завершено за {elapsed:.1f}сек. Модель сохранена в {model_path}")

if __name__ == "__main__":
    main()
