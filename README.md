# Face-to-Prompt Generator 🎨

Этот проект использует **IP-Adapter + Stable Diffusion** для генерации реалистичных портретов, ориентируясь на лицо с фотографии и заданный текстовый промпт. Подход — **online**, то есть генерация осуществляется в режиме реального времени с загрузкой моделей в оперативную память.

> ⚙️ Задача решалась в рамках **Test Task for CV Engineer**.

## 📆 Зависимости

* Python 3.8+
* torch
* diffusers
* transformers
* accelerate
* insightface
* onnxruntime
* opencv-python
* pillow

Установка:

```bash
pip install torch torchvision diffusers transformers accelerate
pip install insightface onnxruntime
pip install opencv-python pillow
```

Веса модели можно скачать тут [https://huggingface.co/h94/IP-Adapter-FaceID/tree/main](https://huggingface.co/h94/IP-Adapter-FaceID/tree/main)

## 🔧 Как работает

1. Загружается фотография с лицом (например, `task2/IMG_6909.jpg`).
2. Извлекается face embedding с помощью `InsightFace`.
3. С помощью IP-Adapter и Stable Diffusion по embedding и заданному промпту создаются изображения.
4. Каждое изображение сохраняется в папку, соответствующую своему промпту.

## 📂 Структура проекта

```
.
├── task2/
│   └── IMG_6909.jpg             # Изображение с лицом
├── generated_images/            # Папки с результатами генерации
│   ├── prompt_01/
│   ├── ...
├── ip-adapter-faceid_sd15.bin   # Чекпойнт IP-Adapter
├── FaceGen.ipynb                    # Основной скрипт генерации
├── README.md                    # Описание проекта
```

## 📃 Примеры промптов

* a photo of a young woman in a garden, wearing a red dress
* a portrait of a woman wearing a vintage hat, sitting by the sea at sunset
* a cinematic headshot of a smiling woman with freckles outdoors
* a photo of a woman on a city street at night with neon lights
* a renaissance painting style portrait of a woman in a forest

## ⚠️ Советы

* Используйте изображение с чётким фронтальным лицом.
* Убедитесь, что файл `ip-adapter-faceid_sd15.bin` загружен и доступен.

