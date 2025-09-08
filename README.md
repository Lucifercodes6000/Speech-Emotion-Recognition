# Speech-Emotion-Recognition
Github Repository for the project "Emotion recognition From human voice using deep learning"


## 📥 Dataset
This project uses the [RAVDESS Emotional Speech Audio dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).

The dataset will be automatically downloaded using [KaggleHub](https://pypi.org/project/kagglehub/):

```python
import kagglehub
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
