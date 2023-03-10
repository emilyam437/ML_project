import pandas as pd
import matplotlib.pyplot as plt
import cv2
import keras
import random

my_letters = pd.read_csv('./src/data/processed_files/DataFrames/my_letters_path.csv')

best_model = keras.models.load_model('./src/models/all_data_44_labels.tf')

my_letters.path = [f'{val[:2]}src/{val[2:]}' for val in my_letters.path]

label_index = ['o', 'i', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'c',
       'D', 'E', 'f', 'G', 'H', 'j', 'k', 'L', 'm', 'N', 'p', 'Q', 'R', 's',
       'T', 'u', 'v', 'w', 'x', 'y', 'z', 'a', 'b', 'd', 'e', 'g', 'h', 'n',
       'q', 'r', 't']

random_values = [random.randint(0, len(my_letters)-1) for i in range(6)]

fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 5))

ax[0].imshow(cv2.imread(my_letters['path'].iloc[random_values[0]]))
ax[0].set_title(str(random_values[0]))
ax[1].imshow(cv2.imread(my_letters['path'].iloc[random_values[1]]))
ax[1].set_title(str(random_values[1]))
ax[2].imshow(cv2.imread(my_letters['path'].iloc[random_values[2]]))
ax[2].set_title(str(random_values[2]))
ax[3].imshow(cv2.imread(my_letters['path'].iloc[random_values[3]]))
ax[3].set_title(str(random_values[3]))
ax[4].imshow(cv2.imread(my_letters['path'].iloc[random_values[4]]))
ax[4].set_title(str(random_values[4]))
ax[5].imshow(cv2.imread(my_letters['path'].iloc[random_values[5]]))
ax[5].set_title(str(random_values[5]))

plt.show();

word = [label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[0]]))[0].reshape(1, 40,40,1)/255).argmax()], 
label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[1]]))[0].reshape(1, 40,40,1)/255).argmax()],
label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[2]]))[0].reshape(1, 40,40,1)/255).argmax()],
label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[3]]))[0].reshape(1, 40,40,1)/255).argmax()],
label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[4]]))[0].reshape(1, 40,40,1)/255).argmax()],
label_index[best_model.predict(cv2.split(cv2.imread(my_letters['path'].iloc[random_values[5]]))[0].reshape(1, 40,40,1)/255).argmax()]]
print(word)
