# import pickle
#
# with open('Evaluate/semi_hard_history_v1.p', 'rb') as f:
#     history_dict = pickle.load(f)
# import matplotlib.pyplot as plt
#
# # Vẽ loss
# plt.plot(history_dict['loss'], label='Training Loss')
#
# # Nếu có validation loss
# if 'val_loss' in history_dict:
#     plt.plot(history_dict['val_loss'], label='Validation Loss')
#
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


import pickle
import matplotlib.pyplot as plt

# Load history đầu (ví dụ lưu 20 epoch đầu)
with open('Evaluate/history_osht.p', 'rb') as f:
    history = pickle.load(f)



plt.plot(history['loss'], label='Train Loss')

if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Triplet Loss over Epochs")
plt.show()
