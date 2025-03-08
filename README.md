NOW I am in the alpha mode to develop original AI model not [the previous model](https://ollama.com/asaad/epyac.1)

**MY First AI MODEL from scratch**
to use it in EPYAC-ter tool and comming epyac-gui and phone app 
i am just asking god to Help me (੭ˊ^ˋ)੭ ♡   

# Epyac AI Model 🤖✨

Welcome to **Epyac AI** **Efficient Processing Yielding Advanced Cognition AI MODEL**, a custom-built transformer-based language model created with PyTorch! This project, lets you train and interact with a text-generating AI locally on your machine. Whether you’re asking about the best cakes 🍰 or exploring natural language generation, this model is your starting point! 🚀

---

## 🌟 Features
- **Custom Transformer**: A decoder-only architecture inspired by models like LLaMA, with ~30M parameters.
- **Local Training**: Train on datasets like WikiText-2 using your GPU (e.g., Quadro P3200) or CPU.
- **Interactive Inference**: Ask questions and get responses in real-time.
- **Flexible Data**: Supports any text file for training, from books to custom corpora.

---

## 📋 Prerequisites
Before diving in, ensure you have:
- **Python 3.7+** 🐍
- **PyTorch** (GPU support recommended) 🔥
- **Transformers** (Hugging Face) 🤗
- **Datasets** (Hugging Face) 📚
- A decent GPU (e.g., NVIDIA Quadro P3200) or patience for CPU training ⏳

---

## 🛠️ Installation
1. **Clone the Repository** (or create it locally):
   ```bash
   git clone 
   cd Epyac_model
    ```
2. **Install Dependencies:**
    ```bash
    pip install requirements-detector
    ```
    ***For GPU support, install PyTorch with CUDA: pytorch.org/get-started.***

## 🚀 Usage

1. Train the Model

- Prepare Data: The default uses WikiText-2, downloaded automatically.
- Run Training:
```
python Train.py
```
- Outputs Model.pth after training (~10 epochs by default).
- Expect ~1 hour on a Quadro P3200 with WikiText-2.

2. Interact with the Model
- Run Inference to Load Model:
```
python Load.py
```
- Type a prompt (e.g., "What is the best cake?") and get a response!
- Type exit to quit.

### Example
```
$ python Load.py
Using cuda on GPU: Quadro P3200
Ask me anything (or type 'exit' to quit): What is the best cake?
Initial input IDs: [[2061, 318, 262, 1266, 11642, 30]]
Step 0: Predicted token ID: 8840
...
Response: (AI Response).
```
## 📂 Project Structure
- Epyac_model/
- ├── Model.py         # Transformer model definition 🤖
- ├── Data.py          # Dataset loading and tokenization 📝
- ├── Train.py         # Training script 🏋️‍♂️
- ├── Load.py          # Inference script 💬
- ├── Model.pth        # Trained model weights (after training) 💾
- ├── wikitext_train.txt  # WikiText-2 training data (generated) 📚
- └── README.md        # This file! 👋

## 🎨 Customization
- Change Dataset: Edit Train.py to use your own text file:

```
dataset = TextDataset("path/to/your_text.txt")
```
- Adjust Hyperparameters: Modify Model.py (e.g., d_model, n_layers) or Train.py (e.g., epochs, lr).

## ⚠️ Known Issues
- epetition: Model may repeat tokens (e.g., ", , ,") if undertrained.
- Tokenizer Warning: Ignored safely as Data.py chunks long sequences.

## 📈 Training Tips
- Epochs: Start with 10, increase to 20-50 for better results.
- Learning Rate: Default is 0.001; try 0.0005 if loss plateaus.
- Loss: Aim for <5 after 10 epochs with WikiText-2.

## 🤝 Contributing
Feel free to fork, tweak, and submit pull requests! Add more datasets, improve the model, or share cool outputs! 🌟

## 📜 License

This project is open-source under the MIT License. Do whatever you like with it! 🎉

## 🙌 Acknowledgments
- Built with ❤️ by [Asaad] (htttps://github.com/asaadzx).
- Powered by PyTorch, Hugging Face, and WikiText-2.