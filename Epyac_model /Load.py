# inference.py
import torch
from Model import SimpleTransformer
from Data import TextDataset

def load_model(model_path, vocab_size, device="cuda"):
    model = SimpleTransformer(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50, device="cuda", temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()
    
    print(f"Initial input IDs: {input_ids.tolist()}")
    
    with torch.no_grad():
        for step in range(max_length):
            output = model(generated_ids)
            next_token_logits = output[:, -1, :] / temperature  # Apply temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)  # Sample instead of argmax
            
            print(f"Step {step}: Predicted token ID: {next_token.item()}")
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id and step > 5:  # Don’t stop too early
                print("EOS token detected, stopping generation.")
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated IDs: {generated_ids.tolist()}")
    return generated_text

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use GPU if available
    if device == "cuda":
        print("Using CUDA on GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    
    dataset = TextDataset()  # Only for tokenizer, no file needed here
    tokenizer = dataset.tokenizer
    
    model = load_model("Model.pth", vocab_size=dataset.vocab_size(), device=device)
    
    while True:
        prompt = input("Ask me anything (or type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        response = generate_text(model, tokenizer, prompt, max_length=50, device=device, temperature=0.7)
        print("Response:", response)