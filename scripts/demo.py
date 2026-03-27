#!/usr/bin/env python3
"""
FischGPT-SFT Demo

Demonstrates the FischGPT-SFT conversational model built from scratch.
This script loads the model from HuggingFace Hub and provides both
automated examples and interactive chat.

Usage:
    python demo.py

Perfect for:
- Job interviews and technical demonstrations
- Showcasing your from-scratch transformer implementation
- Portfolio presentations
"""

import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

class FischGPTDemo:
    """Demo class for FischGPT-SFT conversational model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "kristianfischerai12345/fischgpt-sft"
        
        print("FischGPT-SFT Demo")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Model: {self.model_id}")
        print("=" * 50)
        
    def load_model(self):
        """Load FischGPT-SFT from HuggingFace Hub."""
        
        try:
            print("Loading FischGPT-SFT from HuggingFace Hub...")
            
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, user_message, max_length=200, temperature=0.8, top_p=0.9):
        """Generate response to user message."""
        
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        try:
            # Format as conversation
            prompt = f"<|user|>{user_message}<|assistant|>"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            generation_time = time.time() - start_time
            
            # Decode and extract response
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text.split("<|assistant|>", 1)[1].strip()
            
            # Performance metrics
            new_tokens = len(outputs[0]) - len(inputs[0])
            tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
            
            return {
                "response": response,
                "tokens": new_tokens,
                "time": generation_time,
                "speed": tokens_per_sec
            }
            
        except Exception as e:
            return {"error": f"Generation failed: {e}"}
    
    def run_examples(self):
        """Run automated examples to showcase capabilities."""
        
        print("\nEXAMPLE DEMONSTRATIONS")
        print("=" * 50)
        
        examples = [
            {
                "category": "Technical Explanation", 
                "prompt": "Explain how neural networks work in simple terms"
            },
            {
                "category": "Code Help",
                "prompt": "Write a Python function to reverse a string"
            },
            {
                "category": "Problem Solving",
                "prompt": "What are the key steps in training a machine learning model?"
            },
            {
                "category": "Creative Task",
                "prompt": "Write a haiku about programming"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['category']}")
            print(f"User: {example['prompt']}")
            
            result = self.generate_response(example['prompt'], max_length=150, temperature=0.7)
            
            if "error" in result:
                print(f"{result['error']}")
                continue
            
            print(f"FischGPT: {result['response']}")
            print(f"{result['tokens']} tokens in {result['time']:.2f}s ({result['speed']:.1f} tok/s)")
    
    def interactive_mode(self):
        """Interactive chat mode."""
        
        print("\nINTERACTIVE CHAT MODE")
        print("=" * 50)
        print("Chat with FischGPT! Type 'quit' to exit, 'help' for tips.")
        print("Perfect for live demonstrations and interviews!")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Thanks for trying FischGPT!")
                    break
                elif user_input.lower() == 'help':
                    print("\nTips:")
                    print("• Ask technical questions: 'Explain transformers'")
                    print("• Request code: 'Write a sorting algorithm'") 
                    print("• Get help: 'How do I learn ML?'")
                    print("• Be creative: 'Write a poem about AI'")
                    print()
                    continue
                elif not user_input:
                    continue
                
                print("FischGPT is thinking...")
                result = self.generate_response(user_input, max_length=250, temperature=0.8)
                
                if "error" in result:
                    print(f"{result['error']}")
                    continue
                
                print(f"FischGPT: {result['response']}")
                print(f"{result['tokens']} tokens, {result['speed']:.1f} tok/s")
                print()
                
            except KeyboardInterrupt:
                print("\nDemo interrupted. Thanks for trying FischGPT!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def technical_showcase(self):
        """Show technical details for professional presentation."""
        
        print("\nTECHNICAL ACHIEVEMENTS")
        print("=" * 50)
        print("Built from scratch - No pre-existing transformer libraries")
        print("Flash attention using F.scaled_dot_product_attention")
        print("Mixed precision training (bfloat16)")
        print("Distributed training with DistributedDataParallel")
        print("Trained on 10B tokens (FineWeb-Edu) + OpenAssistant SFT")
        print("Professional HuggingFace deployment")
        print()
        print(f"Model Hub: https://huggingface.co/{self.model_id}")
        print("Architecture: GPT-2 style (12 layers, 768 hidden, 12 heads)")
        print("Parameters: ~124M parameters")
        print("Context: 1024 tokens")
        print()
    
    def run_demo(self):
        """Run the complete demo."""
        
        # Load model
        if not self.load_model():
            print("Cannot proceed without model. Check your internet connection.")
            return
        
        # Technical showcase
        self.technical_showcase()
        
        # Automated examples
        self.run_examples()
        
        # Interactive mode
        print("\n" + "=" * 50)
        choice = input("Enter interactive mode? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '']:
            self.interactive_mode()
        
        print("\nDemo completed!")

def main():
    """Main entry point."""
    demo = FischGPTDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 