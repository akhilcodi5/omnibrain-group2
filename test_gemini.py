import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini_pro():
    # Load the GEMINI_API_KEY from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return

    print("🔑 Configuring Gemini API...")
    genai.configure(api_key=api_key)
    
    model_name = "gemini-3.5-flash-lite"
    print(f"🤖 Initializing model: {model_name}")
    
    try:
        model = genai.GenerativeModel(model_name)
        print("📨 Sending prompt: 'hi'...")
        
        response = model.generate_content("hi")
        
        print("\n=== Model Response ===")
        print(response.text.strip())
        print("======================\n")
        print(f"✅ Success! {model_name} is available and working on your API key.")
        
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")
        print(f"{model_name} might not be available for your API key tier, or there is a network issue.")

if __name__ == "__main__":
    test_gemini_pro()
