import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
import tkinter as tk
from transformers import pipeline

# Preload NLTK data
nltk.download('punkt')
nltk.download('words')

# Predefined list of English words
word_list = words.words()
main_set = set(word_list)

# Context-aware grammar correction model using a dedicated grammar error corrector
grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

# Function to get spelling suggestions
def get_spelling_suggestions(word):
    suggestions = (
        set(DeleteLetter(word)) |
        set(Switch_(word)) |
        set(Replace_(word)) |
        set(Insert_(word))
    )
    return suggestions.intersection(main_set)

# Define spelling modification functions
def DeleteLetter(word):
    split_list = [(word[0:i], word[i:]) for i in range(len(word))]
    return [a + b[1:] for a, b in split_list]

def Switch_(word):
    split_list = [(word[0:i], word[i:]) for i in range(len(word))]
    return [a + b[1] + b[0] + b[2:] for a, b in split_list if len(b) >= 2]

def Replace_(word):
    split_list = [(word[0:i], word[i:]) for i in range(len(word))]
    alphs = 'abcdefghijklmnopqrstuvwxyz'
    return [a + l + (b[1:] if len(b) > 1 else '') for a, b in split_list if b for l in alphs]

def Insert_(word):
    split_list = [(word[0:i], word[i:]) for i in range(len(word) + 1)]
    alphs = 'abcdefghijklmnopqrstuvwxyz'
    return [a + l + b for a, b in split_list for l in alphs]

# Correct text and provide suggestions
def correct_text(sentence):
    tokens = word_tokenize(sentence)
    suggestions = {}

    # Spell-checking
    for token in tokens:
        lower_token = token.lower()
        if lower_token not in main_set:  # If the word is misspelled
            spell_suggestions = get_spelling_suggestions(lower_token)
            suggestions[token] = list(spell_suggestions) if spell_suggestions else ["No suggestions"]

    # Context-aware grammar correction
    try:
        corrected_sentence = grammar_corrector("grammar: " + sentence)[0]["generated_text"]
    except Exception as e:
        corrected_sentence = f"Error in grammar correction: {e}"

    return corrected_sentence, suggestions

# GUI setup
def autocorrect_text():
    user_input = input_text.get("1.0", tk.END).strip()
    corrected, suggestions = correct_text(user_input)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, corrected)

    suggestion_text.delete("1.0", tk.END)
    for word, suggestion_list in suggestions.items():
        suggestion_text.insert(tk.END, f"{word}: {', '.join(suggestion_list)}\n")

def clear_text():
    input_text.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)
    suggestion_text.delete("1.0", tk.END)

# Tkinter GUI setup with dark blue background and white text boxes
root = tk.Tk()
root.title("Context-Aware Autocorrector")
root.geometry("700x500")
root.configure(bg="#0a2a3b")  # Set the background to dark blue

# Labels with white text on dark blue background
tk.Label(root, text="Enter Text:", font=("Arial", 14), bg="#0a2a3b", fg="white").pack(pady=10)
input_text = tk.Text(root, height=5, width=80, font=("Arial", 12), bg="white", fg="black", bd=2)
input_text.pack(pady=10)

# Frame for buttons
button_frame = tk.Frame(root, bg="#0a2a3b")
button_frame.pack(pady=5)

# Buttons with black text inside the frame
tk.Button(button_frame, text="Autocorrect", command=autocorrect_text,
bg="#1f76d3", font=("Arial", 12, "bold"), fg="black").grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Clear", command=clear_text, bg="#f44336", font=("Arial", 12,
"bold"), fg="black").grid(row=0, column=1, padx=5)

# Labels and output with white text
tk.Label(root, text="Suggested Text:", font=("Arial", 14), bg="#0a2a3b", fg="white").pack(pady=10)
output_text = tk.Text(root, height=5, width=80, font=("Arial", 12), bg="white", fg="black", bd=2)
output_text.pack(pady=10)

tk.Label(root, text="Suggestions for Misspelled Words:", font=("Arial", 14), bg="#0a2a3b", fg="white").pack(pady=10)
suggestion_text = tk.Text(root, height=5, width=80, font=("Arial", 12), bg="white", fg="black", bd=2)
suggestion_text.pack(pady=10)

root.mainloop()
