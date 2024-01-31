import tkinter as tk
from transformers import pipeline
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def extract_action_words(text):
    # Tokenize the input text and perform part-of-speech tagging
    doc = nlp(text)
    
    # Extract verbs (action words) from the text
    action_words = [token.text.strip() for token in doc if token.pos_ == "VERB"]

    return action_words

def findActionItems(text):
    actionItems = pipeline("text2text-generation", model="debal/distilbart-samsum-action-items")
    items = actionItems(text)

    # Extract action items from each generated response
    action_items_list = [item['generated_text'].split('[SEP]')[1].split('||') for item in items]

    # Flatten the list of lists into a single list
    action_items = [item.strip() for sublist in action_items_list for item in sublist if item.strip()]

    # Extract action words from the original text
    action_words = extract_action_words(text)

    # Filter action items to include only those that contain action words
    filtered_action_items = [item for item in action_items if any(action_word in item for action_word in action_words)]

    return filtered_action_items

class ChecklistApp:
    def __init__(self, root, action_items):
        self.root = root
        self.root.title("Action Items Checklist")

        self.action_items = action_items
        self.checked_items = []

        self.create_ui()

    def create_ui(self):
        for item in self.action_items:
            var = tk.IntVar()
            chk = tk.Checkbutton(self.root, text=item, variable=var, onvalue=1, offvalue=0)
            chk.pack(anchor="w")
            chk.deselect()  # Initially, checkboxes are unchecked

            # Store the Checkbutton and its associated variable
            self.checked_items.append((chk, var))

        # Add a button to print the checked items
        print_button = tk.Button(self.root, text="Print Checked Items", command=self.print_checked_items)
        print_button.pack()

    def print_checked_items(self):
        checked_items = [item[0]['text'] for item in self.checked_items if item[1].get() == 1]
        print("Checked Items:", checked_items)

def main():
    # Read text from file
    with open("text.txt", "r") as file:
        text = file.read()

    # Extract action items
    action_items = findActionItems(text)

    # Create Tkinter root window
    root = tk.Tk()

    # Create the ChecklistApp instance
    app = ChecklistApp(root, action_items)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
