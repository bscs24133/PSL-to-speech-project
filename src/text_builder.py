# src/text_builder.py
# Handles word-by-word and sentence accumulation logic

class TextBuilder:
    def __init__(self, mode="word"):
        """
        mode: "word"     — translate and speak each recognized sign immediately
              "sentence" — accumulate signs into a sentence, translate on commit
        """
        self.mode = mode
        self.words = []
        self.current = ""

    def add(self, label: str):
        """Add a recognized label."""
        self.current = label
        if self.mode == "sentence":
            self.words.append(label)

    def commit(self) -> str:
        """
        Returns the text to translate and speak.
        Word mode: returns current label.
        Sentence mode: returns all accumulated words joined, then clears.
        """
        if self.mode == "sentence":
            result = " ".join(self.words)
            self.words = []
            self.current = ""
            return result
        return self.current

    def clear(self):
        """Clear all accumulated text."""
        self.words = []
        self.current = ""

    def get_display(self) -> str:
        """Returns what to show on screen as accumulated text so far."""
        if self.mode == "sentence":
            return " ".join(self.words)
        return self.current

    def set_mode(self, mode: str):
        """Switch mode and clear buffer."""
        self.mode = mode
        self.clear()
