"""
Change from your original:
1. Import path fixed:
   OLD (deprecated re-export): from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
   NEW:                        from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
   -> Text splitters were split out of the core `langchain` package into
      their own package, `langchain-text-splitters`, a while back. The old
      `langchain.text_splitter` path still works today via a backward-
      compatibility shim, but it's the deprecated route and could be
      removed in a future major version.
   -> pip install langchain-text-splitters (usually already installed as
      a dependency of langchain, but pin/upgrade it explicitly if needed)

Everything else in your code (.from_language(), .split_text()) is
already using the current API - no other changes needed.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])