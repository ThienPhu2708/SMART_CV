def check_keyword(text, keyword):
    text = text.lower()
    return 1 if keyword.lower() in text else 0
def logic_and(a, b):
    return 1 if a and b else 0
def logic_or(a, b):
    return 1 if a or b else 0
def logic_not(a):
    return 0 if a else 1
def logic_xor(a, b):
    return 1 if a != b else 0