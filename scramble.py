import base64

def scramble_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")

def unscramble_text(scrambled: str) -> str:
    return base64.b64decode(scrambled.encode("utf-8")).decode("utf-8")