import random
import string


def highlight_text(main_text, annot):
    """If we are using an extractive QA pipeline, we'll get answers
    from the API that we highlight in the given context"""
    start_idx = main_text.find(annot)
    end_idx = start_idx + len(annot)
    return main_text[:start_idx] + f"`{annot}`" + main_text[end_idx:]


def highlight_span(main_text, annot, start_idx, end_idx):
    return main_text[:start_idx] + f"`{annot}`" + main_text[end_idx:]


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str
