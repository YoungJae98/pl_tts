""" from https://github.com/keithito/tacotron """
from vits.text.cleaners import ipa_converter
from vits.text.symbols import symbols
import re

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


lang_code = {
    '0' : 'ko',
    '1' : 'hi',
    '2' : 'de',
    '3' : 'it',
    '4' : 'ru',
    '5' : 'es',
    '6' : 'pl',
    '7' : 'uk',
    '8' : 'fr-fr',
    '9' : 'en-us',
} 


# lang_code = {
#     '0': 'bn',
#     '1': 'hi',
#     '2': 'hi',
#     '3' : 'en-us',
#     '4': 'mr',
#     '5' : 'kn',
#     '6' : 'te'
# }

def text_to_sequence(text, language_code):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, lang_code[language_code])
  
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def batch_text_to_sequence(text, language_code, group_size):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequences = []

  clean_text = _clean_text(text, lang_code[language_code])
  print(clean_text)
  clean_text_list = re.split(r"(?<=[.!?])\s+", clean_text.strip())

  grouped_sentences = []
  for i in range(0, len(clean_text_list), group_size):
      group = " ".join(clean_text_list[i:i + group_size])
      grouped_sentences.append(group)

  for c_text in grouped_sentences:
    sequence = [_symbol_to_id[symbol] for symbol in c_text]
    sequence = intersperse(sequence, 0) 
    sequences.append(sequence)
  return sequences


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, language_code):
  text = ipa_converter(text, language_code)
  return text