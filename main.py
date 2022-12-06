import sys
import os

from preprocess import preprocess_data
from model import model_training, final_model_training

def main(targets):
  preprocess_data()

  if len(targets) == 0:
    final_model_training()

  if 'train' in targets:
    model_training()

  return

if __name__ == '__main__':
  targets = sys.argv[1:]
  main(targets)