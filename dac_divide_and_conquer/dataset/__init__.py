from enum import Enum
class Augmentation(Enum):
    ner_sentence = "ner_sentence"
    ner_stripped = "ner_stripped"
    ner_mention = "ner_mention"
    descriptions_labels = "descriptions_labels"

from .base import DACCorpus
from .cantemist import CantemistCorpus
from .codiesp import CodiespCorpus, CodiespSubtask
from .falp import FALPCorpus
from .mesinesp import MESINESPCorpus, MesinespSubtask
