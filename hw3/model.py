from typing import List, Dict, Tuple


class Model:
    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        """
        --> !!! STUDENT: do NOT implement here your predict function (see StudentModel in hw3/implementation.py) !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
            {
                'id': 'train-1',
                'text': "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline.",
                'pron': 'her',
                'p_offset': 274,
                'entity_A': 'Cheryl Cassidy',
                'offset_A': 191,
                'is_coref_A': 'TRUE',
                'entity_B': 'Pauline',
                'offset_B': 207,
                'is_coref_B': 'FALSE'
            }
        Returns:
            A List with your predictions.
            Each prediction is a tuple, composed by two tuples:
            (ambigous_pronoun, ambiguous_pronoun_offset), (coreferent_entity, coreferent_entity_offset))
            for example:
                [(('her', 274), ('Pauline', 418))]

        """

        raise NotImplementedError
