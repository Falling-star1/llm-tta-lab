import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import official_stop_ids, decode_official_prefix, answer_flags


class ProtocolTests(unittest.TestCase):
    def tokenizer(self):
        return SimpleNamespace(eos_token_id=9,additional_special_tokens_ids=[7,9,8],
                               decode=lambda ids,skip_special_tokens:''.join(str(i) for i in ids if i not in [7,8,9]))

    def test_official_eos_and_additional_tokens_deduplicated(self):
        self.assertEqual(official_stop_ids(self.tokenizer()),[9,7,8])

    def test_earliest_official_stop_truncates_saved_greedy_prefix(self):
        r=decode_official_prefix([1,2,7,3,9],self.tokenizer())
        self.assertEqual(r['generated_ids'],[1,2,7]);self.assertEqual(r['predict'],'12')
        self.assertTrue(r['shortened']);self.assertEqual(r['first_stop_position'],2)

    def test_normal_eos_or_no_stop_keeps_ids(self):
        for ids in [[1,2,9],[1,2]]:
            self.assertFalse(decode_official_prefix(ids,self.tokenizer())['shortened'])

    def test_invalid_ids_rejected(self):
        for ids in [[True],[1,-1],[1,'2']]:
            with self.assertRaises(ValueError):decode_official_prefix(ids,self.tokenizer())

    def test_empty_and_title_only_are_separate(self):
        self.assertEqual(answer_flags(''),{'empty':True,'title_only':False,'no_substantive_answer':True})
        self.assertEqual(answer_flags('### Response:\n'),{'empty':False,'title_only':True,'no_substantive_answer':True})
        self.assertFalse(answer_flags('### Response:\nIt is 4.')['no_substantive_answer'])


if __name__=='__main__':unittest.main()
