import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from masking import HINT,transition_mask


class MaskingTests(unittest.TestCase):
    def example(self):
        text='<user>What is 2+2?\n\n### Response: '+HINT+'<eot><assistant>'
        return text,[(i,i+1) for i in range(len(text))],text.index('<eot>')

    def test_eot_control_removes_only_user_terminator_target(self):
        text,offsets,eot=self.example();mask=transition_mask(text,offsets,eot,'eot_masked')
        self.assertEqual([i for i,v in enumerate(mask) if not v],[eot])

    def test_hint_control_keeps_eot_and_question(self):
        text,offsets,eot=self.example();mask=transition_mask(text,offsets,eot,'hint_masked')
        self.assertTrue(mask[eot]);self.assertTrue(all(mask[:text.index(HINT)]))
        self.assertEqual(''.join(c for c,v in zip(text,mask) if not v),HINT)

    def test_both_controls_are_union(self):
        text,offsets,eot=self.example()
        a=transition_mask(text,offsets,eot,'eot_masked');b=transition_mask(text,offsets,eot,'hint_masked')
        self.assertEqual(transition_mask(text,offsets,eot,'both_masked'),[x and y for x,y in zip(a,b)])

    def test_noncanonical_suffix_rejected(self):
        with self.assertRaises(ValueError):transition_mask('question<eot>',[(0,8),(8,13)],1,'eot_masked')


if __name__=='__main__':unittest.main()
