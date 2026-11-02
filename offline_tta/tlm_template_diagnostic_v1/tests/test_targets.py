import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from targets import PREFIX, SUFFIX, target_groups, loss_mask, title_only


class TargetTests(unittest.TestCase):
    def example(self,truncated=False):
        user=PREFIX+'What is 2 + 2?'+('' if truncated else SUFFIX)
        text='<user>'+user+'<eot><assistant>'
        offsets=[(i,i+1) for i in range(len(text))]
        return text,offsets,6+len(user)

    def test_partition_has_three_disjoint_regions(self):
        text,offsets,end=self.example()
        groups=target_groups(text,offsets,end,False)
        self.assertEqual(set(groups),{'role','wrapper','body'})
        self.assertEqual(''.join(c for c,g in zip(text,groups) if g=='body'),'What is 2 + 2?')
        self.assertEqual(len(groups),len(offsets))

    def test_masks_change_only_requested_regions(self):
        groups=['role','wrapper','body','wrapper','role']
        self.assertEqual(loss_mask(groups,'all'),[True]*5)
        self.assertEqual(loss_mask(groups,'role_masked'),[False,True,True,True,False])
        self.assertEqual(loss_mask(groups,'wrapper_masked'),[True,False,True,False,True])
        self.assertEqual(loss_mask(groups,'both_masked'),[False,False,True,False,False])

    def test_truncated_tail_keeps_body(self):
        text,offsets,end=self.example(True)
        groups=target_groups(text,offsets,end,True)
        self.assertEqual(''.join(c for c,g in zip(text,groups) if g=='body'),'What is 2 + 2?')

    def test_unknown_or_missing_scaffold_rejected(self):
        with self.assertRaises(ValueError):target_groups('question',[(0,8)],8,False)
        text,offsets,end=self.example(True)
        with self.assertRaises(ValueError):target_groups(text,offsets,end,False)

    def test_boundary_token_is_kept_as_body(self):
        text,_,end=self.example()
        boundary=6+len(PREFIX)
        groups=target_groups(text,[(boundary-1,boundary+1)],end,False)
        self.assertEqual(groups,['body'])

    def test_title_only_is_not_substantive_answer(self):
        self.assertTrue(title_only('### Response:\n'))
        self.assertTrue(title_only('#### Response:\n\n### Response:\n'))
        self.assertFalse(title_only('### Response:\nThe answer is 4.'))
        self.assertFalse(title_only(''))


if __name__=='__main__':unittest.main()
