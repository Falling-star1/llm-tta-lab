import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class QualityTests(unittest.TestCase):
    def rows(self,n):
        return [{'source_index':i,'predict':'answer '+str(i)} for i in range(n)]

    def test_detects_empty_and_word_loops_against_same_source(self):
        from quality import quality_window
        for kind,text in [('empty',''),('repetition','Pesach '*100)]:
            rows=self.rows(100)
            for row in rows[:6]:
                row['predict']=text
            result=quality_window(rows,self.rows(100))
            self.assertEqual(result['status'],'ALERT')
            self.assertIn(kind,result['alerts'])
            self.assertEqual(quality_window(rows,rows)['status'],'PASS')

    def test_does_not_apply_before_100_outputs(self):
        from quality import quality_window
        self.assertEqual(quality_window(self.rows(99),self.rows(100))['status'],'WAIT')

    def test_checks_latest_100_instead_of_diluting_late_failure(self):
        from quality import quality_window
        rows=self.rows(200)
        for row in rows[100:106]:
            row['predict']=''
        result=quality_window(rows,self.rows(200))
        self.assertEqual(result['status'],'ALERT')
        self.assertEqual(result['window_start'],100)
        self.assertEqual(result['tlm']['empty'],6)

    def test_rejects_misaligned_source(self):
        from quality import quality_window
        source=self.rows(100)
        source[0]['source_index']=5
        with self.assertRaises(ValueError):
            quality_window(self.rows(100),source)

    def test_detects_one_short_answer_collapsing_across_different_questions(self):
        from quality import quality_window
        rows=self.rows(100)
        for row in rows[:60]:
            row['predict']="I'm happy to help."
        report=quality_window(rows,self.rows(100))
        self.assertEqual(report['status'],'ALERT')
        self.assertIn('dominant_answer',report['alerts'])
        self.assertEqual(report['dominant_answer']['tlm_count'],60)
        self.assertEqual(quality_window(rows,rows)['status'],'PASS')

if __name__=='__main__':
    unittest.main()
