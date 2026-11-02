import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import select_indices,paired_counts,require_control,is_stub,require_assets,replay_references,protect_output


class ProtocolTests(unittest.TestCase):
    def test_selection_blind_fixed_unique_and_excluded(self):
        first=select_indices(5000,list(range(116)))
        self.assertEqual(len(first),200)
        self.assertEqual(first,sorted(set(first)))
        self.assertTrue(all(116<=i<5000 for i in first))
        self.assertEqual(first,select_indices(5000,list(reversed(range(116)))))

    def test_selection_rejects_invalid_population_and_exclusions(self):
        for excluded in [[-1],[5000],[True],[1,1]]:
            with self.assertRaises(ValueError):select_indices(5000,excluded)
        for total,count in [(0,200),(5000,0),(5,200),(True,200)]:
            with self.assertRaises(ValueError):select_indices(total,[],count=count)

    def rows(self,scores,stubs):
        return [{'source_index':i,'score':s,'stub_only':b} for i,(s,b) in enumerate(zip(scores,stubs))]

    def test_paired_gains_losses_and_stop_restoration(self):
        base=self.rows([1,1,0,0],[False,False,True,True])
        control=self.rows([1,0,1,0],[False,False,False,True])
        result=paired_counts(base,control)
        self.assertEqual(result,{'n':4,'baseline_correct':2,'control_correct':2,'wrong_to_right':1,'right_to_wrong':1,
                                 'both_correct':1,'both_wrong':1,'stub_removed':1,'stub_introduced':0})

    def test_stub_to_empty_is_only_stub_removed_not_answer_restored(self):
        result=paired_counts(self.rows([0],[True]),self.rows([0],[False]))
        self.assertEqual(result['stub_removed'],1)
        self.assertNotIn('stub_restored',result)

    def test_pair_rejects_misaligned_indexes_and_nonbinary_scores(self):
        base=self.rows([1,0],[False,True])
        for control in [list(reversed(base)),base[:1],[dict(base[0]),dict(base[0])]]:
            with self.assertRaises(ValueError):paired_counts(base,control)
        control=copy.deepcopy(base);control[0]['score']=.5
        with self.assertRaises(ValueError):paired_counts(base,control)

    def control(self):
        return ({'status':'COMPLETE','mode':'eot_masked','steps':2500,'diagnostic_only':True,'lr':1e-6,
                 'controls':{'sample_order_exact':True,'learning_rates_exact':True,'max_base_ce_difference':0.,'max_weight_difference':0.},
                 'optimizer':{'successful_optimizer_steps':2500,'skipped_optimizer_steps':0,'adapter_parameters_finite':True,'optimizer_state_finite':True},
                 'original_runs_modified':False},
                {'original':'/original','mode':'eot_masked','fresh_initial_adapter_exact':True})

    def test_control_recipe_requires_complete_and_exact_trajectory(self):
        summary,manifest=self.control();require_control(summary,manifest,'eot_masked','/original')
        for key,value in [('status','TRAINING'),('steps',1000),('lr',5e-5),('mode','hint_masked')]:
            bad=copy.deepcopy(summary);bad[key]=value
            with self.assertRaises(ValueError):require_control(bad,manifest,'eot_masked','/original')
        for key,value in [('learning_rates_exact',False),('max_base_ce_difference',1e-5),('max_weight_difference',1e-5)]:
            bad=copy.deepcopy(summary);bad['controls'][key]=value
            with self.assertRaises(ValueError):require_control(bad,manifest,'eot_masked','/original')
        bad=copy.deepcopy(summary);bad['optimizer']['skipped_optimizer_steps']=1
        with self.assertRaises(ValueError):require_control(bad,manifest,'eot_masked','/original')

    def test_stub_detection_does_not_count_valid_reasoning(self):
        self.assertTrue(is_stub("Let's think step by step."))
        self.assertFalse(is_stub("Let's think step by step.\n2+2=4"))
        self.assertFalse(is_stub(''))

    def test_asset_coverage_rejects_missing_required_and_bad_hashes(self):
        good={'/original/metrics.json':'a'*64,'/source/source.jsonl':'b'*64}
        require_assets(good,set(good))
        with self.assertRaises(ValueError):require_assets(good,set(good)|{'/inputs/inputs.jsonl'})
        for bad in [{}, {'/original/metrics.json':'xyz'}, {'/original/metrics.json':'g'*64}]:
            with self.assertRaises(ValueError):require_assets(bad,{'/original/metrics.json'})

    def test_replay_references_require_complete_distinct_valid_answers(self):
        rows=[{'source_index':i,'generated_ids':[i+10],'predict':'answer'} for i in [2,4,8]]
        self.assertEqual(replay_references(rows,[2,4,8]),rows[:2])
        for bad in [[],rows[:1],rows[:2],[rows[0],rows[0],rows[2]]]:
            with self.assertRaises(ValueError):replay_references(bad,[2,4,8])
        bad=copy.deepcopy(rows);bad[0]['generated_ids']=[]
        with self.assertRaises(ValueError):replay_references(bad,[2,4,8])

    def test_output_protects_source_inputs_and_original_before_writing(self):
        root=Path.cwd()/'protocol-fixture'
        protected=[root/'original',root/'source',root/'inputs']
        protect_output(root/'new',protected)
        for p in protected:
            with self.assertRaises(ValueError):protect_output(p/'accidental-output',protected)


if __name__=='__main__':unittest.main()
