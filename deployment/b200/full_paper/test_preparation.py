"""Offline standard-library tests for contract and download safety boundaries."""
import copy,json,unittest
from pathlib import Path
from download_assets import select_files

class PreparationTests(unittest.TestCase):
 def setUp(self):
  self.m={'failures':[],'files':[dict(component='metrics',filename='w.pth',bytes=4,sha256='a'*64,path_in_repo='full-paper/20261029/assets/metrics/w.pth')]}
 def test_incomplete_component_fails_before_transfer(self):
  self.m['failures']=[dict(component='metrics',filename='missing')]
  with self.assertRaises(AssertionError):select_files(self.m,{'metrics'})
 def test_unknown_component(self):
  with self.assertRaises(AssertionError):select_files(self.m,{'vlm'})
 def test_traversal_and_url_substitution(self):
  for key,value in [('filename','../x'),('filename','C:\\x'),('path_in_repo','https://evil.invalid/x'),('sha256','bad')]:
   m=copy.deepcopy(self.m);m['files'][0][key]=value
   with self.assertRaises(AssertionError):select_files(m,{'metrics'})
 def test_valid_selection(self):self.assertEqual(len(select_files(self.m,{'metrics'})),1)
 def test_contract_retains_launch_block_and_track_distinction(self):
  c=json.loads(Path(__file__).with_name('CONTRACT.json').read_text(encoding='utf-8'))
  self.assertFalse(c['full_launch_allowed']);self.assertEqual(len(c['tracks']),2)
  self.assertGreaterEqual(len(c['blockers']),8)
  self.assertNotEqual(c['tracks']['paper_formula_v1']['E_weight'],c['tracks']['released_code_v1']['E_weight'])
if __name__=='__main__':unittest.main()
