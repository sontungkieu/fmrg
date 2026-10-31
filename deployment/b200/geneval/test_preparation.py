import copy,hashlib,io,json,socket,tarfile,unittest
from pathlib import PurePosixPath
from unittest.mock import patch
from build_overlay import command,extract
from offline_adapter import network_denied,PINS,bind_evaluator_globals
from strict_evidence import aggregate_geneval,TAGS,validate_inverse_split,normalize_geneval_rows
from restore_overlay import validate_members

class RuntimeTests(unittest.TestCase):
 def test_build_logging_accepts_path(self):
  with patch('build_overlay.subprocess.run') as mock:
   command(['tar',PurePosixPath('/data/archive.tar.gz')])
   self.assertEqual(mock.call_args.args[0],['tar','/data/archive.tar.gz'])
 def test_tcp_network_denied_before_connect(self):
  with network_denied(),socket.socket() as sock:
   with self.assertRaisesRegex(RuntimeError,'network access denied'):sock.connect(('127.0.0.1',9))
 def test_detector_names_are_hf_aliases(self):
  self.assertEqual(PINS['r50'][0],'mask2former_r50_lsj_8x2_50e_coco.pth')
 def test_vendor_global_classnames_are_bound(self):
  ns={};classes=['car','person'];bind_evaluator_globals(ns,None,None,None,None,classes)
  self.assertIs(ns['classnames'],classes)
 def test_overlay_cannot_escape_prefix(self):
  for name in ['../etc/passwd','opt/fmrg/source/a','/opt/fmrg-geneval/a']:
   with self.assertRaises(AssertionError):validate_members([tarfile.TarInfo(name)])
  safe=tarfile.TarInfo('opt/fmrg-geneval/venv/bin/python');safe.type=tarfile.SYMTYPE
  safe.linkname='/opt/fmrg/python/bin/python3.11';validate_members([safe])
  safe.linkname='/etc/passwd'
  with self.assertRaises(AssertionError):validate_members([safe])

class AggregationTests(unittest.TestCase):
 def setUp(self):
  self.meta={i:dict(tag=tag) for i,tag in enumerate(sorted(TAGS))}
  self.expected=[(i,seed) for i in self.meta for seed in [0,1]]
  self.rows=[dict(prompt_id=i,seed=s,tag=self.meta[i]['tag'],correct=(i%2==0),status='COMPLETE') for i,s in self.expected]
 def test_category_mean(self):self.assertEqual(aggregate_geneval(self.rows,self.expected,self.meta)['overall'],.5)
 def test_missing_duplicate_and_failed_are_rejected(self):
  for rows in [self.rows[:-1],self.rows+[self.rows[0]],copy.deepcopy(self.rows)]:
   if len(rows)==len(self.rows):rows[0]['status']='FAILED'
   with self.assertRaises(AssertionError):aggregate_geneval(rows,self.expected,self.meta)
 def test_missing_categories_are_rejected(self):
  with self.assertRaises(AssertionError):aggregate_geneval([],[],{})
 def test_official_result_identity_normalization(self):
  meta={0:dict(prompt='a red car',tag='colors',include=[dict(count=1,**{'class':'car'})])}
  row=dict(filename='/runs/x/00000/samples/00003.png',metadata=json.dumps(meta[0]),prompt='a red car',tag='colors',correct=True)
  normalized=normalize_geneval_rows([row],meta)
  self.assertEqual((normalized[0]['prompt_id'],normalized[0]['seed']),(0,3))
  row['metadata']=json.dumps(dict(prompt='different',tag='colors'))
  with self.assertRaises(AssertionError):normalize_geneval_rows([row],meta)

class SplitTests(unittest.TestCase):
 def test_empty_split_never_certified(self):
  with self.assertRaises(AssertionError):validate_inverse_split(dict(protocol='fmrg-dataset-freeze-v1',claim='reconstructed',preprocessing_sha256='a'*64,rows=[]))

if __name__=='__main__':unittest.main()
