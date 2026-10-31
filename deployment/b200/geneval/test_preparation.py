import copy,hashlib,io,json,socket,tarfile,unittest
from pathlib import PurePosixPath
from unittest.mock import patch
from build_overlay import command,extract
from offline_adapter import network_denied,PINS,bind_evaluator_globals
from strict_evidence import aggregate_geneval,TAGS,validate_inverse_split,normalize_geneval_rows
from restore_overlay import validate_members
from download_publication import select

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
 def test_image_only_freeze_cannot_certify_measurements(self):
  rows=[]
  for dataset in ['AFHQ-Dog','FFHQ']:
   for i in range(1050):
    rows.append(dict(dataset=dataset,split='tuning' if i<50 else 'evaluation',original_id=str(i),
     raw_sha256=f'{i:064x}',preprocessed_sha256=f'{i+1050:064x}',source_revision='a'*40,source_repo='example/data',source_file=str(i)+'.png'))
  manifest=dict(protocol='fmrg-dataset-freeze-v1',claim='reconstructed',preprocessing_sha256='b'*64,rows=rows)
  self.assertFalse(validate_inverse_split(manifest,require_measurements=False)['measurement_hashes_verified'])
  with self.assertRaises((AssertionError,KeyError)):validate_inverse_split(manifest)
  for row in rows:
   row['observations']={k:dict(noise_seed=1,noise_sigma=.03,operator_config_sha256='c'*64,measurement_sha256='d'*64) for k in ['sr4','motion61','mask64']}
  self.assertTrue(validate_inverse_split(manifest)['measurement_hashes_verified'])

class PublicationTests(unittest.TestCase):
 def setUp(self):
  self.manifest=dict(files=[dict(group='runtime',filename='a.tar.gz',sha256='a'*64,bytes=10,
   url='https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable/resolve/'+'b'*40+'/full-paper/20261101/runtime/a.tar.gz')])
 def test_external_download_url_is_rejected(self):
  self.manifest['files'][0]['url']='https://example.invalid/file'
  with self.assertRaises(AssertionError):select(self.manifest,{'runtime'})
 def test_duplicate_or_traversal_destination_is_rejected(self):
  for change in ['duplicate','traversal']:
   m=copy.deepcopy(self.manifest)
   if change=='duplicate':m['files'].append(m['files'][0])
   else:m['files'][0]['filename']='../escape'
   with self.assertRaises(AssertionError):select(m,{'runtime'})

if __name__=='__main__':unittest.main()
