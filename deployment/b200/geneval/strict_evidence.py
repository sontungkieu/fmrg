"""Fail-closed evidence checks. A structural pass is not author-exact provenance."""
import collections,re,json
from pathlib import PurePosixPath
TAGS={'single_object','two_object','counting','colors','position','color_attr'}
HEX=re.compile(r'[0-9a-f]{64}')

def normalize_geneval_rows(rows,prompt_metadata):
 normalized=[]
 for row in rows:
  path=PurePosixPath(row['filename'].replace('\\','/'))
  assert path.parent.name=='samples' and path.parent.parent.name.isdigit() and re.fullmatch(r'[0-9]+\.png',path.name)
  prompt_id=int(path.parent.parent.name);seed=int(path.stem)
  meta=json.loads(row['metadata']) if isinstance(row['metadata'],str) else row['metadata']
  assert meta==prompt_metadata[prompt_id] and row['prompt']==meta['prompt'] and row['tag']==meta['tag']
  assert type(row['correct']) is bool
  normalized.append(dict(prompt_id=prompt_id,seed=seed,tag=row['tag'],correct=row['correct'],status='COMPLETE',source_filename=row['filename']))
 return normalized

def aggregate_geneval(rows,expected_cells,prompt_metadata):
 expected=set(map(tuple,expected_cells));assert len(expected)==len(expected_cells),'Duplicate expected cell'
 assert {p['tag'] for p in prompt_metadata.values()}==TAGS
 actual={};groups=collections.defaultdict(list)
 for row in rows:
  key=(row['prompt_id'],row['seed'])
  assert key in expected and key not in actual,'Unexpected/duplicate result cell'
  assert type(row['correct']) is bool,'Missing/invalid correctness is not zero'
  assert row['tag']==prompt_metadata[row['prompt_id']]['tag'],'Prompt/tag identity mismatch'
  assert row.get('status')=='COMPLETE','Failed row cannot be scored as false'
  actual[key]=row;groups[row['tag']].append(int(row['correct']))
 assert set(actual)==expected,'Missing expected result cells'
 assert set(groups)==TAGS,'All six categories required'
 scores={tag:dict(correct=sum(values),total=len(values),accuracy=sum(values)/len(values)) for tag,values in groups.items()}
 return dict(status='COMPLETE_COUNTS_VERIFIED',cells=len(actual),category_scores=scores,
  overall=sum(x['accuracy'] for x in scores.values())/6,
  aggregation='Pool specified seeds within category, then unweighted mean of six category accuracies',
  equivalence_claim=False)

def validate_inverse_split(manifest,require_measurements=True):
 assert manifest['protocol']=='fmrg-dataset-freeze-v1'
 assert manifest['claim'] in ['author_exact','reconstructed']
 assert HEX.fullmatch(manifest['preprocessing_sha256'])
 seen_ids=set();seen_content={};counts=collections.Counter()
 for row in manifest['rows']:
  dataset,split=row['dataset'],row['split'];assert dataset in ['AFHQ-Dog','FFHQ'] and split in ['tuning','evaluation']
  key=(dataset,row['original_id']);assert key not in seen_ids,'Duplicate dataset image ID'
  seen_ids.add(key);counts[(dataset,split)]+=1
  for name in ['raw_sha256','preprocessed_sha256']:
   digest=row[name];assert HEX.fullmatch(digest)
   content=(dataset,name,digest)
   assert content not in seen_content,'Duplicate content or tuning/evaluation leakage'
   seen_content[content]=split
  assert re.fullmatch('[0-9a-f]{40}',row['source_revision'])
  assert row['source_repo'] and row['source_file']
  if require_measurements:
   observations=row['observations'];assert set(observations)=={'sr4','motion61','mask64'}
   for obs in observations.values():
    assert type(obs['noise_seed']) is int and obs['noise_sigma']==.03
    for name in ['operator_config_sha256','measurement_sha256']:assert HEX.fullmatch(obs[name])
 for dataset in ['AFHQ-Dog','FFHQ']:
  assert counts[(dataset,'tuning')]==50 and counts[(dataset,'evaluation')]==1000,'Split count mismatch'
 if manifest['claim']=='author_exact':
  assert manifest.get('author_split_evidence') and HEX.fullmatch(manifest['author_split_evidence_sha256'])
 return dict(status='STRUCTURAL_PASS',claim=manifest['claim'],author_provenance_independently_verified=False,measurement_hashes_verified=require_measurements,
             full_launch_allowed=False,counts={f'{d}/{s}':n for (d,s),n in counts.items()})
