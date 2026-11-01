import copy
import json
from pathlib import Path
import tempfile
import unittest

from inverse_campaign import batches, canonical, dump_new, find_complete, make_cells, preflight_files, safe_relative, sha
from start_inverse_campaign import worker_args


class CampaignGuards(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((Path(__file__).parent/'INVERSE_RECONSTRUCTED_CONTRACT.json').read_text())
        self.contract['eval_images_per_dataset']=3
        self.contract['tuning_images_per_dataset']=1
        obs=dict(path='x.safetensors',measurement_sha256='a'*64,tensor_payload_sha256='b'*64,
                 operator_state_sha256='c'*64,operator_config_sha256='d'*64)
        self.man=dict(claim='reconstructed',rows=[])
        for ds in self.contract['datasets']:
            for split,ids in [('tuning',['t0']),('evaluation',['e0','e1','e2'])]:
                for identity in ids:
                    self.man['rows'].append(dict(dataset=ds,split=split,original_id=identity,
                        path=f'{ds}/{split}/{identity}.png',png_sha256='e'*64,
                        observations={k:dict(obs) for k in self.contract['tasks']}))

    def test_exact_expected_cells_and_no_tuning_leakage(self):
        cells=make_cells(self.man,self.contract,'eval')
        self.assertEqual(len(cells),36)
        self.assertTrue(all(x['split']=='evaluation' and x['seed']==0 for x in cells))
        self.assertEqual(len({x['cell_id'] for x in cells}),36)
        self.assertEqual(cells,make_cells(copy.deepcopy(self.man),self.contract,'eval'))

    def test_reject_missing_duplicate_and_overlapping_images(self):
        for mutate in ['missing','duplicate','leakage']:
            m=copy.deepcopy(self.man)
            if mutate=='missing':m['rows'].pop()
            if mutate=='duplicate':m['rows'].append(copy.deepcopy(m['rows'][-1]))
            if mutate=='leakage':m['rows'][0]['original_id']='e0'
            with self.assertRaises(AssertionError):make_cells(m,self.contract,'eval')

    def test_shards_disjoint_exhaustive_and_homogeneous(self):
        cells=make_cells(self.man,self.contract,'eval')
        assigned=[]
        for rank in range(4):
            for job in batches(cells,2,rank,4):
                self.assertEqual(len({(x['dataset'],x['task'],x['method']) for x in job}),1)
                assigned.extend(x['cell_id'] for x in job)
        self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(set(assigned),{x['cell_id'] for x in cells})

    def test_resume_requires_verified_artifacts_and_exact_identity(self):
        cell=make_cells(self.man,self.contract,'eval')[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'cells'/cell['cell_id'];d.mkdir(parents=True)
            with self.assertRaises(ValueError):find_complete([root],cell,'contract','runner')
            files={}
            for name in ['prediction.png','prediction.safetensors','latent.safetensors']:
                (d/name).write_bytes(name.encode());files[name]=sha(d/name)
            dump_new(d/'record.json',dict(status='COMPLETE',cell=cell,contract_sha256='contract',runner_sha256='runner',files_sha256=files))
            self.assertIsNotNone(find_complete([root],cell,'contract','runner'))
            with self.assertRaises(AssertionError):find_complete([root],cell,'wrong','runner')
            with self.assertRaises(ValueError):find_complete([root,root],cell,'contract','runner')
            (d/'prediction.png').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):find_complete([root],cell,'contract','runner')

    def test_path_traversal_and_overwrites_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for unsafe in ['../x','/x','a/../../x']:
                with self.assertRaises(ValueError):safe_relative(tmp,unsafe)
            p=Path(tmp)/'x.json';dump_new(p,dict(v=1))
            with self.assertRaises(FileExistsError):dump_new(p,dict(v=2))
            self.assertEqual(json.loads(p.read_text()),dict(v=1))

    def test_missing_image_is_rejected_by_cpu_preflight(self):
        cell=make_cells(self.man,self.contract,'eval')[0]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                preflight_files([cell],dict(splits=root,obs=root))

    def test_e_only_stage_does_not_assign_j(self):
        cells=[c for c in make_cells(self.man,self.contract,'eval') if c['method']=='E']
        ids=[c['cell_id'] for rank in range(4) for group in batches(cells,2,rank,4) for c in group]
        self.assertEqual(len(ids),18)
        self.assertEqual(set(ids),{c['cell_id'] for c in cells})
        args=worker_args(dict(contract='contract',plan='plan',allocation='alloc',gpu_uuids=['a','b','c','d'],batch_size=16,methods=['E']),Path('out'),2,False)
        self.assertEqual(args[args.index('--methods')+1:args.index('--methods')+2],['E'])
        self.assertEqual(args[args.index('--gpu')+1],'c')


if __name__=='__main__':unittest.main()
