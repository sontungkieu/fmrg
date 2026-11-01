import copy
import json
from pathlib import Path
import tempfile
import unittest

from inverse_campaign import batches, canonical, dump_new, find_complete, make_cells, safe_relative, sha


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


if __name__=='__main__':unittest.main()
