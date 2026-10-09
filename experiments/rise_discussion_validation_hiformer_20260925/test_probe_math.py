import unittest
import numpy as np
from probe_math import eligibility,crosspart_templates,crosspart_donor_indices,match_moments,moments,row_diagnostics


class MaskedControlsTest(unittest.TestCase):
    def setUp(self):
        self.parts=np.repeat(np.arange(8),4)
        self.delta=np.random.default_rng(4).normal(size=(32,5))
        self.masks=np.ones((32,5),dtype=bool)
        self.masks[::2,-2:]=False
        self.masks[0]=False
        self.masks[1]=False;self.masks[1,0]=True

    def test_eligibility_and_excluded_rows(self):
        eligible,peers=eligibility(self.parts,self.masks)
        self.assertEqual(int(eligible.sum()),30)
        self.assertEqual(peers[:2].tolist(),[0,0])
        self.assertTrue(eligible[2:].all())

    def test_crossfit_raw_and_rms_mask_preservation(self):
        raw,rms,receipts=crosspart_templates(self.delta,self.parts,self.masks)
        eligible,_=eligibility(self.parts,self.masks)
        for result in (raw,rms):
            np.testing.assert_array_equal(result[~self.masks],self.delta[~self.masks])
            np.testing.assert_array_equal(result[~eligible],self.delta[~eligible])
        for a,b in zip(moments(rms,self.masks),moments(self.delta,self.masks)):
            np.testing.assert_allclose(a,b,atol=1e-12)
        i=2; peers=np.flatnonzero((self.parts!=self.parts[i])&np.all(self.masks==self.masks[i],axis=1))
        expected=self.delta[peers][:,self.masks[i]].mean(0)
        expected=expected-expected.mean()+self.delta[i,self.masks[i]].mean()
        np.testing.assert_allclose(raw[i,self.masks[i]],expected)
        changed=self.delta.copy();changed[self.parts==0]+=1000
        changed_raw,_,_=crosspart_templates(changed,self.parts,self.masks)
        np.testing.assert_allclose(changed_raw[i,self.masks[i]]-1000,raw[i,self.masks[i]],atol=1e-12)

    def test_seeded_donors_are_same_mask_cross_part_and_reproducible(self):
        eligible,_=eligibility(self.parts,self.masks)
        donor=crosspart_donor_indices(self.parts,self.masks,20260925)
        np.testing.assert_array_equal(donor,crosspart_donor_indices(self.parts,self.masks,20260925))
        self.assertTrue(np.all(self.parts[donor[eligible]]!=self.parts[eligible]))
        np.testing.assert_array_equal(self.masks[donor[eligible]],self.masks[eligible])
        np.testing.assert_array_equal(donor[:2],np.arange(2))
        matched=match_moments(self.delta[donor],self.delta,self.masks,eligible)
        for a,b in zip(moments(matched,self.masks),moments(self.delta,self.masks)):
            np.testing.assert_allclose(a,b,atol=1e-12)

    def test_empty_weight_diagnostics_undefined_without_corrupting_eligible(self):
        eligible,_=eligibility(self.parts,self.masks)
        weights=np.divide(self.masks,self.masks.sum(1,keepdims=True),out=np.full((32,5),np.nan),where=self.masks.sum(1,keepdims=True)>0)
        result=row_diagnostics(self.delta,self.delta,weights,weights,self.masks,eligible)
        for name,values in result.items():
            self.assertTrue(np.isnan(values[:2]).all(),name)
            self.assertTrue(np.isfinite(values[2:]).all(),name)
        np.testing.assert_allclose(result['attention_tv_to_full'][2:],0)


if __name__=='__main__':unittest.main()
