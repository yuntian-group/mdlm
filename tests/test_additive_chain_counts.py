"""True count construction, scale cancellation and portable backend guards."""
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from additive_chain.counts import SparseCountPotential
from additive_chain.candidates import CompactCandidates
from additive_chain.cuda import sample_additive_count, dense_pair_reference
from additive_chain import _count_inference as cache


def example(mode='conditional',strength=.005,smoothing=.1):
    counts=torch.tensor([[0.,7.,0.,2.],[3.,0.,4.,0.],[0.,2.,0.,0.],[1.,0.,0.,0.]],dtype=torch.float64)
    keys=counts.flatten().nonzero().flatten()
    pot=SparseCountPotential.from_counts(counts.sum(1),counts.sum(0),keys,
        counts.flatten()[keys],mode=mode,strength=strength,smoothing=smoothing)
    return counts,pot


@pytest.mark.parametrize('mode',['conditional','pmi'])
@pytest.mark.parametrize('strength',[0.,.005,.1,1.,2.])
@pytest.mark.parametrize('smoothing',[.1,1.])
def test_true_smoothed_count_formula_and_absolute_scale(mode,strength,smoothing):
    counts,pot=example(mode,strength,smoothing)
    n=counts.sum()
    bl=(counts.sum(1)+1)/(n+4)
    br=(counts.sum(0)+1)/(n+4)
    joint=(1-smoothing)*counts/n+smoothing*bl[:,None]*br[None,:]
    expected=joint.log()-joint.sum(1).log()[:,None]
    if mode=='pmi': expected-=joint.sum(0).log()[None,:]
    expected*=strength
    actual=pot.log_scores(torch.arange(4)[:,None],torch.arange(4)[None,:])
    torch.testing.assert_close(actual,expected,atol=3e-15,rtol=3e-15)
    # The CUDA score oracle omits only a shared edge constant, never row terms.
    ids=torch.arange(4)[None,None].expand(1,2,4)
    candidates=CompactCandidates(ids,torch.zeros(1,2,4,dtype=torch.float64),4)
    compact=dense_pair_reference(candidates,pot)[0,0].double()+pot.log_scale
    torch.testing.assert_close(compact,expected,atol=5e-7,rtol=2e-7)


def test_empty_counts_and_strength_zero_have_declared_meaning():
    for mode in ('conditional','pmi'):
        p=SparseCountPotential.from_counts(torch.zeros(4),torch.zeros(4),
            torch.empty(0,dtype=torch.long),torch.empty(0),mode=mode,strength=1.)
        expected=torch.full((4,4),-torch.log(torch.tensor(4.)).item() if mode=='conditional' else 0.,dtype=torch.float64)
        torch.testing.assert_close(p.log_scores(torch.arange(4)[:,None],torch.arange(4)[None,:]),expected)
    _,p=example(strength=0.)
    assert p.pair_keys.numel()==0 and p.log_scale==0
    assert torch.equal(p.left,torch.ones(4,dtype=torch.float64))
    assert torch.equal(p.right,torch.ones(4,dtype=torch.float64))


def test_reference_adapter_preserves_csr_order_and_scale_without_old_modules():
    _,p=example()
    row,col=p.pair_keys//4,p.pair_keys%4
    sparse=torch.sparse_coo_tensor(torch.stack((row,col)),p.row_values,(4,4)).coalesce().to_sparse_csr()
    ref=SimpleNamespace(left=p.left,right=p.right,pair_keys=p.pair_keys,
                        sparse=sparse,vocab_size=4,log_scale=p.log_scale)
    copied=SparseCountPotential.from_reference(ref)
    assert copied.left is p.left and copied.right is p.right
    assert torch.equal(copied.row_values,p.row_values) and copied.log_scale==p.log_scale
    ref.pair_keys=ref.pair_keys.flip(0)
    with pytest.raises(ValueError,match='CSR correction order'):
        SparseCountPotential.from_reference(ref)


def test_large_pair_keys_do_not_overflow_int32():
    vocab=50258
    keys=torch.tensor([0,vocab*vocab-1],dtype=torch.long)
    lc=torch.zeros(vocab,dtype=torch.float64); lc[0]=3; lc[-1]=7
    pot=SparseCountPotential.from_counts(lc,lc.clone(),keys,torch.tensor([3.,7.]))
    assert pot.pair_keys[-1].item()>2**31 and pot.pair_keys.dtype==torch.long
    assert torch.isfinite(pot.log_scores(torch.tensor([vocab-1]),torch.tensor([vocab-1]))).all()


def test_bad_counts_and_mutated_cached_potential_are_rejected():
    counts,pot=example()
    keys=counts.flatten().nonzero().flatten()
    with pytest.raises(ValueError,match='marginals'):
        SparseCountPotential.from_counts(counts.sum(1)+1,counts.sum(0),keys,counts.flatten()[keys])
    with pytest.raises(ValueError,match='sorted'):
        SparseCountPotential.from_counts(counts.sum(1),counts.sum(0),keys.flip(0),counts.flatten()[keys].flip(0))
    cache.clear_count_cache()
    cache._TABLES[id(pot)]=(pot,pot.signature(),None,None,None,None)
    pot.left.add_(1.)
    packet=SimpleNamespace(unary=torch.zeros(1,1,1))
    with pytest.raises(ValueError,match='mutated'):
        cache._factors(packet,pot)
    cache.clear_count_cache()
    with torch.inference_mode(),pytest.raises(ValueError,match='outside torch.inference_mode'):
        SparseCountPotential.from_counts(torch.zeros(2),torch.zeros(2),torch.empty(0,dtype=torch.long),torch.empty(0))


def test_core_import_does_not_require_cuda_and_cuda_does_not_fall_back():
    source='import sys; import additive_chain; assert "triton" not in sys.modules; assert "additive_chain.cuda" not in sys.modules'
    subprocess.run([sys.executable,'-c',source],check=True)
    _,pot=example()
    candidates=CompactCandidates(torch.tensor([[[0,1],[0,1]]]),torch.zeros(1,2,2,dtype=torch.float64),4)
    with pytest.raises(ValueError,match='NVIDIA CUDA'):
        sample_additive_count(candidates,pot)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hash-key gate is not a CPU test')
@pytest.mark.parametrize('vocab',[50258,65536])
def test_cuda_hash_high_int32_bits_and_int64_sentinel_boundary(vocab):
    from additive_chain import _count_kernels as kernels
    if kernels.triton is None: pytest.skip('Triton is required for CUDA hash tests')
    keys=torch.tensor([0,vocab*vocab-1],dtype=torch.long,device='cuda')
    values=torch.tensor([.4,-.7],dtype=torch.float32,device='cuda')
    hk,hv=kernels._build_count_hash(keys,values,vocab)
    assert hk.dtype==(torch.int32 if vocab==50258 else torch.int64)
    left=torch.zeros(vocab,dtype=torch.float64,device='cuda')
    right=left.clone()
    ids=torch.tensor([0,vocab-1],device='cuda')
    actual=kernels._lookup_count_pairs(ids,ids,hk,hv,left,right,vocab)
    assert torch.equal(actual,values)
