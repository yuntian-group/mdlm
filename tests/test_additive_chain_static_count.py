"""Static count cache score, ownership, budget and integer-address contracts."""
import importlib
import subprocess
import sys

import pytest
import torch

from additive_chain.counts import SparseCountPotential
from additive_chain.candidates import CompactCandidates
from additive_chain import _count_inference as reference
from additive_chain import _dense_inference as dense
from additive_chain import static


def potential(device='cpu',strength=.1):
    counts=torch.tensor([[9.,1.,0.],[2.,7.,3.],[0.,4.,8.]],dtype=torch.float64,device=device)
    keys=counts.flatten().nonzero().flatten()
    return SparseCountPotential.from_counts(counts.sum(1),counts.sum(0),keys,counts.flatten()[keys],
                                            mode='conditional',strength=strength)


def test_count_table_equals_existing_score_conversion_and_omits_only_global_constant():
    p=potential()
    table=static.prepare_static_count_cache(p,max_temporary_bytes=3*12)
    keys,observed,left,right=reference.count_log_components(p)
    expected=(left[:,None]+right[None,:]).float();expected.view(-1)[keys]=observed.float()
    assert torch.equal(table.log_scores,expected)
    assert table.omitted_log_scale==p.log_scale and table.owner is p
    assert table.diagnostics['table_bytes']==36 and table.diagnostics['rows_per_build_chunk']==1
    assert not table.diagnostics['persistent_files_written']
    # Explicit builds never return a globally cached table that another caller
    # could silently mutate while adding its learned residual.
    second=static.prepare_static_count_cache(p)
    assert second is not table and second.log_scores.data_ptr()!=table.log_scores.data_ptr()
    assert torch.equal(second.log_scores,table.log_scores)


def test_count_boundaries_padding_and_int64_large_vocabulary_offsets():
    table=static.prepare_static_count_cache(potential())
    ids=torch.tensor([[[0,-1,-1],[0,1,2],[2,-1,-1],[0,1,2]]])
    factors=dense.DenseFactors(ids,table)
    rows,pos,state=torch.tensor([0]),torch.tensor([0]),torch.tensor([0])
    assert torch.equal(factors.boundary(rows,pos,state,left_boundary=True),table.log_scores[0][None])
    assert torch.equal(factors.boundary(rows,pos+1,state,left_boundary=False),table.log_scores[:,2][None])
    assert dense.dense_pair_scores(table,torch.tensor([-1,3]),torch.tensor([1,0])).tolist()==[0.,0.]
    high=torch.tensor([50257,45000],dtype=torch.int32)
    offsets=dense.flat_pair_offsets(high,high,50258)
    assert offsets.dtype==torch.int64 and (offsets>2**31).all()
    assert offsets.tolist()==[50257*50258+50257,45000*50258+45000]


@pytest.mark.parametrize('kw',[{'max_table_bytes':35},{'max_temporary_bytes':35}])
def test_explicit_memory_budgets_fail_before_table_allocation(kw):
    with pytest.raises(MemoryError):static.prepare_static_count_cache(potential(),**kw)


def test_no_implicit_cpu_inference_or_table_preparation():
    p=potential();ids=torch.arange(3)[None,None].expand(1,3,3)
    c=CompactCandidates(ids,torch.zeros_like(ids,dtype=torch.float64),3)
    for function in (static.sample_static_count,static.static_count_marginals):
        with pytest.raises(ValueError,match='explicit'):function(c,p,None)
        with pytest.raises(ValueError,match='NVIDIA CUDA'):function(c,p,static.prepare_static_count_cache(p))


def test_plain_package_import_does_not_load_optional_static_or_triton():
    code="import sys, additive_chain; assert 'additive_chain.static' not in sys.modules; assert 'triton' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='Static count parity requires actual NVIDIA CUDA')


@cuda
@pytest.mark.parametrize('width,length',[(3,7),(500,5),(1000,4),(2000,3),(16,1024)])
def test_cuda_static_count_retains_same_noise_draws_and_normalized_marginals(width,length):
    vocab=width+2
    left=torch.linspace(.6,.9,vocab,device='cuda',dtype=torch.float64)
    right=torch.linspace(.7,.4,vocab,device='cuda',dtype=torch.float64)
    keys=torch.tensor([1,vocab,vocab*vocab-2],device='cuda')
    p=SparseCountPotential(left,right,keys,torch.tensor([.3,.6,.2],device='cuda',dtype=torch.float64))
    table=static.prepare_static_count_cache(p,reserve_bytes=64*1024**2)
    ids=torch.arange(width,device='cuda')[None,None].expand(2,length,width).clone()
    unary=torch.arange(2*length*width,device='cuda',dtype=torch.float64).sin().reshape(2,length,width).log_softmax(-1)
    for pos in {0,length//2,length-1}:
        ids[1,pos]=vocab;ids[1,pos,0]=1;unary[1,pos]=-torch.inf;unary[1,pos,0]=0.
    c=CompactCandidates(ids,unary,vocab)
    before=reference.sample_nina_count(c,p,torch.Generator(device='cuda').manual_seed(202),segment_batch_size=3)
    after=static.sample_static_count(c,p,table,torch.Generator(device='cuda').manual_seed(202),segment_batch_size=3)
    assert torch.equal(before,after)
    probability=static.static_count_marginals(c,p,table,segment_batch_size=3)
    assert (probability.sum(-1)-1).abs().max()<2e-6
    assert torch.equal(probability[1,[0,length//2,length-1],0],torch.ones(3,device='cuda'))
    high=torch.tensor([50257,45000,0],dtype=torch.int32,device='cuda')
    assert torch.equal(dense.pair_offset_probe(high,high,50258),dense.flat_pair_offsets(high,high,50258))
    reference.clear_count_cache()
