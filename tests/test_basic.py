import pytest
import cytovi
from scvi.data import synthetic_iid
from anndata import AnnData
import numpy as np



RAW_LAYER_KEY = 'raw'
SCALED_LAYER_KEY = 'scaled'
NAN_LAYER_KEY = '_nan_mask'
BATCH_KEY = 'batch'
LABELS_KEY = 'labels'
N_EPOCHS = 2
SAMPLE_KEY = 'sample_key'

@pytest.fixture(scope="session")
def adata():
    adata = synthetic_iid(batch_size=256,
                       n_genes=30,
                       n_proteins=0,
                       n_regions=0,
                       n_batches=2,
                       n_labels=10)
    
    adata.layers[RAW_LAYER_KEY] = adata.X.copy()
    adata.obs[SAMPLE_KEY] = np.random.choice(['group_a', 'group_b'], size=adata.shape[0])
    return adata

@pytest.fixture(scope="session")
def overlapping_adatas():
    adata1 = synthetic_iid(batch_size=256,
                       n_genes=30,
                       n_proteins=0,
                       n_regions=0,
                       n_batches=1,
                       n_labels=10)
    
    adata2 = synthetic_iid(batch_size=256,
                       n_genes=20,
                       n_proteins=0,
                       n_regions=0,
                       n_batches=1,
                       n_labels=10)

    adata1.layers[RAW_LAYER_KEY] = adata1.X.copy()
    adata2.layers[RAW_LAYER_KEY] = adata2.X.copy()

    return adata1, adata2

def test_cytovi_preprocess(adata, overlapping_adatas):
    cytovi.pp.logp(adata)
    cytovi.pp.arcsinh(adata)
    cytovi.pp.scale(adata)
    adata_sub = cytovi.pp.subsample(adata, n_obs=100)
    assert adata_sub.n_obs == 100

    adata1, adata2 = overlapping_adatas
    cytovi.pp.arcsinh(adata1)
    cytovi.pp.scale(adata1)
    cytovi.pp.arcsinh(adata2)
    cytovi.pp.scale(adata2)
    adata_merged = cytovi.pp.merge_batches([adata1, adata2])
    assert NAN_LAYER_KEY in adata_merged.layers


def test_cytovi(adata):
    cytovi.pp.arcsinh(adata)
    cytovi.pp.scale(adata)

    cytovi.CytoVI.setup_anndata(adata,
                                layer=SCALED_LAYER_KEY,
                                batch_key=BATCH_KEY,
                                sample_key=SAMPLE_KEY,
                                )
    
    model = cytovi.CytoVI(adata)

    model.train(max_epochs= N_EPOCHS)
    assert model.is_trained

    latent = model.get_latent_representation()
    assert latent.shape[0] == adata.n_obs

    imp_exp = model.get_normalized_expression()
    assert imp_exp.shape == adata.shape

    model.posterior_predictive_sample()
    da_res = model.differential_abundance()
    assert da_res.shape == (adata.n_obs, adata.obs[SAMPLE_KEY].nunique())



def test_cytovi_overlapping(overlapping_adatas):
    adata1, adata2 = overlapping_adatas
    cytovi.pp.arcsinh(adata1)
    cytovi.pp.scale(adata1)
    cytovi.pp.arcsinh(adata2)
    cytovi.pp.scale(adata2)
    adata_merged = cytovi.pp.merge_batches([adata1, adata2])

    cytovi.CytoVI.setup_anndata(adata_merged,
                                layer=SCALED_LAYER_KEY,
                                batch_key=BATCH_KEY,
                                )
    
    model = cytovi.CytoVI(adata_merged)

    model.train(max_epochs= N_EPOCHS)

    assert model.is_trained
