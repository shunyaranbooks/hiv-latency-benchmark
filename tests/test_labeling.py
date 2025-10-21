from hivlat.data.labeling import label_from_column

def test_labeling_rules():
    assert label_from_column('Latency_model_untreated_cell_1') == 'latent'
    assert label_from_column('Latency_model_SAHA_cell_5') == 'inducible'
    assert label_from_column('Latency_model_TCR_cell_9') == 'productive'
    assert label_from_column('Donor1_untreated_cell_2') == 'donor_untreated'
    assert label_from_column('Donor3_TCR_cell_8') == 'donor_tcr'
