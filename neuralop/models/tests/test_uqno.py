from pathlib import Path
import shutil

import torch

from neuralop.tests.test_utils import DummyModel
from ..uqno import UQNO

def test_uqno_train_eval():
    dummy_uq = UQNO(base_model=DummyModel(50), residual_model=DummyModel(50))
    

    dummy_uq.train()
    assert dummy_uq.base_model.training
    assert dummy_uq.residual_model.training

    dummy_uq.eval()
    assert not dummy_uq.base_model.training
    assert not dummy_uq.residual_model.training

def test_uqno_checkpoint():
    soln_model = DummyModel(50)
    resid_model = DummyModel(50)
    dummy_uq = UQNO(base_model=soln_model, residual_model=resid_model)

    checkpoint_path = Path("./test_checkpoints")
    torch.save(dummy_uq.state_dict(), checkpoint_path / "uqno.pt"), 

    from neuralop.models.base_model import BaseModel

    # temporarily add DummyModel to the BaseModel registry to allow class creation
    #BaseModel._models.update({'DummyModel': DummyModel})

    dummy_uq = UQNO.from_checkpoint(checkpoint_path / "uqno.pt")

    shutil.rmtree(checkpoint_path)