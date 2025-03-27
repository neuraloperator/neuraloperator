from pathlib import Path
import shutil

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
    dummy_uq = UQNO(base_model=DummyModel(50), residual_model=DummyModel(50))

    checkpoint_path = Path("./test_checkpoints")
    dummy_uq.save_checkpoint(save_folder=checkpoint_path, save_name="uqno")

    for submodule in "solution", "residual":
        save_path = checkpoint_path / f"uqno_{submodule}_state_dict.pt"
        assert save_path.exists()

        metadata_path = checkpoint_path / f"uqno_{submodule}_metadata.pkl"
        assert metadata_path.exists

    from neuralop.models.base_model import BaseModel

    # temporarily add DummyModel to the BaseModel registry to allow class creation
    BaseModel._models.update({'DummyModel': DummyModel})
    dummy_uq = UQNO.from_checkpoint(checkpoint_path, "uqno")

    shutil.rmtree(checkpoint_path)