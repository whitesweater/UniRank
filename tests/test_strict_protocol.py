from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from scripts.audit_sisa_expansion_baselines import EXPERIMENTS
from unirank.utils import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPID = REPOSITORY_ROOT / "run_expid.py"
STRICT_SCRIPT = REPOSITORY_ROOT / "scripts" / "submit_sisa_native_strict.sbatch"
EXPANSION_SCRIPT = REPOSITORY_ROOT / "scripts" / "submit_sisa_expansion.sbatch"
GATE_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "gate_onetrans_taobao_calibration.sbatch"
)


class StrictProtocolTest(unittest.TestCase):
    def test_completed_runs_keep_best_checkpoint_by_default(self):
        source = RUN_EXPID.read_text(encoding="utf-8")
        self.assertIn("Preserved best model checkpoint", source)
        self.assertNotIn("model.delete_checkpoint()", source)
        self.assertNotIn("--delete-checkpoint-after-test", source)

    def test_expansion_configs_match_official_benchmark_parameters(self):
        ignored = {
            "gpu",
            "rank",
            "local_rank",
            "world_size",
            "distributed",
            "model_id",
            "data_root",
            "model_root",
            "train_data",
            "valid_data",
            "test_data",
            "train_user_info",
            "train_item_info",
            "valid_user_info",
            "valid_item_info",
            "test_user_info",
            "test_item_info",
            "enable_torch_compile",
            "gradient_checkpointing",
        }
        decoder = json.JSONDecoder()
        for experiment in EXPERIMENTS:
            model, dataset = experiment.split("_", 1)
            benchmark = (
                REPOSITORY_ROOT
                / "benchmark"
                / dataset
                / f"{model}_{dataset}.log"
            ).read_text(encoding="utf-8", errors="replace")
            marker = "INFO Params: "
            object_start = benchmark.index("{", benchmark.index(marker) + len(marker))
            official = decoder.raw_decode(benchmark[object_start:])[0]
            current = load_config(str(REPOSITORY_ROOT / "config"), experiment)

            with self.subTest(experiment=experiment):
                for name in sorted((set(official) & set(current)) - ignored):
                    self.assertEqual(str(current[name]), str(official[name]), name)

    def test_all_matrix_configs_match_paper_batch_protocol(self):
        models = ("OneTrans", "HiFormer", "RankMixer", "Zenith")
        datasets = (
            "QK_Video_Action",
            "KuaiRand_Video_Action",
            "Taobao_Action",
            "MerRec_Action",
        )
        expected = {
            "batch_size": 8192,
            "accumulation_steps": 1,
            "epochs": 1,
            "seed": 20262027,
            "max_len": 100,
        }
        for model in models:
            for dataset in datasets:
                experiment = f"{model}_{dataset}"
                with self.subTest(experiment=experiment):
                    params = load_config(str(REPOSITORY_ROOT / "config"), experiment)
                    for name, value in expected.items():
                        self.assertEqual(params.get(name), value)

    def test_strict_launcher_uses_four_processes_and_preserves_slurm_devices(self):
        source = STRICT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:rtx4090:4", source)
        self.assertIn("--nproc_per_node=4", source)
        self.assertIn("--gpu 0,1,2,3", source)
        self.assertIn("global_batch=32768", source)
        self.assertIn(
            '[[ "$model" == Zenith && "$dataset" == MerRec_Action ]]',
            source,
        )
        self.assertIn("--sparse-optimizer-foreach false", source)
        self.assertIn("--sparse-adagrad-chunk-size 16777216", source)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", source)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", source)

    def test_strict_launcher_has_valid_bash_syntax(self):
        for script in (STRICT_SCRIPT, EXPANSION_SCRIPT, GATE_SCRIPT):
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_expansion_matrix_and_launcher_match_protocol(self):
        experiments = (
            "OneTrans_TencentGR_10M_Action",
            "HiFormer_TencentGR_10M_Action",
            "RankMixer_TencentGR_10M_Action",
            "Zenith_TencentGR_10M_Action",
            "UniMixer_QK_Video_Action",
            "UniMixer_KuaiRand_Video_Action",
            "UniMixer_TencentGR_10M_Action",
            "UniMixer_Taobao_Action",
            "UniMixer_MerRec_Action",
            "HyFormer_QK_Video_Action",
            "HyFormer_KuaiRand_Video_Action",
            "HyFormer_TencentGR_10M_Action",
            "HyFormer_Taobao_Action",
            "HyFormer_MerRec_Action",
            "UltraHSTU_QK_Video_Action",
            "UltraHSTU_KuaiRand_Video_Action",
            "UltraHSTU_TencentGR_10M_Action",
            "UltraHSTU_Taobao_Action",
            "UltraHSTU_MerRec_Action",
        )
        expected = {
            "batch_size": 8192,
            "accumulation_steps": 1,
            "epochs": 1,
            "seed": 20262027,
            "max_len": 100,
        }
        for experiment in experiments:
            with self.subTest(experiment=experiment):
                params = load_config(str(REPOSITORY_ROOT / "config"), experiment)
                for name, value in expected.items():
                    self.assertEqual(params.get(name), value)

        source = EXPANSION_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-37%1", source)
        self.assertIn("#SBATCH --gres=gpu:rtx4090:4", source)
        self.assertIn("--nproc_per_node=4", source)
        self.assertIn("global_batch=32768", source)
        self.assertIn("SISA_EXPANSION_COMPLETE", source)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", source)


if __name__ == "__main__":
    unittest.main()
