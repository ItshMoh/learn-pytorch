"""Verify reference labs, learning invariants, and local Markdown links.

Run from any directory with the intended Python interpreter. Does not modify
learner exercises or leave checkpoints in the course folder.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "labs"))


def load_lab(name):
    path = ROOT / "labs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_reference_checks():
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from common import Classifier, make_splits, train_classifier, evaluate

    torch.set_num_threads(1)
    results = {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "cpu"}
    for number in range(1, 11):
        import solutions
        from check_exercises import check
        check(number, solutions)
    results["reference_exercises"] = 10
    model, loss = load_lab("04_modules").fit()
    torch.testing.assert_close(model.weight, torch.tensor(3.0), rtol=0, atol=1e-4)
    torch.testing.assert_close(model.bias, torch.tensor(-0.5), rtol=0, atol=1e-5)
    results["regression_loss"] = loss
    load_lab("09_debugging").accumulation_demo()
    load_lab("11_attention").verify()
    model, _, history, (_, _, test) = train_classifier()
    metrics = evaluate(model, DataLoader(test, batch_size=64), torch.device("cpu"))
    assert metrics["accuracy"] > 0.85, metrics
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    results["classifier_test"] = metrics
    results["classifier_first_train_loss"] = history[0]["train_loss"]
    results["classifier_last_train_loss"] = history[-1]["train_loss"]
    # Independent tiny-batch training sanity check, with dropout disabled.
    torch.manual_seed(55)
    train, _, _ = make_splits()
    tiny = Classifier(dropout=0.0)
    tiny.fit_preprocessing(train.x[:16])
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=0.03, weight_decay=0)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(tiny(train.x[:16]), train.y[:16])
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.03, loss.item()
    results["tiny_batch_loss"] = loss.item()

    with tempfile.TemporaryDirectory(prefix="pytorch-course-check-") as temp:
        temp = Path(temp)
        load_lab("08_checkpoints").roundtrip(temp / "classifier.pt")
        results["classifier_checkpoint_and_fixed_next_step"] = "exact match"
        lm = load_lab("12_language_model")
        net, lm_metrics = lm.train(steps=300, output=temp / "lm.pt")
        assert lm_metrics["validation_after"] < lm_metrics["validation_before"] * 0.8
        results["language_model"] = lm_metrics
        checkpoint = torch.load(temp / "lm.pt", map_location="cpu", weights_only=True)
        assert checkpoint["characters"] == lm.CHARS
        restored = lm.TinyLM(**checkpoint["config"])
        restored.load_state_dict(checkpoint["model"])
        net.eval()
        restored.eval()
        ids = torch.tensor([[lm.CHARS.index(c) for c in "the red "]])
        with torch.no_grad():
            torch.testing.assert_close(net(ids), restored(ids), rtol=0, atol=0)
        # Generation must keep working when total prompt+output exceeds context.
        generated = lm.generate(restored, prompt="the ", count=40)
        assert len(generated) == 44
        results["sample"] = generated
        # Causality is checked again at the full-model level, including FFN/norm.
        source = torch.randint(0, len(lm.CHARS), (2, 12))
        changed = source.clone()
        changed[:, 7:] = torch.randint(0, len(lm.CHARS), (2, 5))
        with torch.no_grad():
            torch.testing.assert_close(restored(source)[:, :7], restored(changed)[:, :7])
        results["lm_checkpoint_and_causality"] = "passed"
    return results


def run_scripts():
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    scripts = sorted((ROOT / "labs").glob("[0-9][0-9]_*.py"))
    with tempfile.TemporaryDirectory(prefix="pytorch-course-labs-") as temp:
        for script in scripts:
            args = [sys.executable, str(script)]
            if script.name.startswith("08_"):
                args += ["--output", str(Path(temp) / "classifier.pt")]
            elif script.name.startswith("12_"):
                args += ["--steps", "20", "--output", str(Path(temp) / "lm.pt")]
            result = subprocess.run(args, cwd=ROOT, env=environment, text=True,
                                    capture_output=True, timeout=90)
            if result.returncode:
                raise AssertionError(f"{script.name} failed:\n{result.stdout}\n{result.stderr}")
            print("Script passed:", script.name, flush=True)
    # Exercise starters must remain unfinished; these are learning tasks.
    starter = subprocess.run([sys.executable, str(ROOT / "check_exercises.py"), "--exercise", "1"],
                              cwd=ROOT, env=environment, capture_output=True, text=True)
    assert starter.returncode == 1 and "NotImplementedError" in starter.stdout
    return len(scripts)


def check_links():
    count = 0
    for path in ROOT.rglob("*.md"):
        if ".venv" in path.parts:
            continue
        for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if "://" in destination or destination.startswith("#"):
                continue
            target = destination.split("#", 1)[0]
            assert (path.parent / target).exists(), f"Broken local link in {path}: {destination}"
            count += 1
    return count


def main():
    results = run_reference_checks()
    print("Reference checks passed", flush=True)
    results["standalone_scripts"] = run_scripts()
    results["local_links"] = check_links()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
