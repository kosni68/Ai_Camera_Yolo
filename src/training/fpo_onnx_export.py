"""Export ONNX via fast-plate-ocr, avec contournement de deux bugs (Windows + GELU).

fast_plate_ocr 1.1.0 (cli/export.py) pose deux problemes pour produire un .onnx
exploitable, surtout sous Windows :

1. NamedTemporaryFile garde le handle ouvert puis re-ouvre son chemin en ecriture ->
   verrou Windows : `PermissionError: [Errno 13]`. On remplace NamedTemporaryFile par
   une variante qui libere le handle (cross-plateforme, inoffensif sous Linux/Colab).

2. L'architecture CCT utilise `activation: gelu`. keras 3 abaisse le GELU exact en
   `tf.math.erfc`, et tf2onnx (plafonne a l'opset 18, donc pas de Gelu natif opset 20)
   emet un operateur `Erfc` qui n'existe pas dans le standard ONNX. onnxruntime refuse
   alors de charger le modele ("No Op registered for Erfc"). Le CLI appelle
   InferenceSession de maniere inconditionnelle : il ecrit pourtant le .onnx AVANT de
   planter. On intercepte donc l'echec, on reecrit chaque `Erfc(x)` en `1 - Erf(x)`
   (identite mathematique exacte, operateurs ONNX standard), puis on valide nous-memes.

Usage (memes options que `fast-plate-ocr export`) :
    python -m src.training.fpo_onnx_export --model best.keras \
        --plate-config-file plate_config.yaml --format onnx
"""

import argparse
import os
import sys
import tempfile

import fast_plate_ocr.cli.export as _export_mod


class _ReopenableNamedTemporaryFile:
    """Substitut de NamedTemporaryFile dont le chemin peut etre rouvert (cf. bug 1)."""

    def __init__(self, suffix="", prefix="tmp", dir=None, **_kwargs):
        fd, self.name = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
        os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        try:
            os.unlink(self.name)
        except OSError:
            pass
        return False


def _output_onnx_path(argv):
    """Reproduit la logique de fast_plate_ocr pour deviner le .onnx ecrit.

    out_file = <model>.with_suffix('.onnx'), eventuellement deplace dans --save-dir.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", "-m")
    parser.add_argument("--save-dir")
    parser.add_argument("--format", "-f", default="onnx")
    known, _ = parser.parse_known_args(argv)
    if not known.model or (known.format or "onnx").lower() != "onnx":
        return None
    base = os.path.splitext(known.model)[0] + ".onnx"
    if known.save_dir:
        base = os.path.join(known.save_dir, os.path.basename(base))
    return base


def _fix_erfc(onnx_path):
    """Remplace chaque noeud Erfc par Erf + Sub(1, Erf). Retourne le nombre corrige."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph
    erfc_nodes = [n for n in graph.node if n.op_type == "Erfc"]
    if not erfc_nodes:
        return 0

    one_name = "__fpo_one_const"
    if not any(init.name == one_name for init in graph.initializer):
        graph.initializer.append(
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)
        )
    for node in list(erfc_nodes):
        x_in, y_out = node.input[0], node.output[0]
        erf_out = y_out + "__erf"
        idx = list(graph.node).index(node)
        graph.node.remove(node)
        # Erfc(x) == 1 - Erf(x)
        graph.node.insert(idx, helper.make_node("Sub", [one_name, erf_out], [y_out], name=node.name + "__sub"))
        graph.node.insert(idx, helper.make_node("Erf", [x_in], [erf_out], name=node.name + "__erf"))

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    return len(erfc_nodes)


def main():
    # Bug 1 : NamedTemporaryFile rouvrable.
    _export_mod.NamedTemporaryFile = _ReopenableNamedTemporaryFile

    argv = sys.argv[1:]
    out_path = _output_onnx_path(argv)

    # `export` est une commande click. standalone_mode=False : ne fait pas sys.exit et
    # laisse remonter les exceptions (dont l'echec InferenceSession lie a Erfc).
    try:
        _export_mod.export(argv, standalone_mode=False)
    except SystemExit:
        raise
    except Exception:
        # Bug 2 : l'export a (probablement) ecrit le .onnx avant de planter a la
        # validation a cause d'Erfc. On tente la correction du graphe.
        if not (out_path and os.path.isfile(out_path)):
            raise
        fixed = _fix_erfc(out_path)
        if fixed == 0:
            raise  # echec sans rapport avec Erfc -> on propage l'erreur d'origine
        # Re-validation cote onnxruntime, comme le faisait le CLI.
        import onnxruntime as rt

        rt.InferenceSession(out_path)
        print(f"[EXPORT] {fixed} noeud(s) Erfc remplaces par (1 - Erf) ; ONNX valide : {out_path}")


if __name__ == "__main__":
    main()
