#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.fc(x))


def graph_text_with_shapes(fx_model: torch.nn.Module, example_input: torch.Tensor) -> str:
    ShapeProp(fx_model).propagate(example_input)
    lines = [f"# example_input: {tuple(example_input.shape)}", ""]
    for node in fx_model.graph.nodes:
        shape = ""
        if tensor_meta := node.meta.get("tensor_meta"):
            shape = str(tuple(tensor_meta.shape))
        lines.append(f"{node.op:14} {node.name:8} {str(node.target):32} {shape}")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "fx_out"
    out_dir.mkdir(exist_ok=True)

    model = TinyModel()
    fx_model = symbolic_trace(model)
    x = torch.randn(2, 4)

    assert torch.allclose(model(x), fx_model(x))

    graph_path = out_dir / "model.fx.txt"
    module_path = out_dir / "model.fx.pt"
    graph_path.write_text(graph_text_with_shapes(fx_model, x), encoding="utf-8")
    torch.save(fx_model, module_path)

    print(graph_path.read_text(encoding="utf-8"))
    print(f"saved graph -> {graph_path}")
    print(f"saved module -> {module_path}")
