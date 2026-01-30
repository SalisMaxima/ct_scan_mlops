from pathlib import Path

import yaml  # type: ignore[import-untyped]


def convert_wb_config(input_path, output_path):
    with Path(input_path).open() as f:
        wb_config = yaml.safe_load(f)

    clean_config = {}
    for key, val in wb_config.items():
        if isinstance(val, dict) and "value" in val:
            clean_config[key] = val["value"]
        else:
            clean_config[key] = val

    # If 'cfg' exists and is a string (some exports have this), it might be more complete
    if "cfg" in clean_config and isinstance(clean_config["cfg"], str):
        import ast

        try:
            # The cfg string in the file I saw was actually a string representation of a dict
            inner_cfg = ast.literal_eval(clean_config["cfg"])
            clean_config.update(inner_cfg)
            del clean_config["cfg"]
        except Exception as e:
            print(f"Warning: Could not parse inner 'cfg' string: {e}")

    # Ensure model name is correct for the registry
    if "model" in clean_config and isinstance(clean_config["model"], dict):
        if "name" in clean_config["model"]:
            name = clean_config["model"]["name"]
            if name == "dual_pathway_top_features":
                clean_config["model"]["name"] = "dual_pathway"
    elif "model" in clean_config and isinstance(clean_config["model"], str):
        # If it's just a string, we might need a dict
        model_name = clean_config["model"]
        if model_name == "dual_pathway_top_features":
            model_name = "dual_pathway"

        # We need more info for the model dict if it's just a string
        # Let's try to reconstruct it from other keys if they exist
        clean_config["model"] = {
            "name": model_name,
            "num_classes": clean_config.get("num_classes", 4),
            "radiomics_dim": clean_config.get(
                "radiomics_dim", 16 if clean_config.get("features") == "top_features" else 50
            ),
            "radiomics_hidden": clean_config.get("radiomics_hidden", 512),
            "cnn_feature_dim": clean_config.get("cnn_feature_dim", 512),
            "fusion_hidden": clean_config.get("fusion_hidden", 256),
            "dropout": clean_config.get("dropout", 0.05),
            "pretrained": clean_config.get("pretrained", True),
            "freeze_backbone": clean_config.get("freeze_backbone", False),
        }

    with Path(output_path).open("w") as f:
        yaml.dump(clean_config, f, default_flow_style=False)

    print(f"Converted {input_path} to {output_path}")


if __name__ == "__main__":
    convert_wb_config(
        "configs/model/dual_pathway_bn_finetune_kygevxv0.yaml",
        "configs/model/dual_pathway_bn_finetune_kygevxv0_clean.yaml",
    )
