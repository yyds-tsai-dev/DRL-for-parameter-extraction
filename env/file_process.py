import shutil
import os
import zipfile
import io
import importlib.util
import json
import pandas as pd

def zip_result(folder_path):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 將 zip 中的路徑改為相對於資料夾根目錄的路徑
                arcname = os.path.relpath(file_path, start=folder_path)
                zip_file.write(file_path, arcname=arcname)

    zip_buffer.seek(0)  
    return zip_buffer

def zip_folder_and_feature(folder_path, feature_file_path, output_zip_path, extra_file_paths=None):
    extra_file_paths = extra_file_paths or []
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # model files
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))  # 讓zip裡有完整資料夾層級
                zipf.write(file_path, arcname=arcname)
        
        # feature file
        if os.path.exists(feature_file_path):
            feature_arcname = os.path.join("features", os.path.basename(feature_file_path))  # 放到 zip 裡一個 features/ 資料夾
            zipf.write(feature_file_path, arcname=feature_arcname)

        for extra_file_path in extra_file_paths:
            if os.path.exists(extra_file_path):
                extra_arcname = os.path.join("metadata", os.path.basename(extra_file_path))
                zipf.write(extra_file_path, arcname=extra_arcname)


def zip_folder_contents(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname=arcname)


def unzip_strip_top_level(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # 找出 zip 中所有的路徑
        members = zip_ref.namelist()
        
        # 自動偵測 top-level 資料夾名稱
        top_level_dir = os.path.commonprefix(members).split("/")[0]

        for member in members:
            target_path = member
            if member.startswith(top_level_dir + "/"):
                target_path = member[len(top_level_dir)+1:]  # 去掉 top 資料夾名稱

            if target_path:
                src_path = os.path.join(extract_to, member)
                dest_path = os.path.join(extract_to, target_path)
                # 確保目標資料夾存在
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                zip_ref.extract(member, extract_to)

                # # 如果目標路徑已經存在則刪除
                # if os.path.exists(dest_path):
                #     if os.path.isfile(dest_path):
                #         os.remove(dest_path)
                #     elif os.path.isdir(dest_path):
                #         shutil.rmtree(dest_path)

            
                os.rename(src_path, dest_path)


def find_file(root_path, filename):
    for root, _, files in os.walk(root_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_model_folder(root_path):
    for root, dirs, _ in os.walk(root_path):
        if "committee_models" in dirs and os.path.exists(os.path.join(root, "committee_scalers.pkl")):
            return root
    return None


def load_model_package(root_path):
    metadata_path = find_file(root_path, "training_config.json")
    features_path = find_file(root_path, "features.json")
    model_folder = find_model_folder(root_path)

    if metadata_path is None:
        raise FileNotFoundError("training_config.json was not found in the uploaded model package.")
    if features_path is None:
        raise FileNotFoundError("features.json was not found in the uploaded model package.")
    if model_folder is None:
        raise FileNotFoundError("Could not find committee_models and committee_scalers.pkl in the uploaded model package.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type") or metadata.get("model", {}).get("type")
    if not model_type:
        raise ValueError("Model type was not found in training_config.json.")

    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "features_path": features_path,
        "model_folder": model_folder,
        "model_type": model_type,
    }


def apply_package_feature_encoding(df, metadata):
    preprocessing = metadata.get("preprocessing", {})
    encoding = preprocessing.get("categorical_encoding") or {}
    model_features = metadata.get("features", {}).get("model_features") or []
    original_features = metadata.get("features", {}).get("original_features") or model_features
    categorical_features = encoding.get("categorical_features") or []
    method = encoding.get("method")

    df = df.copy()

    def coerce_model_features_numeric(encoded_df):
        for feature in model_features:
            if feature in encoded_df.columns:
                if encoded_df[feature].dtype == bool:
                    encoded_df[feature] = encoded_df[feature].astype(float)
                else:
                    try:
                        encoded_df[feature] = pd.to_numeric(encoded_df[feature]).astype(float)
                    except Exception:
                        pass
        return encoded_df

    if not categorical_features:
        return coerce_model_features_numeric(df)

    if method == "One-Hot Encoding":
        one_hot_features = encoding.get("one_hot_features") or []
        missing_raw_features = []
        for feature in categorical_features:
            already_encoded = any(col in df.columns for col in one_hot_features if col.startswith(f"{feature}_"))
            if feature not in df.columns and not already_encoded:
                missing_raw_features.append(feature)
        if missing_raw_features:
            raise ValueError(
                "Missing categorical feature columns required for one-hot encoding: "
                + ", ".join(missing_raw_features)
            )

        df = pd.get_dummies(df, columns=[c for c in categorical_features if c in df.columns], prefix=categorical_features)
        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0
        return coerce_model_features_numeric(df)

    if method == "Label Encoding":
        for feature, mapping in (encoding.get("mappings") or {}).items():
            if feature in df.columns:
                df[feature] = df[feature].astype(str).map(mapping).fillna(-1).astype(float)
        return coerce_model_features_numeric(df)

    if method == "Target Encoding":
        global_mean = encoding.get("global_mean", 0.0)
        for feature, mapping in (encoding.get("mappings") or {}).items():
            if feature in df.columns:
                df[feature] = df[feature].astype(str).map(mapping).fillna(global_mean).astype(float)
        return coerce_model_features_numeric(df)

    return coerce_model_features_numeric(df)

def load_feature_and_target(path="datasets/temp_opt_feature.py"):
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            features_dict = json.load(f)
    else:
        spec = importlib.util.spec_from_file_location("feature_config", path)
        feature_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_config)
        features_dict = feature_config.features

    return {
        "features": features_dict["features"],
        "targets": features_dict["target"]
    }
