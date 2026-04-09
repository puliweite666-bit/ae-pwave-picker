import os
import h5py
import pandas as pd

PROJECT_ROOT = r"E:\project"
MINUSIL_DIR = os.path.join(PROJECT_ROOT, "data", "Minusil")
H5_PATH = os.path.join(MINUSIL_DIR, "Minusil_dataset.h5")

CANDIDATE_TABLES = [
    os.path.join(PROJECT_ROOT, "data", "S102_data.csv"),
    os.path.join(MINUSIL_DIR, "quake_phase.csv"),
    os.path.join(MINUSIL_DIR, "quake_phaseNOISE.csv"),
]

TRAIN_LIST = os.path.join(MINUSIL_DIR, "data_list_train.txt")


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def list_h5_objects(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        if len(obj.attrs) > 0:
            print(f"          attrs={list(obj.attrs.keys())}")
    elif isinstance(obj, h5py.Group):
        print(f"[GROUP  ] {name}")
        if len(obj.attrs) > 0:
            print(f"          attrs={list(obj.attrs.keys())}")


def inspect_h5_structure():
    print_header("1) H5 全结构（含 attrs）")
    with h5py.File(H5_PATH, "r") as f:
        f.visititems(list_h5_objects)


def inspect_first_train_keys(n=10):
    print_header("2) 训练清单前几个 key")
    with open(TRAIN_LIST, "r", encoding="utf-8", errors="ignore") as f:
        keys = [x.strip() for x in f.readlines() if x.strip()]

    keys = keys[:n]
    for i, k in enumerate(keys):
        print(f"{i:02d}: {k}")

    print_header("3) 检查这些 key 在 H5 中的 dataset 和 attrs")
    with h5py.File(H5_PATH, "r") as f:
        for i, k in enumerate(keys):
            print("-" * 100)
            print(f"[{i:02d}] key = {k}")
            if k not in f:
                print("  !! 不在 H5 里")
                continue
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(f"  shape={obj.shape}, dtype={obj.dtype}")
                if len(obj.attrs) > 0:
                    print(f"  attrs keys = {list(obj.attrs.keys())}")
                    for ak in obj.attrs.keys():
                        try:
                            print(f"    - {ak}: {obj.attrs[ak]}")
                        except Exception as e:
                            print(f"    - {ak}: <无法打印> {e}")
                else:
                    print("  attrs = None")
            else:
                print(f"  object type = {type(obj)}")


def inspect_candidate_tables():
    for table_path in CANDIDATE_TABLES:
        print_header(f"4) 检查表格文件: {table_path}")
        if not os.path.isfile(table_path):
            print("文件不存在")
            continue

        try:
            if table_path.lower().endswith(".csv"):
                df = pd.read_csv(table_path)
            else:
                df = pd.read_excel(table_path)
        except Exception as e:
            print(f"读取失败: {e}")
            continue

        print("列名：")
        print(list(df.columns))

        print("\n前5行：")
        print(df.head())

        # 查可疑列
        suspect_cols = []
        for c in df.columns:
            c_low = str(c).lower()
            if any(x in c_low for x in [
                "pick", "arrival", "ptime", "phase", "p_arrival",
                "p_pick", "label", "time", "sample", "event", "trace", "key"
            ]):
                suspect_cols.append(c)

        print("\n可疑列：")
        print(suspect_cols if suspect_cols else "未发现明显可疑列")


def main():
    if not os.path.isfile(H5_PATH):
        raise FileNotFoundError(H5_PATH)
    if not os.path.isfile(TRAIN_LIST):
        raise FileNotFoundError(TRAIN_LIST)

    inspect_h5_structure()
    inspect_first_train_keys(n=10)
    inspect_candidate_tables()


if __name__ == "__main__":
    main()