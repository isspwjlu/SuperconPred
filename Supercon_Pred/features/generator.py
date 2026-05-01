"""从化学式生成材料特征向量。"""
import os
import numpy as np
import pandas as pd
from chemlib import Element
from ..utils import ensure_dir
from .atoms import Atoms


class FeatureGenerator:
    """特征生成器：从化合物列表计算 43 维材料特征向量。"""

    NUMERIC_CHARACTS = ['radius', 'mass', 'valence', 'electronnegativity', 'firstionization']
    SPECIAL_CHARACTS = ['shell_range', 'd_mean', 'd_ratio', 'd_unfill']
    STAT_NAMES = ['mean', 'mean_wtd', 'std', 'std_wtd', 'range', 'range_wtd', 'entropy', 'entropy_wtd']
    MAX_ELECTRONS = {'s': 2, 'p': 6, 'd': 10, 'f': 14}

    def __init__(self):
        self.dict_atoms = Atoms.get_dict()

    @staticmethod
    def _get_element_data(charact: str, element: str, dict_atoms: dict) -> float | str:
        """获取元素指定属性的数值或电子配置字符串。"""
        if charact == 'mass':
            return Element(element).properties['AtomicMass']
        if charact == 'firstionization':
            return Element(element).properties['FirstIonization']
        if charact in ('shell_range', 'd_mean', 'd_ratio', 'd_unfill'):
            val = Element(element).properties['Config']
            return val[val.index(']') + 2:] if ']' in val else val
        if charact == 'radius':
            return dict_atoms[element][1]
        if charact == 'electronnegativity':
            return dict_atoms[element][2]
        if charact == 'valence':
            return dict_atoms[element][3]
        raise ValueError(f"Unknown character: {charact}")

    @staticmethod
    def _split_shell_valence(s: str) -> tuple[list, dict[str, float]]:
        """解析电子配置字符串，如 '3d8 4s2' -> ([3, 4], {'d': 8.0, 's': 2.0})。"""
        parts = s.split()
        nums: list[int] = []
        valence: dict[str, float] = {}
        for p in parts:
            valence[p[1]] = float(p[2:])
            n = p[0]
            nums.append(ord(n) if n.isalpha() else int(n))
        return nums, valence

    @staticmethod
    def _dict_add(d1: dict, d2: dict) -> dict:
        """两个字典相加，相同键的值相加。"""
        keys = d1.keys() | d2.keys()
        return {k: d1.get(k, 0) + d2.get(k, 0) for k in keys}

    def generate(self, input_csv: str, output_csv: str,
                 formula_col: str = 'formula', tc_col: str | None = "Tc") -> str:
        """从输入CSV读取化合物列表并生成特征CSV。

        Args:
            input_csv: 输入文件路径（必须包含 formula_col 列）
            output_csv: 输出特征文件路径
            formula_col: 化学式列名
            tc_col: Tc列名（可选）。为None时不复制Tc值。

        Returns:
            输出文件路径
        """
        try:
            df = pd.read_csv(input_csv, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding='utf-8')
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[数据] 已删除 {n_dropped} 行含缺失值的数据")
        if df.empty:
            raise ValueError(f"数据为空（所有行均含缺失值）: {input_csv}")
        formulas = df[formula_col].tolist()

        feature_names = self._build_feature_names()
        all_rows = [self._compute_features(f) for f in formulas]

        out_df = pd.DataFrame(all_rows, columns=feature_names)
        out_df.insert(0, formula_col, formulas)
        if tc_col and tc_col in df.columns:
            out_df[tc_col] = df[tc_col].values

        ensure_dir(os.path.dirname(output_csv))
        out_df.to_csv(output_csv, index=False, encoding='gbk')
        print(f"[特征] 已生成: {output_csv} ({len(formulas)} 个化合物)")
        return output_csv

    def _build_feature_names(self) -> list[str]:
        """构建特征列名列表（共43维）。"""
        names: list[str] = ['nelements']
        for c in self.NUMERIC_CHARACTS:
            for s in self.STAT_NAMES:
                names.append(f"{c}_{s}")
        names += self.SPECIAL_CHARACTS
        return names

    def _compute_features(self, formula: str) -> list:
        """计算单个化合物的完整特征向量。"""
        elem_dict = Atoms.extract_elements(formula)
        num_elem = len(elem_dict)
        num_atoms = sum(elem_dict.values())
        ratios = {e: c / num_atoms for e, c in elem_dict.items()}
        feats = [num_elem]

        # 数值特征（radius, mass, valence, etc.）
        for charact in self.NUMERIC_CHARACTS:
            vals = [self._get_element_data(charact, e, self.dict_atoms) for e in elem_dict]
            wtd = [v * elem_dict[e] / num_atoms for e, v in zip(elem_dict, vals)]
            sum_v = sum(vals)
            sum_w = sum(wtd)
            mean_v = np.mean(vals)
            mean_w = np.mean(wtd)
            std_v = np.std(vals)
            std_w = np.sqrt(sum((v - mean_w) ** 2 * ratios[e] for e, v in zip(elem_dict, vals)))
            rng_v = max(vals) - min(vals)
            rng_w = max(wtd) - min(wtd)
            ent_v = -sum(v / sum_v * np.log(v / sum_v) for v in vals if v > 0) if sum_v > 0 else 0
            ent_w = -sum(v / sum_w * np.log(v / sum_w) for v in wtd if v > 0) if sum_w > 0 else 0
            feats += [mean_v, mean_w, std_v, std_w, rng_v, rng_w, ent_v, ent_w]

        # shell_range
        all_nums: list[int] = []
        for e in elem_dict:
            cfg = self._get_element_data('shell_range', e, self.dict_atoms)
            nums, _ = self._split_shell_valence(cfg)
            all_nums.extend(nums)
        feats.append(max(all_nums) - min(all_nums))

        # d_mean, d_ratio, d_unfill
        total_shell: dict[str, float] = {}
        total_unfill: dict[str, float] = {}
        num_d = 0
        for e, c in elem_dict.items():
            cfg = self._get_element_data('d_mean', e, self.dict_atoms)
            _, sd = self._split_shell_valence(cfg)
            r = c / num_atoms
            weighted = {k: v * r for k, v in sd.items()}
            unfilled = {k: (self.MAX_ELECTRONS[k] - sd[k]) * r for k in sd}
            total_shell = self._dict_add(total_shell, weighted)
            total_unfill = self._dict_add(total_unfill, unfilled)
            if 'd' in sd:
                num_d += c

        d_mean = total_shell.get('d', 0)
        total_val = sum(total_shell.values())
        d_ratio = d_mean / total_val if total_val else 0
        d_unfill = total_unfill.get('d', 0) * num_atoms / num_d if num_d else 0
        feats += [d_mean, d_ratio, d_unfill]
        return feats
