"""
ML-based rule generation module.
Uses machine learning to generate and order rules by performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import json


class Rule:
    """Represents a single classification rule"""

    def __init__(self, condition: str, prediction: int,
                 precision: float, recall: float, coverage: float,
                 rule_type: str = 'simple'):
        self.condition = condition
        self.prediction = prediction
        self.precision = precision
        self.recall = recall
        self.coverage = coverage
        self.f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        self.score = self.f1_score * coverage  # Combined score
        self.rule_type = rule_type

    def __repr__(self):
        return f"Rule(IF {self.condition} THEN {self.prediction}, " \
               f"P={self.precision:.3f}, R={self.recall:.3f}, Cov={self.coverage:.3f}, Score={self.score:.3f})"

    def to_dict(self):
        return {
            'condition': self.condition,
            'prediction': self.prediction,
            'precision': self.precision,
            'recall': self.recall,
            'coverage': self.coverage,
            'f1_score': self.f1_score,
            'score': self.score,
            'rule_type': self.rule_type
        }


class MLRuleGenerator:
    """Generate and optimize classification rules using ML"""

    def __init__(self, min_support: int = 10, min_precision: float = 0.6):
        self.min_support = min_support
        self.min_precision = min_precision
        self.rules = []

    def generate_tld_rules(self, df: pd.DataFrame, tld_col: str = 'tld',
                          label_col: str = 'label') -> List[Rule]:
        """Generate rules based on TLD analysis"""
        rules = []

        tld_stats = df.groupby(tld_col).agg({
            label_col: ['count', 'sum', 'mean']
        })
        tld_stats.columns = ['total_count', 'malicious_count', 'malicious_ratio']

        for tld, row in tld_stats.iterrows():
            if row['total_count'] < self.min_support:
                continue

            # High-risk TLD rule (predicts malicious)
            if row['malicious_ratio'] >= self.min_precision:
                mask = df[tld_col] == tld
                precision = row['malicious_ratio']
                # Recall: how many malicious domains with this TLD we catch
                recall = row['malicious_count'] / df[df[label_col] == 1].shape[0]
                coverage = row['total_count'] / len(df)

                rule = Rule(
                    condition=f"tld == '{tld}'",
                    prediction=1,
                    precision=precision,
                    recall=recall,
                    coverage=coverage,
                    rule_type='tld'
                )
                rules.append(rule)

            # Low-risk TLD rule (predicts legitimate)
            elif row['malicious_ratio'] <= (1 - self.min_precision):
                precision = 1 - row['malicious_ratio']
                recall = (row['total_count'] - row['malicious_count']) / df[df[label_col] == 0].shape[0]
                coverage = row['total_count'] / len(df)

                rule = Rule(
                    condition=f"tld == '{tld}'",
                    prediction=0,
                    precision=precision,
                    recall=recall,
                    coverage=coverage,
                    rule_type='tld'
                )
                rules.append(rule)

        return rules

    def generate_threshold_rules(self, df: pd.DataFrame, feature_col: str,
                                 label_col: str = 'label',
                                 thresholds: List[float] = None) -> List[Rule]:
        """Generate threshold-based rules for numerical features"""
        rules = []

        if thresholds is None:
            # Auto-generate thresholds using percentiles
            thresholds = df[feature_col].quantile([0.25, 0.5, 0.75, 0.9, 0.95]).values

        for threshold in thresholds:
            # Rule: feature > threshold
            mask_above = df[feature_col] > threshold
            if mask_above.sum() >= self.min_support:
                malicious_above = df[mask_above][label_col].sum()
                total_above = mask_above.sum()
                precision = malicious_above / total_above if total_above > 0 else 0

                if precision >= self.min_precision:
                    recall = malicious_above / df[df[label_col] == 1].shape[0]
                    coverage = total_above / len(df)

                    rule = Rule(
                        condition=f"{feature_col} > {threshold:.3f}",
                        prediction=1,
                        precision=precision,
                        recall=recall,
                        coverage=coverage,
                        rule_type='threshold'
                    )
                    rules.append(rule)

            # Rule: feature <= threshold
            mask_below = df[feature_col] <= threshold
            if mask_below.sum() >= self.min_support:
                legitimate_below = (df[mask_below][label_col] == 0).sum()
                total_below = mask_below.sum()
                precision = legitimate_below / total_below if total_below > 0 else 0

                if precision >= self.min_precision:
                    recall = legitimate_below / df[df[label_col] == 0].shape[0]
                    coverage = total_below / len(df)

                    rule = Rule(
                        condition=f"{feature_col} <= {threshold:.3f}",
                        prediction=0,
                        precision=precision,
                        recall=recall,
                        coverage=coverage,
                        rule_type='threshold'
                    )
                    rules.append(rule)

        return rules

    def generate_decision_tree_rules(self, X: pd.DataFrame, y: pd.Series,
                                     max_depth: int = 10,
                                     min_samples_leaf: int = 20) -> List[Rule]:
        """Extract rules from a decision tree"""
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=self.min_support * 2,
            random_state=42
        )
        dt.fit(X, y)

        rules = []
        feature_names = list(X.columns)

        def extract_rules_from_tree(tree, node_id=0, condition=''):
            """Recursively extract rules from decision tree"""
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]

                # Left child (<=)
                left_condition = f"{condition} AND {feature} <= {threshold:.3f}" if condition else f"{feature} <= {threshold:.3f}"
                extract_rules_from_tree(tree, tree.children_left[node_id], left_condition)

                # Right child (>)
                right_condition = f"{condition} AND {feature} > {threshold:.3f}" if condition else f"{feature} > {threshold:.3f}"
                extract_rules_from_tree(tree, tree.children_right[node_id], right_condition)
            else:
                # Leaf node - create rule
                samples = tree.n_node_samples[node_id]
                values = tree.value[node_id][0]
                prediction = 1 if values[1] > values[0] else 0

                if prediction == 1:
                    precision = values[1] / samples if samples > 0 else 0
                    recall = values[1] / (y == 1).sum()
                else:
                    precision = values[0] / samples if samples > 0 else 0
                    recall = values[0] / (y == 0).sum()

                coverage = samples / len(y)

                if precision >= self.min_precision and samples >= self.min_support:
                    rule = Rule(
                        condition=condition,
                        prediction=prediction,
                        precision=precision,
                        recall=recall,
                        coverage=coverage,
                        rule_type='decision_tree'
                    )
                    rules.append(rule)

        extract_rules_from_tree(dt.tree_)
        return rules

    def optimize_rule_order(self, rules: List[Rule]) -> List[Rule]:
        """
        Order rules by performance metrics.
        Priority: F1-score * coverage (balances precision, recall, and coverage)
        """
        return sorted(rules, key=lambda r: r.score, reverse=True)

    def generate_all_rules(self, df: pd.DataFrame, feature_columns: List[str],
                          label_col: str = 'label') -> List[Rule]:
        """
        Generate comprehensive rule set from all methods.
        Returns optimally ordered rules.
        """
        all_rules = []

        # 1. TLD-based rules
        if 'tld' in df.columns:
            tld_rules = self.generate_tld_rules(df)
            all_rules.extend(tld_rules)
            print(f"Generated {len(tld_rules)} TLD rules")

        # 2. Threshold-based rules for numerical features
        numerical_features = df[feature_columns].select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            threshold_rules = self.generate_threshold_rules(df, feature, label_col)
            all_rules.extend(threshold_rules)
        print(f"Generated {len([r for r in all_rules if r.rule_type == 'threshold'])} threshold rules")

        # 3. Decision tree rules
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[label_col]
        dt_rules = self.generate_decision_tree_rules(X, y)
        all_rules.extend(dt_rules)
        print(f"Generated {len(dt_rules)} decision tree rules")

        # Optimize order
        optimized_rules = self.optimize_rule_order(all_rules)
        print(f"\nTotal rules generated: {len(optimized_rules)}")

        self.rules = optimized_rules
        return optimized_rules

    def save_rules(self, filepath: str):
        """Save rules to JSON file"""
        rules_data = [rule.to_dict() for rule in self.rules]
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
        print(f"Rules saved to {filepath}")

    def load_rules(self, filepath: str):
        """Load rules from JSON file"""
        with open(filepath, 'r') as f:
            rules_data = json.load(f)

        self.rules = [
            Rule(
                condition=r['condition'],
                prediction=r['prediction'],
                precision=r['precision'],
                recall=r['recall'],
                coverage=r['coverage'],
                rule_type=r['rule_type']
            )
            for r in rules_data
        ]
        print(f"Loaded {len(self.rules)} rules from {filepath}")


if __name__ == '__main__':
    # Test rule generator
    print("Rule generator module loaded successfully")
